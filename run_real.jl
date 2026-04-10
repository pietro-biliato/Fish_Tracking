# ============================================================================
# Fish Tracking Analysis with Parallel Tempering (Execution Script)
#
# April 10, 2026 -- Sebastian Waruszynski
# sebastian.waruszynski@studenti.unipd.it
# ============================================================================

using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Load required packages
using CSV, DataFrames, Dates
using Plots, Statistics
using TransformVariables, Random
using LogDensityProblems
using Pigeons
using Pigeons: round_trip, traces, record_default, online, index_process, swap_acceptance_pr

# Include the newly refactored functions file
include("real_functs.jl")

rng = Random.GLOBAL_RNG

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
const SPATIAL_RESOLUTION = 200
const START_IDX = 1
const END_IDX   = 1400

const init_segment = 1301
const end_segment  = 1400
const tmax = end_segment - init_segment + 1

# Initial starting position
const s0 = (x=709757.111649658, y=6.26772603565296e6)

# Spatial bounds for visualization
const X_MIN = 700000.0
const X_MAX = 715000.0
const Y_MIN = 6.2550e6
const Y_MAX = 6.2720e6

# File paths (Update these to your local paths if necessary)
const BASE_DIR      = "C:\\our_path_where_data_is"
const BATHY_PATH    = joinpath(BASE_DIR, "bathymetry", "map_Firth_of_Lorn_200m.tif")
const DEPTH_CSV     = joinpath(BASE_DIR, "observation", "depth.csv")
const MOORINGS_CSV  = joinpath(BASE_DIR, "observation", "moorings.csv")
const ACOUSTICS_CSV = joinpath(BASE_DIR, "observation", "acoustics.csv")

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================
println("Loading bathymetry data...")
bathy_data = load_bathymetry(BATHY_PATH)

println("Loading observation data...")
acoustic_data = load_acoustic_data(MOORINGS_CSV, ACOUSTICS_CSV, START_IDX, END_IDX)

# Slice data to the targeted segment
acoustic_array_segmented = acoustic_data.acoustic_array[init_segment:end_segment, :]
depth_obs_df = load_depth_data(DEPTH_CSV, init_segment, end_segment)

# Prepare format for Pigeons
Yaccustic, receivers = prepare_acoustic_observations(acoustic_array_segmented, acoustic_data.moorings_df)
Ydepth = prepare_depth_observations(depth_obs_df.depth)

# ============================================================================
# 2. INITIAL TRAJECTORY BUILD
# ============================================================================
println("Building initial trajectory...")

s_init = simulateRW_acoustic(tmax, Yaccustic; 
                             s0=s0, sigma=30.0, drift_cap=2.0,
                             bathy=bathy_data.bathy, ex_itp=bathy_data.ex_itp,
                             rng=rng)

plt_s_init = plot_trajectory(bathy_data.bathy, receivers, s_init; 
                             title="Initial Trajectory (s_init)")
display(plt_s_init)

# ============================================================================
# 3. PIGEONS PARALLEL TEMPERING SETUP
# ============================================================================
println("Setting up mapping and potentials...")

mapping = TransformVariables.as((
    traj    = TransformVariables.as(Array, TransformVariables.as((x = TransformVariables.asℝ, y = TransformVariables.asℝ)), tmax),
    σ       = TransformVariables.asℝ₊,
    σ_depth = TransformVariables.asℝ₊
))

v_init = TransformVariables.inverse(mapping, (traj=s_init, σ=30.0, σ_depth=30.0))

fish_prior_lp = FishPriorPotential(
    mapping, v_init, tmax, Yaccustic, Ydepth,
    bathy_data.bathymetry_int, bathy_data.x_origin, bathy_data.y_origin, 
    bathy_data.dx, bathy_data.dy, bathy_data.bathy, bathy_data.ex_itp
)

fish_lp = FishLogPotential(
    Ydepth, Yaccustic, bathy_data.bathymetry_int,
    bathy_data.x_origin, bathy_data.y_origin, bathy_data.dx, bathy_data.dy,
    mapping, v_init, bathy_data.bathy, bathy_data.ex_itp
)

# PT Execution Parameters
n_chains = 9 
n_rounds = 7

println("Starting Pigeons sampler...")
pt = pigeons(
    target        = fish_lp,
    reference     = fish_prior_lp,
    seed          = 1234,
    n_rounds      = n_rounds,
    n_chains      = n_chains,
    checkpoint    = true,
    multithreaded = false,
    explorer      = SliceSampler(),
    record        = [traces, online, round_trip, Pigeons.timing_extrema, 
                     Pigeons.allocation_extrema, index_process, swap_acceptance_pr]
)

# ============================================================================
# 4. RESULTS EXTRACTION & ANALYSIS
# ============================================================================
println("\nExtracting results...")

pt_samples  = Chains(pt)
cold_last_v = pt_samples.value[end, 1:(2*tmax + 2)] |> vec
params_last = TransformVariables.transform(mapping, cold_last_v)

cold_last_S       = params_last.traj
sigma_estimated   = params_last.σ
σ_depth_estimated = params_last.σ_depth

# Plot the indexing process
myplot3 = plot(pt.reduced_recorders.index_process, title="PT Chains=$n_chains, Rounds=$n_rounds")
display(myplot3)

# Extract parameter chains
sigma_samples       = [exp(pt_samples.value[i, 2*tmax + 1]) for i in 1:size(pt_samples.value, 1)]
sigma_depth_samples = [exp(pt_samples.value[i, 2*tmax + 2]) for i in 1:size(pt_samples.value, 1)]

# Print Basic Stats
println("\n=== POSTERIOR OF σ ===")
println("Mean:    $(round(mean(sigma_samples), digits=2)) m")
println("Median:  $(round(median(sigma_samples), digits=2)) m")
println("Std:     $(round(std(sigma_samples), digits=2)) m")

# Calculate Step & Distance statistics
calculate_step_statistics(cold_last_S)
receiver_analysis = analyze_receiver_distances(cold_last_S, receivers)

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
println("\nGenerating final plots...")

# Histograms
plt_sigma_hist = histogram(sigma_samples, bins=50, normalize=:pdf, title="Posterior of σ", label="Samples")
vline!(plt_sigma_hist, [30.0], label="Initial σ (30 m)", linestyle=:dash, color=:red)
display(plt_sigma_hist)

# Trajectory Comparison
plt_comparison = plot_trajectories_comparison(
    bathy_data.bathy, receivers, s_init, cold_last_S; 
    title="Comparison of Trajectories"
)
display(plt_comparison)

# Depth Comparison 
tempi = 1:tmax
depth_observed  = [Ydepth[t][2] for t in tempi]
depth_estimated = [get_depth_at(bathy_data.bathy, bathy_data.ex_itp, p.x, p.y) for p in cold_last_S]

plt_depth = plot(tempi, depth_observed, xlabel="Time", ylabel="Depth", label="Observed", legend=:topright)
plot!(plt_depth, tempi, depth_estimated, label="Pigeons Estimate")
display(plt_depth)