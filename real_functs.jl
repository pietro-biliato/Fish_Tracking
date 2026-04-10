## -------------------------------------------------------
## Core functions and types for Fish Tracking Inference
##
## April 10, 2026 -- Sebastian Waruszynski
## sebastian.waruszynski@studenti.unipd.it
## -------------------------------------------------------

using Distributions, Interpolations, Plots, Random
using TransformVariables, LogDensityProblems, LogDensityProblemsAD
using Pigeons, MCMCChains, StatsPlots, GeoArrays, StaticArrays
using CSV, DataFrames, Dates, Missings

import LogDensityProblems: logdensity, dimension, capabilities
import Pigeons: initialization, sample_iid!
import TransformVariables: AbstractTransform

# ============================================================================
# TYPES & STRUCTS
# ============================================================================
abstract type Sensor end

struct Receiver <: Sensor
    x::Float64
    y::Float64
    dist::Float64  # receiving distance
    k::Float64     # smoothness
end

Receiver(c; dist=50.0, k=30.0) = Receiver(c.x, c.y, dist, k)

struct DepthGauge <: Sensor end

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

#Load and prepare bathymetry data with interpolation
function load_bathymetry(path::String)
    bathy = GeoArrays.read(path)
    arr = GeoArrays.values(bathy)
    itp = interpolate(arr, BSpline(Linear()))
    ex_itp = extrapolate(itp, -1.0)
    bathymetry_int = extrapolate(interpolate(bathy, BSpline(Linear())), -1.0)
    
    # Extract coordinate transformation parameters
    aff = bathy.f
    A = aff.linear
    b = aff.translation
    
    return (
        bathy = bathy,
        ex_itp = ex_itp,
        bathymetry_int = bathymetry_int,
        x_origin = b[1],
        y_origin = b[2],
        dx = A[1,1],
        dy = A[2,2]
    )
end

#Load and clean depth observations
function load_depth_data(path::String, start_idx::Int, end_idx::Int)
    df = CSV.read(path, DataFrame, dateformat="yyyy-mm-dd H:M:S")
    rename!(df, :Column1 => :time)
    
    # Fix incompatible depth data
    df.depth[2041:2047] .= 205
    df.depth[4842:4845] .= 205
    
    return df[start_idx:end_idx, :]
end

#Load acoustic receiver positions and observations
function load_acoustic_data(moorings_path::String, acoustics_path::String, start_idx::Int, end_idx::Int)
    moorings_df = CSV.read(moorings_path, DataFrame)
    acoustic_obs_df = CSV.read(acoustics_path, DataFrame, dateformat="yyyy-mm-dd H:M:S", missingstring="NA")
    acoustic_obs_df = acoustic_obs_df[start_idx:end_idx, :]

    acoustic_col_names = string.(names(acoustic_obs_df)[2:end])
    rec_id_to_row = Dict(string(row.receiver_id) => i for (i, row) in enumerate(eachrow(moorings_df)))
    ordered_indices = [rec_id_to_row[id] for id in acoustic_col_names]

    acoustic_pos = [(x=moorings_df.receiver_x[i], y=moorings_df.receiver_y[i]) for i in ordered_indices]
    acoustic_pos_starfish = [(moorings_df.receiver_x[i], moorings_df.receiver_y[i]) for i in ordered_indices]

    acoustic_array = coalesce.(Array(acoustic_obs_df[:,2:end]), -1)
    acoustic_signals = [acoustic_array[:, r] for r in 1:size(acoustic_array, 2)]
    
    return (
        moorings_df = moorings_df,
        acoustic_pos = acoustic_pos,
        acoustic_pos_starfish = acoustic_pos_starfish,
        acoustic_array = acoustic_array,
        acoustic_signals = acoustic_signals
    )
end

# ============================================================================
# MATH & SPATIAL UTILITIES
# ============================================================================

dist(c1, c2) = sqrt((c1.x - c2.x)^2 + (c1.y - c2.y)^2)
direction(c1, c2) = atan(c1.x - c2.x,  c1.y - c2.y)

#Get depth at specific real-world coordinates
function get_depth_at(bathy::GeoArray, interp, x::Real, y::Real)
    f_inv = inv(bathy.f)
    colrow = f_inv(SVector(x, y))
    row = colrow[1]
    col = colrow[2]
    return interp(row, col)
end

# ============================================================================
# TRAJECTORY SIMULATION
# ============================================================================

#Generate a pure random walk trajectory, ensuring points stay in the water.
function simulateRW_free(tmax; s0, sigma = 30.0,
                         bathy = nothing, ex_itp = nothing,
                         max_land_tries = 100,
                         rng = Random.GLOBAL_RNG)
    
    in_water(x, y) = (bathy === nothing || ex_itp === nothing) || get_depth_at(bathy, ex_itp, x, y) > 0.0

    traj = Vector{NamedTuple{(:x,:y),Tuple{Float64,Float64}}}(undef, tmax)
    traj[1] = s0
    
    for t in 2:tmax
        prev = traj[t-1]
        candidate_x, candidate_y = prev.x, prev.y
        
        for _ in 1:max_land_tries
            cx = prev.x + randn(rng) * sigma
            cy = prev.y + randn(rng) * sigma
            if in_water(cx, cy)
                candidate_x, candidate_y = cx, cy
                break
            end
        end
        traj[t] = (x = candidate_x, y = candidate_y)
    end
    return traj
end

#Generate a trajectory with a soft drift towards acoustic detections.
#After the last detection, it reverts to a pure random walk.
function simulateRW_acoustic(tmax, Yacc; s0, sigma = 30.0,
                             drift_cap = 2.0,        
                             bathy = nothing,         
                             ex_itp = nothing,        
                             max_land_tries = 100,    
                             rng = Random.GLOBAL_RNG)

    in_water(x, y) = (bathy === nothing || ex_itp === nothing) || get_depth_at(bathy, ex_itp, x, y) > 0.0

    # Map t => anchor position (average if multiple receivers are active)
    anchor_pos   = Dict{Int, Tuple{Float64,Float64}}()
    anchor_count = Dict{Int, Int}()
    
    for (t, signal, device) in Yacc
        signal == :detect || continue
        1 <= t <= tmax    || continue
        
        if haskey(anchor_pos, t)
            n = anchor_count[t]
            ax, ay = anchor_pos[t]
            anchor_pos[t]   = ((ax * n + device.x) / (n + 1), (ay * n + device.y) / (n + 1))
            anchor_count[t] = n + 1
        else
            anchor_pos[t]   = (device.x, device.y)
            anchor_count[t] = 1
        end
    end

    anchor_times = sort(collect(keys(anchor_pos)))

    # Fallback to pure RW if no detections exist
    if isempty(anchor_times)
        return simulateRW_free(tmax; s0 = s0, sigma = sigma,
                               bathy = bathy, ex_itp = ex_itp,
                               max_land_tries = max_land_tries, rng = rng)
    end

    # Find next anchor for each time t
    next_anchor = Dict{Int, Tuple{Int,Float64,Float64}}()
    for t_anch in anchor_times
        ax, ay = anchor_pos[t_anch]
        t_prev_anch = searchsortedfirst(anchor_times, t_anch) == 1 ? 0 : anchor_times[searchsortedfirst(anchor_times, t_anch) - 1]
        for t in (t_prev_anch + 1):(t_anch - 1)
            next_anchor[t] = (t_anch, ax, ay)
        end
    end

    max_step = drift_cap * sigma
    traj = Vector{NamedTuple{(:x,:y),Tuple{Float64,Float64}}}(undef, tmax)
    traj[1] = s0

    for t in 2:tmax
        prev = traj[t-1]
        drift_x = 0.0
        drift_y = 0.0
        
        if haskey(next_anchor, t - 1)
            t_anch, ax, ay = next_anchor[t - 1]
            dt = t_anch - (t - 1)

            # Ideal bridge drift
            drift_x = (ax - prev.x) / dt
            drift_y = (ay - prev.y) / dt

            # Cap the drift
            drift_norm = sqrt(drift_x^2 + drift_y^2)
            if drift_norm > max_step
                scale   = max_step / drift_norm
                drift_x *= scale
                drift_y *= scale
            end
        end

        # Rejection sampling for land avoidance
        candidate_x = prev.x
        candidate_y = prev.y
        accepted = false
        
        for _ in 1:max_land_tries
            cx = prev.x + drift_x + randn(rng) * sigma
            cy = prev.y + drift_y + randn(rng) * sigma
            if in_water(cx, cy)
                candidate_x = cx
                candidate_y = cy
                accepted = true
                break
            end
        end
        
        if !accepted
            @warn "t=$t: Failed to find water after $max_land_tries tries. Staying at previous position."
        end

        traj[t] = (x = candidate_x, y = candidate_y)
    end

    return traj
end

# ============================================================================
# OBSERVATION PREPARATION
# ============================================================================

#Build receiver activation sequence from acoustic data
function build_receiver_sequence(acoustic_array, acoustic_pos; start_point::NamedTuple)
    n_time, n_receivers = size(acoustic_array)
    last_state = falses(n_receivers)
    events = Tuple{Int,Int}[]
    
    for t in 1:n_time, r in 1:n_receivers
        cur = acoustic_array[t, r] == 1
        if cur && !last_state[r]
            push!(events, (t, r))
        end
        last_state[r] = cur
    end
    
    if isempty(events)
        receiver_seq = [start_point, start_point]
        t_steps = [max(1, n_time - 1)]
    elseif length(events) == 1
        only_rec = acoustic_pos[events[1][2]]
        receiver_seq = [only_rec, only_rec]
        t_steps = [max(1, n_time - events[1][1])]
    else
        receiver_seq = [acoustic_pos[r] for (_t, r) in events]
        t_steps = [events[i+1][1] - events[i][1] for i in 1:length(events)-1]
    end
    
    return receiver_seq, t_steps, events
end

#Prepare acoustic observations for Pigeons
function prepare_acoustic_observations(acoustic_array, moorings_df)
    receivers = [
        Receiver(
            moorings_df.receiver_x[i],
            moorings_df.receiver_y[i],
            50.0,   # detection distance
            30.0    # detection parameter k
        ) for i in 1:nrow(moorings_df)
    ]
    
    Yaccustic = Tuple{Int, Symbol, Receiver}[]
    for t in 1:size(acoustic_array, 1)
        for r in 1:size(acoustic_array, 2)
            stato = acoustic_array[t, r]
            if stato != -1
                signal = stato == 1 ? :detect : :nondetect
                push!(Yaccustic, (t, signal, receivers[r]))
            end
        end
    end
    
    return Yaccustic, receivers
end

#Prepare depth observations for Pigeons
function prepare_depth_observations(depth_signals)
    Ydepth = Tuple{Int, Float64, DepthGauge}[]
    depthgauge = DepthGauge()
    
    for (t, d) in enumerate(depth_signals)
        push!(Ydepth, (t, d, depthgauge))
    end
    
    return Ydepth
end

# ============================================================================
# LIKELIHOODS & PRIORS
# ============================================================================

function log_p_moveRW(c1, c2, σ::Real)
    return logpdf(Normal(c2.x, σ), c1.x) + logpdf(Normal(c2.y, σ), c1.y)
end

function log_prob_signal(signal, s::NamedTuple, device::Receiver, k::Real)
    d = dist((x=device.x, y=device.y), s)
    d0 = device.dist  
    raw = 1 - 1/(1 + exp(-(d - d0)/k))
    prob_detect = clamp(raw, 1e-15, 1 - 1e-15)
    
    if signal == :detect
        return log(prob_detect)
    elseif signal == :nondetect
        return log(1 - prob_detect)
    end
    error("unknown signal: $(signal)")
end

function log_prob_signal(signal, s::NamedTuple, device::DepthGauge, 
                         bathymetry_int, x_origin, y_origin, dx, dy, 
                         bathy, ex_itp, σ_depth::Real)
    max_depth = get_depth_at(bathy, ex_itp, s.x, s.y)
    dist_distribution = Normal(max_depth, σ_depth)
    return logpdf(dist_distribution, signal)
end

function log_posterior(S, σ::Real, k::Real, σ_depth::Real,
                       Ydepth, Yaccustic, bathymetry_int,
                       x_origin, y_origin, dx, dy, bathy, ex_itp)
    tmax = length(S)
    lp   = 0.0
    for t in 2:tmax
        lp += log_p_moveRW(S[t], S[t-1], σ)
    end

    # Priors and Jacobians
    lp_val  = logpdf(Normal(60, 15), σ)
    lp_val += logpdf(LogNormal(3, 0.5), σ_depth)
    lp_val += log(σ)        # Jacobian: σ = exp(u)
    lp_val += log(σ_depth)  # Jacobian: σ_depth = exp(u)

    for (t, signal, device) in Ydepth
        t ≤ tmax || continue
        lp += log_prob_signal(signal, S[t], device, bathymetry_int,
                              x_origin, y_origin, dx, dy, bathy, ex_itp, σ_depth)
    end
    
    for (t, signal, device) in Yaccustic
        t ≤ tmax || continue
        lp += log_prob_signal(signal, S[t], device, k)
    end
    
    return lp + lp_val
end

# ============================================================================
# PIGEONS POTENTIALS (TARGET & REFERENCE)
# ============================================================================

struct FishPriorPotential{M,V,YA,YD,BI,B,E}
    mapping      :: M
    v_init       :: V
    tmax         :: Int
    Yacc         :: Vector{YA}
    Ydepth       :: Vector{YD}
    bathy_int    :: BI   
    x_origin     :: Float64     
    y_origin     :: Float64     
    dx           :: Float64     
    dy           :: Float64     
    bathy        :: B
    ex_itp       :: E
end

# Functor for the Reference (Prior)
function (lp::FishPriorPotential)(v::AbstractVector)
    params  = TransformVariables.transform(lp.mapping, v)
    S       = params.traj
    σ       = params.σ
    σ_depth = params.σ_depth 
    k       = 30.0 

    lp_val  = logpdf(Normal(60, 15), σ)
    for t in 2:lp.tmax
        lp_val += log_p_moveRW(S[t], S[t-1], σ)
    end

    lp_val += logpdf(LogNormal(3, 0.5), σ_depth)
    lp_val += log(σ)        
    lp_val += log(σ_depth)  
    
    for (t, signal, device) in lp.Yacc
        t ≤ length(S) || continue
        lp_val += log_prob_signal(signal, S[t], device, k)
    end
    
    # Note: one MAY add/omit the accustic likelihood in the PRIOR too (being aware that
    # formulating a data-dependent prior is usually not a good practice): we know a 
    # priori that the fish must pass close to the active receivers, when they're active.
    
    return lp_val
end

struct FishLogPotential{YD,YA,BI,M,B,E}
    Ydepth    ::Vector{YD}
    Yacc      ::Vector{YA}
    bathy_int ::BI
    x_origin  ::Float64
    y_origin  ::Float64
    dx        ::Float64
    dy        ::Float64
    mapping   ::M
    v_init    ::Vector{Float64}
    bathy     ::B     
    ex_itp    ::E          
end

# Functor for the Target (Posterior)
function (lp::FishLogPotential)(v::AbstractVector)
    params  = TransformVariables.transform(lp.mapping, v)
    S       = params.traj
    σ       = params.σ
    σ_depth = params.σ_depth 
    k       = 30.0
    return log_posterior(S, σ, k, σ_depth, lp.Ydepth, lp.Yacc, lp.bathy_int,
                         lp.x_origin, lp.y_origin, lp.dx, lp.dy,
                         lp.bathy, lp.ex_itp)
end

# ============================================================================
# PIGEONS INTERFACE METHODS
# ============================================================================

dimension(lp::FishLogPotential) = length(lp.v_init)
logdensity(lp::FishLogPotential, v::AbstractVector) = lp(v)
capabilities(::FishLogPotential) = LogDensityProblems.LogDensityOrder{0}()

dimension(lp::FishPriorPotential) = length(lp.v_init)
logdensity(lp::FishPriorPotential, v::AbstractVector) = lp(v)
capabilities(::FishPriorPotential) = LogDensityProblems.LogDensityOrder{0}()

function initialization(lp::FishLogPotential, rng::AbstractRNG, replica_index::Int)
    return copy(lp.v_init)
end

function initialization(lp::FishLogPotential, rng::AbstractRNG, v::AbstractVector)
    copyto!(v, lp.v_init)
    return v
end

function initialization(lp::FishPriorPotential, rng::AbstractRNG, replica_index::Int)
    return copy(lp.v_init)
end

function sample_iid!(lp::FishPriorPotential, rng::AbstractRNG, v::AbstractVector; tries=300)
    s0 = (x=709757.111649658, y=6.26772603565296e6) # Hardcoded start point
    σ_sample = rand(rng, truncated(Normal(60, 15), 1.0, Inf))
    σ_depth_sample = rand(rng, LogNormal(3, 0.5)) 

    for _ in 1:tries
        # Using the cleaned acoustic simulator
        traj = simulateRW_acoustic(lp.tmax, lp.Yacc; s0=s0, sigma=σ_sample, drift_cap=2.0, bathy=lp.bathy, ex_itp=lp.ex_itp, rng=rng)
        
        # Land rejection: if more than 10% of the trajectory is on land, reject and retry
        n_land = count(p -> get_depth_at(lp.bathy, lp.ex_itp, p.x, p.y) <= 0.0, traj)
        n_land / lp.tmax > 0.1 && continue

        params = (traj=traj, σ=σ_sample, σ_depth=σ_depth_sample)
        copyto!(v, TransformVariables.inverse(lp.mapping, params))
        return v
    end

    # Fallback if no valid trajectory is found after `tries`
    @warn "sample_iid! fallback after $tries tries -> returning flat path"
    traj   = fill(s0, lp.tmax)
    params = (traj=traj, σ=σ_sample, σ_depth=σ_depth_sample)
    copyto!(v, TransformVariables.inverse(lp.mapping, params))
    return v
end

function sample_iid!(lp::FishPriorPotential, replica, shared)
    sample_iid!(lp, replica.rng, replica.state)   
    return replica.state
end

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

#Analyze active receivers in the specified segment
function analyze_active_receivers(acoustic_array, init_segment, end_segment)
    segment_array = acoustic_array[init_segment:end_segment, :]
    n_time, n_receivers = size(segment_array)
    
    println("\n=== ACTIVE RECEIVERS ANALYSIS ===")
    println("Analyzing segment from index $init_segment to $end_segment")
    println("Total time steps in segment: $n_time")
    println("Total receivers: $n_receivers")
    
    activations = []
    for t in 1:n_time
        for r in 1:n_receivers
            if segment_array[t, r] == 1
                original_time_idx = t + init_segment - 1
                push!(activations, (receiver=r, time_idx=original_time_idx, segment_time=t))
                println("Receiver $r is active at original index $original_time_idx (segment time $t)")
            end
        end
    end
    
    if isempty(activations)
        println("No active receivers found in the specified segment.")
    else
        println("\nSummary: Found $(length(activations)) activations in the segment")
        receiver_activations = Dict{Int, Vector{Int}}()
        for act in activations
            if !haskey(receiver_activations, act.receiver)
                receiver_activations[act.receiver] = []
            end
            push!(receiver_activations[act.receiver], act.time_idx)
        end
        
        println("\nActivations by receiver:")
        for (receiver, times) in sort(collect(receiver_activations))
            println("  Receiver $receiver: active at indices $(times)")
        end
    end
    
    return activations
end

#Calculate step statistics for a trajectory
function calculate_step_statistics(trajectory)
    if length(trajectory) < 2
        println("\n=== STEP STATISTICS ===")
        println("Trajectory too short for step analysis (length: $(length(trajectory)))")
        return
    end
    
    distances = [dist(trajectory[i-1], trajectory[i]) for i in 2:length(trajectory)]
    
    mean_step = mean(distances)
    max_step = maximum(distances)
    min_step = minimum(distances)
    std_step = std(distances)
    
    println("\n=== STEP STATISTICS ===")
    println("Total steps analyzed: $(length(distances))")
    println("Mean step length: $(round(mean_step, digits=2)) meters")
    println("Maximum step length: $(round(max_step, digits=2)) meters")
    println("Minimum step length: $(round(min_step, digits=2)) meters")
    println("Standard deviation: $(round(std_step, digits=2)) meters")
    
    return (mean=mean_step, max=max_step, min=min_step, std=std_step)
end

#Calculate distance to nearest receiver for each point in trajectory
function analyze_receiver_distances(trajectory, receivers)
    println("\n=== DISTANCE TO NEAREST RECEIVER ANALYSIS ===")
    println("Analyzing $(length(trajectory)) trajectory points against $(length(receivers)) receivers")
    
    nearest_distances = []
    nearest_receiver_ids = []
    
    for (i, point) in enumerate(trajectory)
        min_dist = Inf
        nearest_receiver_id = -1
        
        for (r_id, receiver) in enumerate(receivers)
            d = dist(point, receiver)
            if d < min_dist
                min_dist = d
                nearest_receiver_id = r_id
            end
        end
        
        push!(nearest_distances, min_dist)
        push!(nearest_receiver_ids, nearest_receiver_id)
        println("Point $i: nearest receiver = $nearest_receiver_id, distance = $(round(min_dist, digits=2)) m")
    end
    
    mean_dist = mean(nearest_distances)
    max_dist = maximum(nearest_distances)
    min_dist = minimum(nearest_distances)
    std_dist = std(nearest_distances)
    
    println("\n=== SUMMARY STATISTICS ===")
    println("Mean distance: $(round(mean_dist, digits=2)) m")
    println("Max distance: $(round(max_dist, digits=2)) m")
    println("Min distance: $(round(min_dist, digits=2)) m")
    println("Standard deviation: $(round(std_dist, digits=2)) m")
    
    detection_distance = 50.0
    points_within_detection = sum(nearest_distances .<= detection_distance)
    percentage_within = (points_within_detection / length(nearest_distances)) * 100
    
    println("\n=== DETECTION CONSTRAINT ANALYSIS ===")
    println("Detection distance threshold: $(detection_distance) m")
    println("Points within detection range: $points_within_detection/$(length(nearest_distances)) ($(round(percentage_within, digits=1))%)")
    
    points_beyond_detection = length(nearest_distances) - points_within_detection
    if points_beyond_detection > 0
        println("⚠️  WARNING: $points_beyond_detection points are beyond detection range!")
    else
        println("✅ All points are within detection range")
    end
    
    return (
        distances = nearest_distances,
        receiver_ids = nearest_receiver_ids,
        mean_dist = mean_dist,
        max_dist = max_dist,
        min_dist = min_dist,
        points_within_detection = points_within_detection,
        percentage_within = percentage_within
    )
end

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

#Create bathymetry plot with receivers and trajectory
function plot_trajectory(bathy, receivers, trajectory; 
                         title="Fish Trajectory", xlims=(700000.0, 715000.0), ylims=(6.2550e6, 6.2720e6))
    plt = heatmap(bathy; color=:blues, legend=false, title=title, xlims=xlims, ylims=ylims)
    
    xs_rec = [r.x for r in receivers]
    ys_rec = [r.y for r in receivers]
    scatter!(plt, xs_rec, ys_rec; color=:red, markersize=4, label="Receivers")
    
    xs_traj = [p.x for p in trajectory]
    ys_traj = [p.y for p in trajectory]
    plot!(plt, xs_traj, ys_traj, lw=2, color=:orange, label="Trajectory")
    
    return plt
end

#Create bathymetry plot with receivers and multiple trajectories
function plot_trajectories_comparison(bathy, receivers, traj_init, traj_pigeons; 
                                      title="Fish Trajectory Comparison", xlims=(700000.0, 715000.0), ylims=(6.2550e6, 6.2720e6))
    plt = heatmap(bathy; color=:blues, legend=:topright, title=title, xlims=xlims, ylims=ylims)
    
    xs_rec = [r.x for r in receivers]
    ys_rec = [r.y for r in receivers]
    scatter!(plt, xs_rec, ys_rec; color=:red, markersize=4, label="Receivers")
    
    xs_init = [p.x for p in traj_init]
    ys_init = [p.y for p in traj_init]
    plot!(plt, xs_init, ys_init, lw=2, color=:pink, alpha=0.7, label="Initial Trajectory")
    
    xs_pigeons = [p.x for p in traj_pigeons]
    ys_pigeons = [p.y for p in traj_pigeons]
    plot!(plt, xs_pigeons, ys_pigeons, lw=2, color=:orange, label="Pigeons Result")
    
    return plt
end