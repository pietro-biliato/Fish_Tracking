#=------------------

This code runs the path recostruction for both the toy and real
bathymetry, on toy data, with the possibility of performing a
grid search (and properly plotting the results) on the number 
of chains and the number of rounds, eventually repeating multiple
times each run.

Note: the shift from toy to real bathy, as for now, has to be 
performed manually. The things to change are listed below; however
they're also flagged by proper comments in the code.

- name of the bathy file
- set of receivers 
- change tmax
- change 'coarse_steps'

In the 'run_toy.jl' included file change:

- In the function 'get_depth' add or remove the '/100', needed 
  for proper rescaling
- the sigma of the gaussian of the depth model 'log_prob_signal'


October 03, 2025 -- Pietro Biliato
pietro.biliato@studenti.unipd.it
------------------=#

import Pkg

Pkg.activate(".")
Pkg.instantiate()

using Images: load, Gray, channelview
using Dates

include("run_toy.jl")
gr()
Random.seed!(1274)


#--------- basics
# Load bathymetry, negative values are treated as land:
bathymetry_map = channelview(Gray.(load("bathymetry_maps/map_Firth_of_Lorn_200m.tif"))) * 100 .- 1;
bathymetry_int = extrapolate(interpolate(bathymetry_map, BSpline(Linear())),-1.0);

#For toy bathy
receiver1 = Receiver((x=10, y=45), k=30.0, dist=30.0)
receiver2 = Receiver((x=100, y=50), k=30.0, dist=30.0)
receiver3 = Receiver((x=110, y=140), k=30.0, dist=30.0)

#For real bathy
#=receiver1 = Receiver((x=100, y=100), k=30.0, dist=30.0)
receiver2 = Receiver((x=300, y=100), k=30.0, dist=30.0)
receiver3 = Receiver((x=300, y=100), k=30.0, dist=30.0)=#

#--------- bridges
#Building bridges from receiver1 to receiver 2, which is a way to get all (if n_bridges, tmax and sigma are big enough to explore all the channels) the plausible 
#paths that the fish may have followed:
tmax = 50       #100 for toy bathimetry, 50 for real bathy

n_bridges = 8000
bridges = []
failed_traj=0
n=n_bridges/2
for _ in (1:(n_bridges/2)) 
    path = simulate_unbiased_path(
        tmax;
        rec1=receiver1,
        rec2=receiver2,
        σ_step=3.0,
        bathymetry_int=bathymetry_int,
        coarse_steps=50,  #70 for toy bathimetry, 50 for real
        min_endpoint_prob=0.5, #the endpoint lands in an area that corresponds to this acceptance probability
        noise_σ=0.5, #perturbation of the interpolation points
        noise_retries::Int = 10, #if adding noise causes the trajectory to fail, retry this amount of times
        max_depth_diff::Float64 = 15.0  #maximum depth difference allowed between two consecutive steps
    )
    
    if path !== nothing
        push!(bridges, path)
    else
        failed_traj=failed_traj+1
    end
end
print("Number of failed trajectories for horizontal bridges: $failed_traj, over $n")

failed_traj=0
for _ in ((n_bridges/2 +1) : n_bridges)
    path = simulate_unbiased_path(
        tmax;
        rec1=receiver1,
        rec2=receiver3,
        σ_step=3.0,
        bathymetry_int=bathymetry_int,
        coarse_steps=65,  #70 for toy bathimetry, 65 for real
        min_endpoint_prob=0.5,
        noise_σ=0.5,
        noise_retries::Int = 10,
        max_depth_diff::Float64 = 20.0  
    )
    
    if path !== nothing
        push!(bridges, path)
    else
        failed_traj=failed_traj+1
    end
end
print("\nNumber of failed trajectories for oblique bridges: $failed_traj, over $n")

bridges = filter(!isnothing, bridges)  # keep just the valid trajectories
bridges_x = [[p.x for p in bridge] for bridge in bridges]
bridges_y = [[p.y for p in bridge] for bridge in bridges]

#To plot all the bridges
#=plt = heatmap(bathymetry_map[end:-1:1,:],
              xlim=(0, 200), ylim=(0, 200),
              color=:blues,
              legend=false,
              title="Brownian bridges betw receiver1 e receiver2")
for i in 1:length(bridges)
    plot!(plt, bridges_x[i], bridges_y[i], lw=2)
end
plot!(plt, make_circle(receiver1.x, receiver1.y, receiver1.dist), lw=2, color=:green, label="Receiver 1")
plot!(plt, make_circle(receiver2.x, receiver2.y, receiver2.dist), lw=2, color=:green, label="Receiver 2")
plot!(plt, make_circle(receiver2.x, receiver2.y, receiver2.dist), lw=2, color=:green, label="Receiver 3")
display(plt)=#


s_init = bridges[end]
s_depth = bridges[1]

xs_d = [p.x for p in s_depth]
ys_d = [p.y for p in s_depth]

xs = [p.x for p in s_init]
ys = [p.y for p in s_init]

#scatter!(plt, [receiver1.x, receiver2.x], [receiver1.y, receiver2.y], color=:red, label="Receivers")

plt2 = heatmap(bathymetry_map[end:-1:1,:],
              xlim=(0, 200), ylim=(0, 200),
              color=:blues,
              legend=false,
              title="Brownian bridges tra receiver1 e receiver2")
plot!(plt2, xs, ys, lw=3, color=:red, label="chosen trajectory")
plot!(plt2, xs_d, ys_d, lw=3, color=:black, label="goal trajectory")
plot!(plt2, make_circle(receiver1.x, receiver1.y, receiver1.dist), lw=2, color=:green, label="activation 1")
plot!(plt2, make_circle(receiver2.x, receiver2.y, receiver2.dist), lw=2, color=:green, label="activation 2")
plot!(plt2, make_circle(receiver3.x, receiver3.y, receiver3.dist), lw=2, color=:green, label="activation 3") #for real bathy
display(plt2)

#--------- data
# Accustic signals:
receivers = [receiver1, receiver2]
Yaccustic = build_Yaccustic_from_trajectory(s_depth, receivers)

#Depth signal:
Ydepth = Tuple{Int, Float64, DepthGauge}[] 
depthgauge = DepthGauge()

#"Geolocating Fish Using Hidden Markov Models and Data Storage Tags" uses uniform noise in [-10,10], and a "depth model" that is a gaussian with sigma 15
for (t, point) in enumerate(s_depth)
    # Get the depth from the bathymetry
    d = get_depth((x=point.x, y=point.y), bathymetry_int)
    noisy_d = d + rand(Uniform(-10, 10))   #-1,1  for toy bahtimetry, -10,10 for real bathymetry
    push!(Ydepth, (t+1, noisy_d, depthgauge))
end

#For gaussian noise instead
#=sigma_noise=0.5
for (t, point) in enumerate(bridges[2])
    d = get_depth((x=point.x, y=point.y), bathymetry_int)
    noisy_d = d + randn() * sigma_noise
    push!(Ydepth, (t+1, noisy_d, depthgauge))
end=#

#--------- PT 
@info "Optimizing posterior for a MAP starting point…"              

#Setting the problem characteristics:
mapping = TransformVariables.as(Array, 
                                TransformVariables.as((x = TransformVariables.asℝ, y = TransformVariables.asℝ)),
                                tmax)
v_init = TransformVariables.inverse(mapping, s_init)

fish_lp = FishLogPotential(Ydepth, Yaccustic, bathymetry_int, mapping, v_init, bridges)
fish_ref = FishReferencePotential(bathymetry_int, mapping, v_init, bridges)


# Grid search ranges
n_chains_list = [6,10,15] #the min has to be >='n_local_mpi_processes'
n_rounds_list = [9,10]
n_repeats = 1  # Number of repetitions

# Store results
all_trajectories = []
all_logposteriors = []
logpost_cold_values = Float64[]
std_devs = Float64[]
prof=[]

# Placeholder Ydepth for scatter comparison
Ydepth_values = [y[2] for y in Ydepth] 


#=Pkg.add("PairPlots")
Pkg.add("CairoMakie")
using PairPlots
using CairoMakie=#


for n_chains in n_chains_list
    for n_rounds in n_rounds_list
        timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
        for rep in 1:n_repeats
            println("Running with n_chains=$n_chains, n_rounds=$n_rounds, repetition=$rep")
            find_grid_path = abspath("find_grid.jl")
            pt_results = pigeons(
                target        = fish_lp,
                reference     = fish_ref,
                seed          = rep,  # vary seed for reproducibility
                n_rounds      = n_rounds,
                n_chains      = n_chains,
                checkpoint    = true,
                multithreaded = true,
                on = ChildProcess(n_local_mpi_processes = 5,
                                  n_threads=2,
                                  dependencies = [find_grid_path]),
                explorer      = SliceSampler(),#=AutoMALA(
                           step_size            = 6.0,           # passo iniziale MALA
                           base_n_refresh       = 13,     #13        # passi base per esplorazione
                           exponent_n_refresh   = 0.5,           # scala con √dim
                           default_autodiff_backend = AutoEnzyme()        # backend autodiff
                       ),=#
                record        = [traces, online, round_trip, Pigeons.timing_extrema, Pigeons.allocation_extrema, index_process]
            )
            pt = Pigeons.load(pt_results)
            pt_samples = Chains(pt)
            myplot3 = plot(pt.reduced_recorders.index_process, title="chains=$n_chains, rounds=$n_rounds")
            folder_path = "images/index_process"
            mkpath(folder_path)
            savefig(myplot3, "images/index_process/" * timestamp * ".svg")
            #myplot4 = pairplot(pt_samples) 
            #CairoMakie.save("pair_plot5.svg", myplot4)
            cold_last_v = pt_samples.value[end, 1:2*tmax] |> vec
            cold_last_S = TransformVariables.transform(mapping, cold_last_v)

            push!(all_trajectories, cold_last_S)

            logposteriors = [fish_lp(pt_samples.value[i, 1:2*tmax]) for i in 1:size(pt_samples.value, 1)]
            push!(all_logposteriors, logposteriors)

            # Log posterior of last cold chain sample
            logpost_cold = fish_lp(cold_last_v)
            push!(logpost_cold_values, logpost_cold)

            # Depth measurement
            YdepthPIGEONS = Tuple{Int, Float64, DepthGauge}[] 
            for (t, point) in enumerate(cold_last_S)
                d = get_depth((x=point.x, y=point.y), bathymetry_int)
                push!(YdepthPIGEONS, (t+1,d, depthgauge))
            end

            profonditaP = [y[2] for y in YdepthPIGEONS]
            push!(prof, profonditaP)
            std_dev = std(profonditaP .- Ydepth_values[1:length(profonditaP)])
            push!(std_devs, std_dev)
        end
    end
end


