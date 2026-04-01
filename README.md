# Fish Tracking via Bayesian Inference

## Project Overview

[cite_start]Understanding the movements and migrations of marine animals, specifically rays, is a fundamental challenge in spatial ecology[cite: 4]. [cite_start]This knowledge is essential for explaining observed ecological patterns, predicting responses to environmental changes, and informing effective conservation and management strategies[cite: 5]. [cite_start]Because direct observation of individual rays over large scales is rarely feasible in vast and opaque aquatic environments [cite: 6][cite_start], researchers rely on indirect measurements collected via electronic data storage tags attached to the animals[cite: 7]. 

[cite_start]These tags can record various environmental variables[cite: 8]. [cite_start]However, the measurements from these tags are inherently noisy and indirect due to sensor inaccuracies, environmental variability, and incomplete spatial coverage[cite: 9]. To address this, our project reconstructs plausible ray trajectories by leveraging the following specific types of available data:

* [cite_start]**Depth Measurements ($Y_d$):** Readings from pressure sensors that are matched to known bathymetry maps to estimate position[cite: 10]. [cite_start]Uncertainties in this data often arise from tidal fluctuations, sensor drift, or imprecise bathymetric data[cite: 10].
* [cite_start]**Acoustic Data ($Y_a$):** Records of detections (and non-detections) from acoustic tags passing in proximity to fixed receivers[cite: 11, 28]. [cite_start]Detection probabilities naturally decrease with distance due to signal attenuation in water[cite: 11].
* [cite_start]**Bathymetry Data:** Detailed maps of the underwater depth profile used alongside the depth measurements to enforce spatial realism and constraint the feasible state of the ray's trajectory[cite: 42, 245].

[cite_start]To address the uncertainties in these diverse datasets, this project utilizes a statistical framework to reconstruct plausible trajectories[cite: 12]. [cite_start]We infer the unknown ray trajectory using a Bayesian sampling approach—specifically Parallel Tempering (PT) Markov Chain Monte Carlo (MCMC)—to draw samples from the posterior distribution of positions given all depth and acoustic observations[cite: 20, 23, 242].

## Methodology

[cite_start]The problem of track reconstruction can be cast in a Bayesian inference framework[cite: 240]. [cite_start]The inference problem consists of estimating the posterior distribution of trajectories conditional on the observed depth and acoustic data[cite: 241]. 

[cite_start]We tackled the problem of track-reconstruction using three successive approaches[cite: 48]:
* [cite_start]**Toy bathymetry toy data**: Works in a fully controllable scenario using hand-drawn channel-like bathymetries[cite: 49, 56, 57].
* [cite_start]**Real bathymetry toy data**: Uses the same generation approach but implements a portion of a real bathymetry map[cite: 50, 104, 105].
* [cite_start]**Real bathymetry real data**: A real-life application where both depth and acoustic data are already available[cite: 51, 117, 119].

[cite_start]To efficiently explore the computationally intractable and multimodal posterior distributions, this project utilizes Parallel Tempering[cite: 249, 250]. [cite_start]The sampling procedure is implemented via the `Pigeons.jl` Julia package, which supports non-reversible parallel tempering (NRPT)[cite: 319, 374]. 

### Sampling Explorers
[cite_start]In the `Pigeons.jl` package, samplers (referred to as "explorers") are the core Markov chain Monte Carlo (MCMC) kernels used within each chain of the parallel tempering framework[cite: 394]. [cite_start]This project evaluates two primary explorers[cite: 400]:
* [cite_start]**AutoMALA**: A gradient-based MCMC method that dynamically adapts the step size at each iteration based on the local geometry of the target[cite: 407, 413].
* [cite_start]**Slice Sampler**: An alternative method that adaptively adjusts the scale of proposed changes based on the local properties of the distribution[cite: 470, 475].

## Trajectory Preprocessing Pipeline
[cite_start]For the real-data scenario, the trajectory preprocessing workflow leverages several specialized functions to prepare acoustic and depth observation data[cite: 120]:
* [cite_start]**Acoustic Data Import**: Imports and formats receiver positions and detection observations[cite: 122, 123].
* [cite_start]**Receiver Activation Sequence Construction**: Processes acoustic data to identify the chronological sequence of activated receivers[cite: 143, 144].
* [cite_start]**Missing Data Interpolation**: Detects contiguous runs of missing data and performs linear interpolation between the nearest valid points[cite: 173, 176].
* [cite_start]**Elimination of Consecutive Duplicates**: Compresses consecutive duplicate coordinates into single representative points to mitigate issues for the Pigeons algorithm[cite: 199, 202].
* [cite_start]**Trajectory Expansion**: Linearly interpolates between fundamental points to match the required trajectory length, adding Gaussian noise to avoid artificial regularity[cite: 223, 225, 226].

## Repository Structure

* **`toy_data.jl`**: The main script for the toy data analysis, covering both the toy bathymetry and real bathymetry configurations.
* **`run_toy.jl`**: The module/script containing the core functions utilized by `toy_data.jl`.
* **`run_inference_PT.jl`**: The main execution script for analyzing the real bathymetry with real observed data.
* **`find_PT.jl`**: The script containing the specialized functions and preprocessing pipeline steps required by `run_inference_PT.jl`.
