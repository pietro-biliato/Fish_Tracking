# Fish Tracking via Bayesian Inference

## Project Overview

Understanding the movements and migrations of marine animals, specifically rays, is a fundamental challenge in spatial ecology. This knowledge is essential for explaining observed ecological patterns, predicting responses to environmental changes, and informing effective conservation and management strategies. Because direct observation of individual rays over large scales is rarely feasible in vast and opaque aquatic environments, researchers rely on indirect measurements collected via electronic data storage tags attached to the animals. 

These tags can record various environmental variables. However, the measurements from these tags are inherently noisy and indirect due to sensor inaccuracies, environmental variability, and incomplete spatial coverage. To address this, our project reconstructs plausible ray trajectories by leveraging the following specific types of available data:

* **Depth Measurements ($Y_d$):** Readings from pressure sensors that are matched to known bathymetry maps to estimate position. Uncertainties in this data often arise from tidal fluctuations, sensor drift, or imprecise bathymetric data.
* **Acoustic Data ($Y_a$):** Records of detections (and non-detections) from acoustic tags passing in proximity to fixed receivers. Detection probabilities naturally decrease with distance due to signal attenuation in water.
* **Bathymetry Data:** Detailed maps of the underwater depth profile used alongside the depth measurements to enforce spatial realism and constraint the feasible state of the ray's trajectory.

To address the uncertainties in these diverse datasets, this project utilizes a statistical framework to reconstruct plausible trajectories. We infer the unknown ray trajectory using a Bayesian sampling approach—specifically Parallel Tempering (PT) Markov Chain Monte Carlo (MCMC)—to draw samples from the posterior distribution of positions given all depth and acoustic observations.

## Methodology

The problem of track reconstruction can be cast in a Bayesian inference framework. The inference problem consists of estimating the posterior distribution of trajectories conditional on the observed depth and acoustic data. 

We tackled the problem of track-reconstruction using three successive approaches:
* **Toy bathymetry toy data**: Works in a fully controllable scenario using hand-drawn channel-like bathymetries.
* **Real bathymetry toy data**: Uses the same generation approach but implements a portion of a real bathymetry map.
* **Real bathymetry real data**: A real-life application where both depth and acoustic data are already available.

To efficiently explore the computationally intractable and multimodal posterior distributions, this project utilizes Parallel Tempering. The sampling procedure is implemented via the `Pigeons.jl` Julia package, which supports non-reversible parallel tempering (NRPT). 

### Sampling Explorers
In the `Pigeons.jl` package, samplers (referred to as "explorers") are the core Markov chain Monte Carlo (MCMC) kernels used within each chain of the parallel tempering framework. This project evaluates two primary explorers:
* **AutoMALA**: A gradient-based MCMC method that dynamically adapts the step size at each iteration based on the local geometry of the target.
* **Slice Sampler**: An alternative method that adaptively adjusts the scale of proposed changes based on the local properties of the distribution.

## Trajectory Preprocessing Pipeline
For the real-data scenario, the trajectory preprocessing workflow leverages several specialized functions to prepare acoustic and depth observation data:
* **Acoustic Data Import**: Imports and formats receiver positions and detection observations.
* **Receiver Activation Sequence Construction**: Processes acoustic data to identify the chronological sequence of activated receivers.
* **Missing Data Interpolation**: Detects contiguous runs of missing data and performs linear interpolation between the nearest valid points.
* **Elimination of Consecutive Duplicates**: Compresses consecutive duplicate coordinates into single representative points to mitigate issues for the Pigeons algorithm.
* **Trajectory Expansion**: Linearly interpolates between fundamental points to match the required trajectory length, adding Gaussian noise to avoid artificial regularity.

## Repository Structure

* **`toy_data.jl`**: The main script for the toy data analysis, covering both the toy bathymetry and real bathymetry configurations.
* **`run_toy.jl`**: The module/script containing the core functions utilized by `toy_data.jl`.
* **`run_inference_PT.jl`**: The main execution script for analyzing the real bathymetry with real observed data.
* **`find_PT.jl`**: The script containing the specialized functions and preprocessing pipeline steps required by `run_inference_PT.jl`.
