# Interpretable Self-Aware Prediction (ISAP)

We propose the use of evidential deep learning to perform one-shot epistemic uncertainty estimation over a low-dimensional, interpretable latent space in a trajectory prediction setting. This code runs the qualitative and quantitative experiments to validate the proposed Interpretable Self-Aware Prediction (ISAP) framework. 

See our [paper](https://arxiv.org/abs/2211.08701) for more details:

M. Itkina and M. J. Kochenderfer. "Interpretable Self-Aware Neural Networks for Robust Trajectory Prediction". In Conference on Robot Learning (CoRL), 2022. 


## Instructions

The required dependencies are listed in `dependencies.txt`.

The NuScenes trajectory prediction dataset has to be downloaded from: https://www.nuscenes.org/nuscenes#download and placed into the `data/nuscenes/` folder, including a `covernet_traj_set` containing the trajectory sets, `maps` directory, and `v1.0-trainval` data. The NuScenes github repository: https://github.com/nutonomy/nuscenes-devkit should be cloned and the `nuscenes-devkit` folder to be placed at the top-level.

The PostNet code should be cloned from: https://github.com/sharpenb/Posterior-Network and placed at the top-level.

This code was developed and tested with `Python 3.6.12`.

To replicate the input trajectory speed experiments, please run the following files: 

``run_isap_agent_speed.sh``

``run_postcovernet_agent_speed.sh``

``run_ensembles_agent_speed.sh``
