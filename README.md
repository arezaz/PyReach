# PyReach

PyReach is an implementation of point-to-point arm reaching motions. Dynamics of the upper-limb has been simplified to a two-link-arm model. The goal is to replicate classical motor control experiments using various control schemes from classical approaches to RL-based methods.

### What's Here
* A gym-compatible environment of the upper-limb.
* Three controllers:
    * Impedance-based.
    * Soft actor-critic (SAC).
    * Deep Deterministic Policy Gradiant (DDPG).
* Tools to replicate motor control experiments.

### File Description:
* Arm:
    * `arm_params.py`
        > Mechanical properties of the upper-limb.
    * `Arm2DEnv.py`
        > A gym-like environment of goal-directed arm movement.
* Controllers:
    * `imp_ctrl.py`
        > Classical impendace control based on minimum-jerk trajectory.
    * `sac.py`
        > Train soft actor-critic controller.
    * `ddpg`
        > Train deep deterministic policy gradiant controller.
* Tools
    * `mainReach.ipynb`
        > Hook a controller to the environment and visualize trajectories, rewards, etc.
    * `mainExperiment.ipynb`
        > Hook a controller to the environment and visualize experimental results.
    * `utils.py`
        > Helper functions for all codes.

### How to Run:
If you want to train an RL controller (SAC, or DDPG), run the corresponding code. When running the training code a snapshot of all codes will be saved to `./sandbox/` directory marked with a timestamp and the name of the algorithm. All training logs will be saved to the same directory. While the training is in progress, you can load the logs and the model's checkpoint using the snapshots of the codes and visualzie results.

---
Alireza Rezazadeh | 
rezaz003@umn.edu | Fall 2020
