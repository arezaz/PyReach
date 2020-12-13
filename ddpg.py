#%% train ddpg
import os
import gym
import numpy as np
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.noise import NormalActionNoise
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback

import Arm2DEnv as ae
from utils import SaveOnBestTrainingRewardCallback, snap_code
#%% sac

log_dir_root = './sandbox/ddpg/'
os.makedirs(log_dir_root, exist_ok=True)

# create a snapshot of code
log_dir = snap_code(log_dir_root)

env = ae.ArmModel()
env.mode = 'train_curriculum'
env.curriculum = [1E4, 5E4, 1E5]
env = Monitor(env, log_dir)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

model = DDPG(LnMlpPolicy, env, verbose=1, gamma=0.98, tau=0.01,
             actor_lr=0.0001, critic_lr=0.001, action_noise=action_noise,
             buffer_size=int(5E6), batch_size=128, random_exploration=0.001)

callback = SaveOnBestTrainingRewardCallback(check_freq=int(5E4), log_dir=log_dir, check_points=[1E4, 5E4, 1E5])

model.learn(total_timesteps=int(6E6), callback=callback)#(total_timesteps=int(4E5))

# %%
model.save("twolink_arm_ddpg")