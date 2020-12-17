#%% train sac
import os
import gym
import numpy as np
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.sac.policies import LnMlpPolicy
from stable_baselines.common.noise import NormalActionNoise
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback

import Arm2DEnv as ae
from utils import SaveOnBestTrainingRewardCallback, snap_code
#%% sac

log_dir_root = './sandbox/sac/'
os.makedirs(log_dir_root, exist_ok=True)

# create a snapshot of code
log_dir = snap_code(log_dir_root)

env = ae.ArmModel()
env.mode = 'train_curriculum_rand_onetarg'
env.curriculum = [2.0E6, 4.0E6, 4.5E6]
env = Monitor(env, log_dir)

model = SAC(LnMlpPolicy, env, buffer_size=int(1E6), batch_size=128, gamma=0.98, learning_rate = 0.001, tau = 0.01, verbose=1)

check_points = {
                0.5E6: 'baseline',
                1.0E6: 'baseline',
                1.5E6: 'baseline',
                1.6E6: 'baseline',
                1.8E6: 'baseline',
                2.0E6: 'baseline',
                2.2E6: 'adaptation',
                2.4E6: 'adaptation',
                2.6E6: 'adaptation',
                2.8E6: 'adaptation',
                2.9E6: 'adaptation',
                3.0E6: 'adaptation',
                3.1E6: 'adaptation',
                3.2E6: 'adaptation',
                3.4E6: 'adaptation',
                3.6E6: 'adaptation',
                3.8E6: 'adaptation',
                4.0E6: 'adaptation',
                4.2E6: 'washout',
                4.4E6: 'washout',
                4.5E6: 'washout',
                }

callback = SaveOnBestTrainingRewardCallback(check_freq=int(5E4), log_dir=log_dir, check_points=check_points)
model.learn(total_timesteps=int(4.5E6), log_interval=10, callback=callback)

# %%
model.save("twolink_arm_sac")