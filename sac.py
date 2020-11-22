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
env = Monitor(env, log_dir)

model = SAC(LnMlpPolicy, env, buffer_size=int(5E5), batch_size=128, gamma=0.98, learning_rate = 0.001, tau = 0.01, verbose=1)

callback = SaveOnBestTrainingRewardCallback(check_freq=int(5E4), log_dir=log_dir)
model.learn(total_timesteps=int(3E6), log_interval=10, callback=callback)

# %%
model.save("twolink_arm_sac")