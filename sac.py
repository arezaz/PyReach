#%%
import gym
import numpy as np
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.sac.policies import LnMlpPolicy

import Arm2DEnv as ae
#%%
env = ae.ArmModel()
model = SAC(MlpPolicy, env, buffer_size=20000, batch_size=64, gamma=1, learning_rate = 0.001, tau = 0.01, verbose=1)
model.learn(total_timesteps=80000, log_interval=10)

# %%
model.save("twolink-arm")

