#%% train soft actor-critic
import gym
import numpy as np
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.sac.policies import LnMlpPolicy

import Arm2DEnv as ae
#%%
env = ae.ArmModel()
model = SAC(LnMlpPolicy, env, buffer_size=int(5E5), batch_size=128, gamma=0.98, learning_rate = 0.001, tau = 0.01, verbose=1)
model.learn(total_timesteps=int(3E6), log_interval=10)

# %%
model.save("twolink-arm")

