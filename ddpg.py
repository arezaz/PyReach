#%% train soft actor-critic
import gym
import numpy as np
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.noise import NormalActionNoise

import Arm2DEnv as ae
#%% sac
env = ae.ArmModel()

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

model = DDPG(LnMlpPolicy, env, verbose=1, gamma=0.98, tau=0.01,
             actor_lr=0.0001, critic_lr=0.001, action_noise=action_noise,
             buffer_size=int(5E6), batch_size=128, random_exploration=0.0)

model.learn(total_timesteps=int(4E4))#(total_timesteps=int(4E5))

# %%
model.save("twolink_arm_ddpg")