#%% importing stuff
import Arm2DEnv as ae
import numpy as np
from arm_params import *
from utils import plot_arm
from imp_cntrl import imp_cntrl

#%% load controler
from stable_baselines import SAC, DDPG
#cntrl = SAC.load("twolink-arm-sac")
cntrl = DDPG.load("twolink-arm-ddpg")


#%% arm dynamics and reward function
t = 0.9 # second
tstep = round(t/dt)

env = ae.ArmModel()
x0 = env.reset()

x = np.copy(x0)
X = []
C = []
for step in range(tstep):
    u = imp_cntrl(step*dt, x, t, env) #cntrl.predict(x)[0] #
    x_next,c,done,info = env.step(u)
    X.append(x)
    C.append(c)

    if done:
        break
    else:
        x = np.copy(x_next)

#%% plot trajectory

import matplotlib.pyplot as plt
from utils import Joint2Hand

hand = np.apply_along_axis(Joint2Hand, 1, np.array(X), 'lower','pos', 'vel')

plt.plot(hand[:,0], hand[:,1],'k.')
plt.plot(env.origin_hand[0], env.origin_hand[1], marker='o', markersize=10, color="red")
plt.plot(env.target_hand[0], env.target_hand[1], marker='o', markersize=10, color="green")

plt.axis('equal')
plt.grid()
plt.show()

#%% plot cost
plt.plot(C)
plt.grid()
plt.show()

# %% plot joint states
joint = np.vstack(X)
plt.plot(joint[:,0])
plt.plot(joint[:,1])

plt.grid()
plt.show()

# %% plot arm motion
#%matplotlib qt
plot_arm(X, env, True)

# %%
