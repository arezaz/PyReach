#%% importing stuff
import Arm2DEnv as ae
import numpy as np
from arm_params import *
from utils import plot_arm

#%% arm dynamics and reward function
t = 0.8 # second
tstep = round(0.8/dt)

env = ae.ArmModel()
env.origin_hand = np.array([-0.0,  0.3])
env.set_target(env.wsapce_center+np.array([0.1, 0]))

x0 = env.reset()

x = np.copy(x0)
X = []
C = []
for step in range(tstep):
    u = np.array([6, -6])
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
%matplotlib qt
plot_arm(X)

# %%
