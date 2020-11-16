#%%
import Arm2DEnv as ae
import numpy as np
from arm_params import *
from utils import plot_arm

t = 0.8 # second
tstep = round(0.8/dt)
env = ae.ArmModel()
x0 = env.reset()
env.set_target(env.wsapce_center+np.array([0.1, 0]))
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

#%%
import matplotlib.pyplot as plt
from utils import Joint2Hand

hand = np.apply_along_axis(Joint2Hand, 1, np.array(X), 'lower','pos', 'vel')

plt.plot(hand[:,0], hand[:,1],'k.')
plt.axis('equal')
plt.grid()
plt.show()

#%%
plt.plot(C)
plt.grid()
plt.show()

# %%

joint = np.vstack(X)
plt.plot(joint[:,0])
plt.plot(joint[:,1])

plt.grid()
plt.show()

# %%


lower = np.apply_along_axis(Joint2Hand, 1, np.array(X), 'lower','pos', 'vel')
upper = np.apply_along_axis(Joint2Hand, 1, np.array(X), 'upper','pos', 'vel')

fig = plt.figure()
ax = fig.add_subplot(111)
    
for counter in range(upper.shape[0]): 
    ax.plot([0,upper[counter, 0]],[0,upper[counter, 1]],'b')
    ax.plot([upper[counter, 0],lower[counter, 0]],[upper[counter, 1],lower[counter, 1]],'k')
ax.plot(hand[:,0], hand[:,1],'r.')
ax.axis('equal')
ax.grid()

# %%
%matplotlib qt
plot_arm(X)
# %%
