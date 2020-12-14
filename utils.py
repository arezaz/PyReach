import os
import numpy as np
import math
from arm_params import *
import matplotlib.pyplot as plt
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy

# some helper functions

# equations of motion for a two link manipulator
#            X_dot = f(t, X, u)
# where X is the state vector of joint coordinates
#
# X(1), X(2) = joint angles
# X(3), X(4) = 1st derivatives of joint angles
# and U is the joint toruques

# state X: [q1, q2, q1d, q2d]
# q1: shoulder, q2: elbow

def ArmDynamicsFun(X, U, FF):
    dum1= m1*l1_c**2 + m2*l2_c**2 + m2*l1**2 + J1+J2
    dum2= 2*m2*l2_c*l1
    dum3= -m2*l1*l2_c*np.sin(X[1])

    I = np.array([[dum1+dum2*np.cos(X[1]) , m2*l2_c**2+J2+0.5*dum2*np.cos(X[1])],
                    [m2*l2_c**2+J2+0.5*dum2*np.cos(X[1]), m2*l2_c**2+J2]])
    C = np.array([[dum3*X[3], dum3*(X[2]+X[3])],
                    [-dum3*X[2],0]])
    if FF == 0:
        dum4 = np.linalg.inv(I)@(np.transpose(U - np.matmul(C, X[2:4])))
    else:
        Jacob = Jacobian(X)
        U_FF = np.matmul(Jacob.transpose(), np.matmul(FF*Curl, np.matmul(Jacob, np.array([[X[2]],[X[3]]]))))
        dum4 = np.linalg.inv(I)@(np.transpose(U + U_FF.transpose() - np.matmul(C, X[2:4])))

    dX_dt = np.array([X[2], X[3], dum4[0], dum4[1]])

    return dX_dt


## jacobian and jacobian derivative
def Jacobian(X):
    # jacobian matrix of two-link arm
    # X: [q1, q2, q1d, q2d]
    return np.array([[-l1*np.sin(X[0])-l2*np.sin(X[0]+X[1]),  -l2*np.sin(X[0]+X[1])],
                        [ l1*np.cos(X[0])+l2*np.cos(X[0]+X[1]),   l2*np.cos(X[0]+X[1])]])

def Jacobian_dot(X):
    # derivative of the jacobian matrix 
    # X: [q1, q2, q1d, q2d]
    return np.array([[-l1*X[2]*np.cos(X[0])-l2*(X[2]+X[3])*np.cos(X[0]+X[1]),
                        -l2*(X[2]+X[3])*np.cos(X[0]+X[1])],
                        [-l1*(X[2])*np.sin(X[0])-l2*(X[2]+X[3])*np.sin(X[0]+X[1]),
                        -l2*(X[2]+X[3])*np.sin(X[0]+X[1])]])

## mapping hand coordinate to joint coordinate
def Hand2Joint(X, *argv):
    # X: [x, y, xd, yd, xdd, ydd]
    # argv: 'pos', 'vel', 'acc'
    out = []

    # angle
    if 'pos' in argv:
        D = (X[0]**2+X[1]**2-l1**2-l2**2)/(2*l1*l2)
        q2 = math.atan2(abs(np.sqrt(1-D**2)),D)
        q1 = math.atan2(X[1],X[0]) - math.atan2(l2*np.sin(q2),(l1 + l2*np.cos(q2)))
        out = np.concatenate((np.array(out), q1, q2), axis=None)

    # angular velocity
    if 'vel' in argv:
        qd = np.matmul(np.linalg.inv(Jacobian(np.array([q1, q2]))),X[2:4])
        out = np.concatenate((np.array(out), qd), axis=None)

    # angular acceleration
    if 'acc' in argv:
        qdd = np.matmul(np.linalg.inv(Jacobian(np.array([q1, q2]))),X[4:6]) - \
                np.matmul( np.linalg.inv(Jacobian(np.array([q1, q2]))),
                np.matmul(Jacobian_dot(np.array([q1, q2, qd[0], qd[1]])), qd) )
        out = np.concatenate((np.array(out), qdd), axis=None)

    return out

## mapping hand joint coordinate to hand coordinate 
def Joint2Hand(X, link = 'lower', *argv):
    # X: [q1, q2, q1d, q2d]
    # argv: 'pos', 'vel'
    out = []

    if link == 'lower':
        if 'pos' in argv:
            x = l1*math.cos(X[0])+l2*math.cos(X[0]+X[1])
            y = l1*math.sin(X[0])+l2*math.sin(X[0]+X[1])
            out = np.concatenate((np.array(out), x, y), axis=None)

        if 'vel' in argv:
            vel = np.matmul(Jacobian(X), np.array([X[2], X[3]]))
            out = np.concatenate((np.array(out), vel), axis=None)
            
    if link == 'upper':
        if 'pos' in argv:
            x = l1*math.cos(X[0])
            y = l1*math.sin(X[0])
            out = np.concatenate((np.array(out), x, y), axis=None)

    return out

## compute distance of a point from straight line
def dist_from_straight(point, origin_hand, target_hand):
    O_x, O_y = origin_hand[0], origin_hand[1]
    T_x, T_y = target_hand[0], target_hand[1]
    x0, y0 = point[0], point[1]

    den = np.sqrt((T_y-O_y)**2+(T_x-O_x)**2)
    ratio = ((T_y - O_y)*x0 - (T_x - O_x)*y0 + T_x*O_y - T_y*O_x)/den
    dist = np.abs(ratio)

    return dist

## generate random target points on a circle
def rand_targ_circle(r):
    theta = np.random.rand() * 2 * np.pi
    return np.array([np.cos(theta) * r, np.sin(theta) * r])

## generate target points on a circle given theta
def targ_circle(r, theta):
    return np.array([np.cos(theta) * r, np.sin(theta) * r])

## generate random fibonacci samples  
def fibonacci_samples(nb_samples, center=np.array([-0.15, 0.30]),
                      ws_high=np.array([-0.15, 0.30])+np.array([0.15, 0.15]), ws_low=np.array([-0.15, 0.30])+np.array([-0.15, -0.0])):
    shift = 1.0
    alpha=1

    radius = np.linalg.norm(ws_high-center)

    ga = np.pi * (3.0 - np.sqrt(5.0))
    # Boundary points
    np_boundary = round(alpha * np.sqrt(nb_samples))
    
    ss = np.zeros((nb_samples,2))
    j = 0
    for i in range(nb_samples):
        if i > nb_samples - (np_boundary + 1):
            r = 1.0
        else:
            r = np.sqrt((i + 0.5) / (nb_samples - 0.5 * (np_boundary + 1)))
        phi   = ga * (i + shift)
        ss[j,:] = np.array([r * np.cos(phi), r * np.sin(phi)])
        j += 1

    ss = radius * ss # scale
    ss = ss + center # add offset
    ss = ss[(ss[:,0]<ws_high[0]) & (ss[:,0]>ws_low[0]) & (ss[:,1]<ws_high[1]) & (ss[:,1]>ws_low[1])] # trim points out of the workspace

    return ss

## visualize arm's motion
def plot_arm(X, env, animate = 'True'):
    hand = np.apply_along_axis(Joint2Hand, 1, np.array(X), 'lower','pos', 'vel')
    # X: joint space states
    plt.figure()

    X = np.array(X)
    for step in range(X.shape[0]):
        lower = np.apply_along_axis(Joint2Hand, 1, np.array(X), 'lower','pos', 'vel')
        upper = np.apply_along_axis(Joint2Hand, 1, np.array(X), 'upper','pos', 'vel')

        plt.clf()
        #Identfying Figure
        plt.axes(xlim=(-.8, .8), ylim=(-0.05, .8))
        plt.plot(hand[:step,0], hand[:step,1],'r.')
        plt.plot([0,upper[step, 0]],[0,upper[step, 1]],'b')
        plt.plot([upper[step, 0],lower[step, 0]],[upper[step, 1],lower[step, 1]],'k')
        plt.plot(env.origin_hand[0], env.origin_hand[1], marker='o', markersize=5, color="red")
        plt.plot(env.target_hand[0], env.target_hand[1], marker='o', markersize=5, color="green")

        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')
        if animate == True:
            plt.pause(0.01*dt)


## save rl algorithms training logs
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, check_points=[], verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.check_points=check_points

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
              # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)

        if self.n_calls in list(self.check_points.keys()):
            self.save_path_cp = os.path.join(self.log_dir, 'best_model', 'checkpoint_'+str(self.n_calls)+'_'+str(self.check_points[self.n_calls]))
            os.makedirs(self.save_path_cp, exist_ok=True)

          # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
              # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Checkpoint at timestep: {}".format(self.num_timesteps))

              # New best model, you could save the agent here
                self.best_mean_reward = mean_reward
                # Example for saving best model
                if self.verbose > 0:
                    print("Saving checkpoint model to {}".format(self.save_path_cp))
                self.model.save(self.save_path_cp)

        return True


## moving average to plot noisy returns
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


## plot noisy returns using moving average
def plot_returns(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.grid()
    plt.show()


## create snapshots of the code for reproductability
import shutil
import os
from datetime import datetime

def snap_code(target_dir):
    source_dir = './'        
    file_names = os.listdir(source_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    target_dir = os.path.join(target_dir, timestamp)
    os.makedirs(target_dir, exist_ok=True)

    target_dir_snap = os.path.join(target_dir, 'snapshot')
    os.makedirs(target_dir_snap, exist_ok=True)    

    for file_name in file_names:
        if file_name.endswith('.py') or file_name .endswith('.ipynb'):
            shutil.copy(os.path.join(source_dir, file_name), target_dir_snap)

    return target_dir
