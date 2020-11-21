import os
import numpy as np
import math
from arm_params import *
import matplotlib.pyplot as plt
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy

# some helper functions

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

## visualize arm's motion
def plot_arm(X, env, motion = 'True'):
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
        plt.plot(env.origin_hand[0], env.origin_hand[1], marker='o', markersize=10, color="red")
        plt.plot(env.target_hand[0], env.target_hand[1], marker='o', markersize=10, color="green")

        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')
        if motion == True:
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
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

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

        return True