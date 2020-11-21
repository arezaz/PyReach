import numpy as np
import math
from arm_params import *
import matplotlib.pyplot as plt

# some helper functions
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

def dist_from_straight(point, origin_hand, target_hand):
    O_x, O_y = origin_hand[0], origin_hand[1]
    T_x, T_y = target_hand[0], target_hand[1]
    x0, y0 = point[0], point[1]

    den = np.sqrt((T_y-O_y)**2+(T_x-O_x)**2)
    ratio = ((T_y - O_y)*x0 - (T_x - O_x)*y0 + T_x*O_y - T_y*O_x)/den
    dist = np.abs(ratio)

    return dist

def rand_targ_circle(r):
    theta = np.random.rand() * 2 * np.pi
    return np.array([np.cos(theta) * r, np.sin(theta) * r])

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