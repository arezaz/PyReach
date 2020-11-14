import numpy as np
import math
from arm_params import *

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

