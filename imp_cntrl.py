import numpy as np
from utils import Hand2Joint, Jacobian
from arm_params import *

def min_Jerk(t, tf, env, d):
    Xo, Xf = env.origin_hand[:2], env.target_hand[:2]
    to = 0
    T = (t-to)/(tf-to)
    tfo = tf - to

    if d == 'pos':
        # function to calculate min jerk position
        if t <= to:
            Xd = Xo
        elif (t > to and t <= tf):
            Xd0 = Xo[0] + (Xo[0] - Xf[0]) * ( 15*T**4 - 6*T**5 - 10*T**3 )
            Xd1 = Xo[1] + (Xo[1] - Xf[1]) * ( 15*T**4 - 6*T**5 - 10*T**3 )
            Xd = np.array([Xd0, Xd1])
        else:
            Xd = Xf
        return Xd

    if d == 'vel':
        #function to calculate min jerk velocity
        if t <= to:
            Xd = np.array([[0],[0]])
        elif (t > to and t <= tf):
            Xd0 = (Xo[0] - Xf[0]) * ( 60*T**3 - 30*T**4 - 30*T**2 )/tfo
            Xd1 = (Xo[1] - Xf[1]) * ( 60*T**3 - 30*T**4 - 30*T**2 )/tfo
            Xd = np.array([Xd0, Xd1])
        else:
            Xd = np.array([[0],[0]])
        return Xd.squeeze()

    if d == 'acc':
        #function to calculate min jerk acceleration
        if t <= to:
            Xd = np.array([[0],[0]])
        elif (t > to and t <= tf):
            Xd0 = (Xo[0] - Xf[0]) * (180*T**2 - 120*T**3 - 60*T)/(tfo*tfo)
            Xd1 = (Xo[1] - Xf[1]) * (180*T**2 - 120*T**3 - 60*T)/(tfo*tfo)
            Xd = np.array([Xd0, Xd1])
        else:
            Xd = np.array([[0],[0]])
        return Xd.squeeze()


def imp_cntrl(t, x, Tf, env):

    Jacob = Jacobian(x)

    hand_mj = np.array([min_Jerk(t, Tf, env, 'pos'),
                        min_Jerk(t, Tf, env, 'vel'),
                        min_Jerk(t, Tf, env, 'acc')])
    hand_mj = hand_mj.flatten()
    x_desired = Hand2Joint(hand_mj, 'pos', 'vel', 'acc')

    dum1= m1*l1_c**2 + m2*l2_c**2 + m2*l1**2 + J1+J2
    dum2= 2*m2*l2_c*l1

    I_d = np.array([[dum1+dum2*np.cos(x_desired[1]) , m2*l2_c**2+J2+0.5*dum2*np.cos(x_desired[1])],
                 [m2*l2_c**2+J2+0.5*dum2*np.cos(x_desired[1]), m2*l2_c**2+J2]])
    dumd= -m2*l1*l2_c*np.sin(x_desired[1])
    C_d = np.array([[dumd*x_desired[3], dumd*(x_desired[2]+x_desired[3])],
                    [-dumd*x_desired[2],0]])
   
    feedback = Kp@(x_desired[0:2] - x[0:2])+ Kd@(x_desired[2:4] - x[2:4])
    feedforward = C_d@x_desired[2:4] + I_d@x_desired[4:]

    command = feedforward + feedback

    return command