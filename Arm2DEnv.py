
#%%
import numpy as np
import torch
import scipy.integrate as itg 
import gym

from utils import Jacobian, Jacobian_dot, Hand2Joint, Joint2Hand, dist_from_straight, rand_targ_circle
from arm_params import *

#%%

# equations of motion for a two link manipulator
#            X_dot = f(t, X, u)
# where X is the state vector of joint coordinates
#
# X(1), X(2) = joint angles
# X(3), X(4) = 1st derivatives of joint angles
# and U is the joint toruques

# state X: [q1, q2, q1d, q2d]
# q1: shoulder, q2: elbow

def ArmDynamicsFun(X, U):
    dum1= m1*l1_c**2 + m2*l2_c**2 + m2*l1**2 + J1+J2
    dum2= 2*m2*l2_c*l1
    dum3= -m2*l1*l2_c*np.sin(X[1])

    I = np.array([[dum1+dum2*np.cos(X[1]) , m2*l2_c**2+J2+0.5*dum2*np.cos(X[1])],
                    [m2*l2_c**2+J2+0.5*dum2*np.cos(X[1]), m2*l2_c**2+J2]])
    C = np.array([[dum3*X[3], dum3*(X[2]+X[3])],
                    [-dum3*X[2],0]])
    dum4 = np.linalg.inv(I)@(np.transpose(U - np.matmul(C, X[2:4])))
    dX_dt = np.array([X[2], X[3], dum4[0], dum4[1]])

    return dX_dt

#%%
# arm movement constraints :
# The Human Arm Kinematics and Dynamics
# During Daily Activities – Toward a 7 DOF
# Upper Limb Powered Exoskeleton 

arm_cnstr = {
    'shoulder':{
        'UB_U': 10.0,
        'LB_U': -10.0,
        'UB_X': np.deg2rad(135),
        'LB_X': np.deg2rad(-60)
    },

    'elbow':{
        'UB_U': 10.0,
        'LB_U':-10.0,
        'UB_X': np.deg2rad(175),
        'LB_X': np.deg2rad(0)
    }
}

# arm environment gym class
class ArmModel(gym.Env):
    
    def __init__(self):
        # arm biophysical constraints
        self.arm_cnstr = arm_cnstr
        # center of the workspace, initial position of the arm for experiments
        self.wsapce_center = np.array([-0.15, 0.30]) 
        # workspace: a [0.5x0.35] rectangle on the center
        self.wspace = gym.spaces.Box(
            low = self.wsapce_center + np.array([-0.15, -0.0]),
            high = self.wsapce_center + np.array([0.15, 0.15])
        )

        self.observation_space = gym.spaces.Box(
            low = np.array([self.arm_cnstr['shoulder']['LB_X'], self.arm_cnstr['elbow']['LB_X'], -10.0, -10.0]),
            high = np.array([self.arm_cnstr['shoulder']['UB_X'], self.arm_cnstr['elbow']['UB_X'], +10.0, +10.0])
        )

        self.action_space = gym.spaces.Box(
            low  = np.array([self.arm_cnstr['shoulder']['LB_U'], self.arm_cnstr['elbow']['LB_U']]),
            high = np.array([self.arm_cnstr['shoulder']['UB_U'], self.arm_cnstr['elbow']['UB_U']])
        )
        self.dt = dt
        self.metadata = {'render.modes': []}
        self.flag_done = False
        self.state = None
        
    def set_origin(self, position):
        self.origin_hand = np.array([position[0], position[1]]) # initially set the origin to the center of the workspace
        self.origin_joint = np.concatenate((Hand2Joint(self.origin_hand, 'pos'), 0.0, 0.0), axis=None)

    def set_target(self, position):
        # target position in hand space, assuming target velocity is always zero
        # [x y xd yd]
        self.target_hand = np.array([position[0], position[1], 0.0, 0.0])
        # [q1 q2 q1d q2d]
        self.target_joint = np.concatenate((Hand2Joint(self.target_hand, 'pos'), 0.0, 0.0), axis=None)
        
    def is_feasible(self, X, U):
        # making sure arm's constrains are met
        q1_feas = np.min(X[0]>=self.arm_cnstr['shoulder']['LB_X']) and np.min(X[0] <= self.arm_cnstr['shoulder']['UB_X'])
        q2_feas = np.min(X[1]>=self.arm_cnstr['elbow']['LB_X']) and np.min(X[1] <= self.arm_cnstr['elbow']['UB_X'])

        u1_feas = np.min(U[0]>=self.arm_cnstr['shoulder']['LB_U']) and np.min(U[0] <= self.arm_cnstr['shoulder']['UB_U'])
        u2_feas = np.min(U[1]>=self.arm_cnstr['elbow']['LB_U']) and np.min(U[1] <= self.arm_cnstr['elbow']['UB_U'])

        return q1_feas and q2_feas and u1_feas and u2_feas

    def ArmDynamics(self,t,X,U):
        dX_dt = np.array(ArmDynamicsFun(X,U)).squeeze()
        return dX_dt

    def cost(self, X_joint, U):
        # X: [q1, q2, q1d, q2e]
        # convert to hand space
        X_hand = Joint2Hand(X_joint, 'lower', 'pos', 'vel')
        X_t_hand = self.target_hand
        X_t_joint = self.target_joint

        reward = 0
        eps = 0.005
        lmbd = 0.5
        dist_p = np.linalg.norm((X_hand[:2]-X_t_hand[:2]), ord=2) # position
        dist_o = np.linalg.norm((X_joint[:2]-X_t_joint[:2]), ord=2) # orientation
        dist = lmbd*dist_p + (1-lmbd)*dist_o

        if dist_p > eps and dist_o > eps:
            reward += -dist
        else:
            reward += 1

        return reward

    def step_from_state(self,X,U):
        done = not self.is_feasible(X,U)
        c = self.cost(X,U)
        info = {}
        if done:
            c = -1 
            return X,c,done,info

        res = itg.solve_ivp(self.ArmDynamics,(0,dt),X,args=(U,))
        X_next = res.y[:,-1]

        done = not self.is_feasible(X_next,U)
        if done:
            c = -1
        done = self.flag_done
        return X_next,c,done,info
    
    def step(self,U):
        X_next,c,done,info = self.step_from_state(self.X,U)
        self.state = np.copy(X_next)
        return X_next,c,done,info

    def reset(self):
        self.flag_done = False

        rand_origin = self.wsapce_center #0.2*self.wspace.sample()  #0.4*self.wspace.sample() #self.wsapce_center #+ 0.4*self.wspace.sample()
        self.set_origin(rand_origin) 

        rand_targ = self.wsapce_center+rand_targ_circle(0.1) #0.9*self.wspace.sample() #
        self.set_target(rand_targ)

        self.state = Hand2Joint(np.array([self.origin_hand[0], self.origin_hand[1], 0.0, 0.0]), 'pos', 'vel')
        return np.copy(self.state)


#%%