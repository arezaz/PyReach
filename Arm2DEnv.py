
#%%
import numpy as np
import casadi as cdi
import torch
import scipy.integrate as itg 
import gym

from utils import Jacobian, Jacobian_dot, Hand2Joint, Joint2Hand
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
X = cdi.SX.sym('X', 4)

# command U: joint torques
U = cdi.SX.sym('U', 2)


q1 = X[0]
q2 = X[1]
q1d = X[2]
q2d = X[3]
s2 = cdi.sin(q2)
c2 = cdi.cos(q2)

# two-arm motion dynamic formulation
# I(X)*d2X_dt2 + C(X, dX_dt)*d_X = U
I11_a = m1*l1_c**2
I11_b = l1**2 + l2_c**2 + 2*l1*l2_c*c2
I11 = I11_a + m2*I11_b + J1 + J2
I12 = m2*( l2_c**2 + l1*l2_c*c2 ) + J2
I22 = m2*l2_c**2 + J2
h = -m2*l1*l2_c*s2

den = I11 - I12*I12/I22
part = U[1] + h*q1d**2

q1dd = ( U[0] - h*q2d**2 - 2*h*q1d*q2d - I12*part/I22 ) / den
q2dd = ( U[1] + h*q1d**2 - I12*q1dd ) / I22

dX_dt = cdi.vcat([q1d, q2d, q1dd, q2dd])

# arm dynamics model
ArmDynamicsFun = cdi.Function('ArmDynamics',[X,U],[dX_dt])

#%%
# arm movement constraints :
# The Human Arm Kinematics and Dynamics
# During Daily Activities – Toward a 7 DOF
# Upper Limb Powered Exoskeleton 

arm_cnstr = {
    'shoulder':{
        'UB_U': 6.0,
        'LB_U': -6.0,
        'UB_X': np.deg2rad(85),
        'LB_X': np.deg2rad(-60)
    },

    'elbow':{
        'UB_U': 6.0,
        'LB_U':-6.0,
        'UB_X': np.deg2rad(170),
        'LB_X': np.deg2rad(0)
    }
}

# arm environment gym class
class ArmModel:

    
    def __init__(self):
        # center of the workspace, initial position of the arm for experiments
        self.wsapce_center = np.array([-0.15, 0.30]) 
        # workspace: a [0.5x0.35] rectangle on the center
        self.wspace = gym.spaces.Box(
            low = np.array([self.wsapce_center]) + np.array([-0.25, -0.1]),
            high = np.array([self.wsapce_center]) + np.array([0.25, 0.25])
        )

        # arm biophysical constraints
        self.arm_cnstr = arm_cnstr
        self.action_space = gym.spaces.Box(
            low  = np.array([self.arm_cnstr['shoulder']['LB_U'], self.arm_cnstr['elbow']['LB_U']]),
            high = np.array([self.arm_cnstr['shoulder']['UB_U'], self.arm_cnstr['elbow']['UB_U']])
        )
        self.dt = dt

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

    def cost(self, X, U):
        # X: [q1, q2, q1d, q2e]
        # convert to hand space
        X_hand = Joint2Hand(X, 'lower', 'pos', 'vel')
        X_target = self.target_hand

        cost_dist = np.linalg.norm(X_hand[0:2]-X_target[0:2])/l1 # normalized with arm length
        cost_effort = (np.absolute(U[0])+np.absolute(U[1]))/(self.arm_cnstr['shoulder']['UB_U']+self.arm_cnstr['elbow']['UB_U']) #normalized by max torque

        return cost_dist + cost_effort

    def step_from_state(self,X,U):
        done = not self.is_feasible(X,U)
        c = self.cost(X,U)
        info = {}
        if done:
            c = self.cost(np.array([self.wsapce_center[0], self.wsapce_center[1], 0.0, 0.0]), np.array([0.0, 0.0]))
            return X,c,done,info

        res = itg.solve_ivp(self.ArmDynamics,(0,dt),X,args=(U,))
        X_next = res.y[:,-1]

        done = not self.is_feasible(X_next,U)
        if done:
            c = self.cost(np.array([self.wsapce_center[0], self.wsapce_center[1], 0.0, 0.0]), np.array([0.0, 0.0]))
        return X_next,c,done,info
    
    def step(self,U):
        X_next,c,done,info = self.step_from_state(self.X,U)
        self.X = np.copy(X_next)
        return X_next,c,done,info

    def reset(self):
        self.X = Hand2Joint(np.array([self.wsapce_center[0], self.wsapce_center[1], 0.0, 0.0]), 'pos', 'vel')
        return np.copy(self.X)

    






#%%