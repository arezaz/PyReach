
#%%
import numpy as np
import torch
import scipy.integrate as itg 
import gym

from utils import ArmDynamicsFun, Jacobian, Jacobian_dot, Hand2Joint, Joint2Hand, dist_from_straight, rand_targ_circle, targ_circle, fibonacci_samples
from arm_params import *

#%%
# arm movement constraints :
# The Human Arm Kinematics and Dynamics
# During Daily Activities – Toward a 7 DOF
# Upper Limb Powered Exoskeleton 

_torque_thres = 25

arm_cnstr = {
    'shoulder':{
        'UB_U': 1*_torque_thres,
        'LB_U': -1*_torque_thres,
        'UB_X': np.deg2rad(135),
        'LB_X': np.deg2rad(-60)
    },

    'elbow':{
        'UB_U': 1*_torque_thres,
        'LB_U': -1*_torque_thres,
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
        _ws_low = self.wsapce_center + np.array([-0.15, -0.0])
        _ws_high = self.wsapce_center + np.array([0.15, 0.15])

        self.wspace = gym.spaces.Box(
            low = _ws_low,
            high = _ws_high
        )
        # equidistant random points that covers workspace about the 
        self.fibo_ws = fibonacci_samples(nb_samples=100, center=self.wsapce_center, ws_high=_ws_high, ws_low=_ws_low)

        self.mode = 'train' # 'train' or 'eval'

        _joint_high = np.array([self.arm_cnstr['shoulder']['LB_X'], self.arm_cnstr['elbow']['LB_X']])
        _joint_low = np.array([self.arm_cnstr['shoulder']['UB_X'], self.arm_cnstr['elbow']['UB_X']])
        _joint_vel_thresh = np.array([10.]*2)
        _hand_vel_thresh = np.array([1.]*2)
        _loc_high = np.concatenate(([self.wspace.high.max()], [self.wspace.high.max()]))
        _loc_low = np.concatenate(([self.wspace.high.min()], [self.wspace.high.min()]))

        self.observation_space = gym.spaces.Box(
            low = np.concatenate((_joint_low, -1*_joint_vel_thresh, _loc_low, -1*_hand_vel_thresh, _loc_low, _loc_low, [0.])),
            high = np.concatenate((_joint_high, +1*_joint_vel_thresh, _loc_high, +1*_hand_vel_thresh, _loc_high, _loc_high, [1.])),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low  = np.array([self.arm_cnstr['shoulder']['LB_U'], self.arm_cnstr['elbow']['LB_U']]),
            high = np.array([self.arm_cnstr['shoulder']['UB_U'], self.arm_cnstr['elbow']['UB_U']])
        )

        self.dt = dt
        self.metadata = {'render.modes': []}
        self.flag_reached = False
        self.state = None
        self.VISION = None
        self.obs = None
        self.FF = 0
        self.iter = 0
        self._numcalls = 0

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
        q1_feas = X[0]>=self.arm_cnstr['shoulder']['LB_X'] and X[0] <= self.arm_cnstr['shoulder']['UB_X']
        q2_feas = X[1]>=self.arm_cnstr['elbow']['LB_X'] and X[1] <= self.arm_cnstr['elbow']['UB_X']

        u1_feas = U[0]>=self.arm_cnstr['shoulder']['LB_U'] and U[0] <= self.arm_cnstr['shoulder']['UB_U']
        u2_feas = U[1]>=self.arm_cnstr['elbow']['LB_U'] and U[1] <= self.arm_cnstr['elbow']['UB_U']

        return q1_feas and q2_feas and u1_feas and u2_feas

    def ArmDynamics(self,t,X,U):
        dX_dt = np.array(ArmDynamicsFun(X,U,self.FF)).squeeze()
        return dX_dt

    def cost(self, X_joint, U):
        # X: [q1, q2, q1d, q2e]
        # convert to hand space
        X_hand = Joint2Hand(X_joint, 'lower', 'pos', 'vel')
        X_t_hand = self.target_hand
        X_t_joint = self.target_joint

        reward = 0
        eps = 0.005 + 0.05/(self.iter+1)**0.8 #**0.8 # shrink the epsilon circle while iterating timesteps
        lmbd = 0.5
        dist_p = np.linalg.norm((X_hand[:2]-X_t_hand[:2]), ord=2) # position
        dist_o = np.linalg.norm((X_joint[:2]-X_t_joint[:2]), ord=2) # orientation
        dist = lmbd*dist_p + (1-lmbd)*dist_o

        if dist_p > eps and dist_o > eps:
            reward += -dist
        else:
            reward += 1
            self.flag_reached = True

        return reward

    def step_from_state(self,state,U):
        done = not self.is_feasible(state,U)
        c = self.cost(state,U)
        info = {}
        if done:
            c = -5 
            return state,c,done,info

        res = itg.solve_ivp(self.ArmDynamics,(0,dt),state,args=(U,))
        state_next = res.y[:,-1]

        done = not self.is_feasible(state_next,U)
        if done:
            c = -5
        done = done or self.flag_reached
        return state_next,c,done,info
    
    def step(self,U):
        # obs: [states(q1, q2, q1d, q2d), (hand_x, hand_y, hand_xd, handyd)], goal_x, goal_y, reached_goal
        #self.state = self.obs[0:4] # q1, q2, q1d, q2d
        state_next,c,done,info = self.step_from_state(self.state,U)

        self.state = np.copy(state_next)
        self.VISION = np.concatenate((self.state, Joint2Hand(state_next, 'lower', 'pos', 'vel')))
        self.obs = np.concatenate((self.VISION, self.target_hand[0:2], self.origin_hand, [1. if self.flag_reached else 0.]))

        obs_next = np.copy(self.obs)
        self.iter += 1
        return obs_next,c,done,info

    def reset(self):
        self.flag_reached = False
        self.iter = 0
        self._numcalls = self._numcalls + 1 # keeping track of call numbers for curriculum learning
        
        # target random around ws center:
        #rand_origin = self.wsapce_center
        #self.set_origin(rand_origin)
        #rand_targ = self.wsapce_center+rand_targ_circle(0.1) # random target about the center of the workspace
        #self.set_target(rand_targ)

        # fibonacci start and end position:
        if self.mode == 'train_fibo_ranc':
            origin_idx, targ_idx = np.random.choice(self.fibo_ws.shape[0], 2, replace = False)
            rand_origin = self.fibo_ws[origin_idx,:]
            self.set_origin(rand_origin)
            rand_targ = self.fibo_ws[targ_idx,:] # random target about the center of the workspace
            self.set_target(rand_targ)

        if self.mode == 'train_curriculum':
            _numcalls = self._numcalls
            _cur_1_call = self.curriculum[0]
            _cur_2_call = self.curriculum[1]
            _cur_3_call = self.curriculum[2]

            if _numcalls <= _cur_1_call:
                self.FF = 0
                # Cur-1: Baseline
                _rampup = _numcalls/_cur_1_call
                _origin = self.wsapce_center+ np.random.uniform(-0.01*_rampup, 0.01*_rampup)
                self.set_origin(_origin)
                _theta = np.random.randint(8)*np.pi/4 + np.random.uniform(-0.1*_rampup, +0.1*_rampup) #(_numcalls%8)*np.pi/4 + np.random.uniform(-0.1*_rampup, +0.1*_rampup) #
                _targ_r = 0.1 + np.random.uniform(-0.02*_rampup, 0.02*_rampup)
                self.set_target(_origin + targ_circle(_targ_r, _theta))

            elif self._numcalls <= _cur_2_call:
                self.FF = 15
                # Cur-1: FF 
                _rampup = (_numcalls-_cur_1_call)/(_cur_2_call-_cur_1_call)
                _origin = self.wsapce_center+ np.random.uniform(-0.01*_rampup, 0.01*_rampup)
                self.set_origin(_origin)
                _theta = np.random.randint(8)*np.pi/4 + np.random.uniform(-0.1*_rampup, +0.1*_rampup) #(_numcalls%8)*np.pi/4 + np.random.uniform(-0.1*_rampup, +0.1*_rampup) #
                _targ_r = 0.1 + np.random.uniform(-0.02*_rampup, 0.02*_rampup)
                self.set_target(_origin + targ_circle(_targ_r, _theta))

            elif self._numcalls > _cur_2_call:
                self.FF = 0
                # Cur-3: Washout
                _rampup = (_numcalls-_cur_2_call)/(_cur_3_call-_cur_2_call)
                _origin = self.wsapce_center+ np.random.uniform(-0.01*_rampup, 0.01*_rampup)
                self.set_origin(_origin)
                _theta = np.random.randint(8)*np.pi/4 + np.random.uniform(-0.1*_rampup, +0.1*_rampup) #(_numcalls%8)*np.pi/4 + np.random.uniform(-0.1*_rampup, +0.1*_rampup) #
                _targ_r = 0.1 + np.random.uniform(-0.02*_rampup, 0.02*_rampup)
                self.set_target(_origin + targ_circle(_targ_r, _theta))

        self.state = Hand2Joint(np.array([self.origin_hand[0], self.origin_hand[1], 0.0, 0.0]), 'pos', 'vel')
        self.VISION = np.concatenate((self.state, Joint2Hand(self.state, 'lower', 'pos', 'vel')))
        self.obs = np.concatenate((self.VISION, self.target_hand[0:2], self.origin_hand, [1. if self.flag_reached else 0.]))
        
        return np.copy(self.obs)

#%%