#!/usr/bin/env python3

import torch
from torch.autograd import Function, Variable, grad
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr
import numpy.testing as npt
# from numpy.testing import dec

# import cvxpy as cp

import numdifftools as nd

import gc
import os

import sys
sys.path.insert(0, '/home/swaminathan/Workspace/diffmpc/mpc.pytorch/')
from mpc import mpc, util, pnqp
from mpc.dynamics import NNDynamics, AffineDynamics
from mpc.lqr_step import LQRStep
from mpc.mpc import GradMethods, QuadCost, LinDx

from envs import PendulumEnv, PendulumDynamics


class PendulumExpert:
    def __init__(self, env, type='mpc'):
        """
        Initialize the MPC controller with the necessary parameters.

        Args:
            env: The PendulumEnv environment.
            type: The type of controller to use. Can be 'mpc' or 'ppo' or 'sac'.        
        """

        self.type = type

        if self.type == 'mpc':
            self.T = 30
            self.goal_state = torch.Tensor([0., 0.])
            self.goal_weights = torch.Tensor([10., 0.1])
            self.ctrl_penalty = 0.001
            self.mpc_eps = 1e-3
            self.linesearch_decay = 0.2
            self.max_linesearch_iter = 5
            self.nx = env.observation_space.shape[0]
            self.nu = env.action_space.shape[0]
            self.bsz = 1

            self.u_lower = torch.tensor(env.action_space.low, dtype=torch.float32)
            self.u_upper = torch.tensor(env.action_space.high, dtype=torch.float32)

            self.qp_iter = 10
            self.u_init = torch.zeros(self.T, self.bsz, self.nu)
            self.q = torch.cat((
                self.goal_weights,
                self.ctrl_penalty*torch.ones(self.nu)
            ))
            self.px = -torch.sqrt(self.goal_weights)*self.goal_state
            self.p = torch.cat((self.px, torch.zeros(self.nu)))
            self.Q = torch.diag(self.q).unsqueeze(0).unsqueeze(0).repeat(
                self.T, self.bsz, 1, 1
            )
            self.p = self.p.unsqueeze(0).repeat(self.T, self.bsz, 1)

            self.ctrl = mpc.MPC(
                self.nx, self.nu, self.T, 
                u_lower=self.u_lower.item(), u_upper=self.u_upper.item(), 
                lqr_iter=self.qp_iter, exit_unconverged=False, 
                eps=1e-2, n_batch=self.bsz, backprop=False, 
                verbose=0, u_init=self.u_init, 
                grad_method=mpc.GradMethods.AUTO_DIFF, 
            )
            self.cost = mpc.QuadCost(self.Q, self.p)

    def optimize_action(self, state):
        """Solve the MPC problem for the given state."""
        # ipdb.set_trace()
        nominal_actions = self.ctrl(state, self.cost, env.dynamics)
        return nominal_actions[0]  # Return the first action in the optimal sequence


def test_mpc(env):
    """
    Get expert trajectories for pendulum environment using MPC for trajectory optimization.
    Args:
        env: The PendulumEnv environment.
        mpc_controller: The PendulumExpert.
        num_traj: Number of trajectories to save.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples.
    """
    mpc_controller = PendulumExpert(env)
    trajectories = []
    state = env.reset()  # Reset environment to a new initial state
    traj = []
    done = False
    action = mpc_controller.optimize_action(torch.tensor(state, dtype=torch.float32).view(1, -1))

    return action

if __name__ == '__main__':
    print("Starting!")
    # ipdb.set_trace()
    env = PendulumEnv(stabilization=False)
    test_mpc(env)