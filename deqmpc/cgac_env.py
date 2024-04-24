# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#from numpy.lib.function_base import angle
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from envs_v1 import OneLinkCartpoleEnv, TwoLinkCartpoleEnv
from envs import PendulumEnv

import torch
import torch.nn as nn
import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

class PytorchEnv(nn.Module):

    def __init__(self, render=False, env_name='pendulum', device='cuda:0', num_envs=4096, seed=0, episode_length=1000, no_grad=True, stochastic_init=False, MM_caching_frequency = 1, early_termination = True):
        super(PytorchEnv, self).__init__()

        self.env_name = env_name
        self.device = device
        if self.env_name == 'pendulum':
            self.env = PendulumEnv(batch_size=num_envs, episode_length=episode_length)
        elif self.env_name == 'cartpole1link':
            self.env = OneLinkCartpoleEnv(batch_size=num_envs, episode_length=episode_length)
        elif self.env_name == 'cartpole2link':
            self.env = TwoLinkCartpoleEnv(batch_size=num_envs, episode_length=episode_length)

        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space
        self.progress_buf = torch.zeros((num_envs), device = self.device, dtype = torch.long)
        self.obs_buf = torch.zeros((num_envs, self.obs_space.shape[0]), device = self.device, dtype = torch.float)
        self.progress_buf_mask = torch.zeros((num_envs), device = self.device, dtype = torch.bool)
        self.ep_lens = torch.zeros((num_envs), device = self.device, dtype = torch.long)
        self.max_episode_length = episode_length
        self.reset_buf = torch.zeros((num_envs), device = self.device, dtype = torch.bool)
        self.reset_goal_buf = torch.zeros((num_envs), device = self.device, dtype = torch.float)
        self.early_termination = early_termination

        self.env.reset()

    def step(self, actions):
        actions = actions.view((self.num_envs, self.num_actions))
        self.obs_buf, self.rew_buf, done, info = self.env.step(actions)
        self.reset_buf = done

        self.progress_buf += 1
        self.num_frames += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                }

        if len(env_ids) > 0:
           self.reset(env_ids)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def reset(self, env_ids = None, force_reset = True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = self.env.reset(env_ids)
        return self.obs_buf
    
    '''
    cut off the gradient from the current state to previous states
    '''
    def clear_grad(self, checkpoint = None):
        pass

    def get_checkpoint(self):
        checkpoint = {}
        checkpoint['obs'] = self.obs_buf.clone()
        return checkpoint

    def calculateObservations(self):
        self.obs_buf = self.env.state

    def calculateReward(self, rew):
        self.rew_buf = rew
        
        # reset agents
        self.reset_buf = torch.where(self.progress_buf > self.episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        if self.early_termination:
            self.reset_buf = torch.where(self.obs_buf[:, 0] < self.termination_height, torch.ones_like(self.reset_buf), self.reset_buf)