import math
import time

import numpy as np
import torch
import torch.autograd as autograd
import qpth.qp_wrapper as mpc
import ipdb
import os
from envs import PendulumEnv, PendulumDynamics


class MPCPendulumController:
    def __init__(self, env):
        """
        Initialize the MPC controller with the necessary parameters.
        Args:
            nx (int): Number of state dimensions.
            nu (int): Number of control dimensions.
            T (int): Time horizon for the MPC.
            u_lower (float): Lower bound for control signal.
            u_upper (float): Upper bound for control signal.
            max_iter (int): Maximum iterations for the QP solver.
            bsz (int): Batch size for MPC optimization.
            u_init (float): Initial guess for control signal.
            Q (np.ndarray): State cost matrix.
            p (np.ndarray): Control cost matrix.
        """
        nx = env.observation_space.shape[0]
        nu = env.action_space.shape[0]
        T = env.T
        u_lower = torch.tensor(env.action_space[:, 0], dtype=torch.float32)
        u_upper = torch.tensor(env.action_space[:, 1], dtype=torch.float32)
        max_iter = 1
        bsz = 1
        u_init = torch.zeros(T, bsz, nu)
        q,p = env.get_true_obj()
        Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
            T, bsz, 1, 1
        )
        p = p.unsqueeze(0).repeat(T, bsz, 1)
        self.ctrl = mpc.MPC(
            nx, nu, T, 
            u_lower=u_lower, u_upper=u_upper, 
            qp_iter=max_iter, exit_unconverged=False, 
            eps=1e-2, n_batch=bsz, backprop=False, 
            verbose=0, u_init=u_init, 
            grad_method=mpc.GradMethods.AUTO_DIFF, 
            solver_type='dense'
        )
        self.cost = mpc.QuadCost(Q, p)

    def optimize_action(self, state):
        """Solve the MPC problem for the given state."""
        # ipdb.set_trace()
        nominal_states, nominal_actions = self.ctrl(state, self.cost, PendulumDynamics())
        return nominal_actions[0]  # Return the first action in the optimal sequence

def get_pendulum_expert_traj(env, num_traj):
    """
    Get expert trajectories for pendulum environment using MPC for trajectory optimization.
    Args:
        env: The PendulumEnv environment.
        mpc_controller: The MPCPendulumController.
        num_traj: Number of trajectories to save.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples.
    """
    mpc_controller = MPCPendulumController(env)
    trajectories = []
    for _ in range(num_traj):
        state = env.reset()  # Reset environment to a new initial state
        traj = []
        done = False
        while not done:
            action = mpc_controller.optimize_action(torch.tensor(state, dtype=torch.float32).view(1, -1))
            next_state, _, done, _ = env.step(action)
            traj.append((state, action.numpy()))
            state = next_state
            # if len(traj) >= env.T:
            #     ipdb.set_trace()
        print(f"Trajectory length: {len(traj)}")
        trajectories.append(traj)
    return trajectories

def save_expert_traj(env, num_traj):
    """Save expert trajectories to a file.

    Args:
        env: gym environment
        num_traj: number of trajectories to save
    """

    ## use env name to choose which function to use to get expert trajectories
    if env.spec_id == 'Pendulum-v0':
        expert_traj = get_pendulum_expert_traj(env, num_traj)
    else:
        raise NotImplementedError
    
    ## save expert trajectories to a file in data folder
    if os.path.exists('data') == False:
        os.makedirs('data')
    np.save(f'data/expert_traj-{env.sped_id}.npy', expert_traj)

if __name__ == '__main__':
    print("Starting!")
    # ipdb.set_trace()
    env = PendulumEnv()
    save_expert_traj(env, 10)