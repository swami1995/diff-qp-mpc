import math
import time

import numpy as np
import torch
import torch.autograd as autograd
import sys
# sys.path.insert(0, '/home/swaminathan/Workspace/qpth/')
sys.path.insert(0, '/home/sgurumur/locuslab/diff-qp-mpc/')
import qpth.qp_wrapper as mpc
import ipdb
import os
from envs_v1 import OneLinkCartpoleEnv
from ppo_train import PPO, GaussianPolicy
import pickle


class CartpoleExpert:
    def __init__(self, env, type="mpc"):
        """
        Initialize the MPC controller with the necessary parameters.

        Args:
            env: The PendulumEnv environment.
            type: The type of controller to use. Can be 'mpc' or 'ppo' or 'sac'.
        """

        self.type = type

        if self.type == "mpc":
            self.T = 100
            self.goal_state = torch.Tensor([0, np.pi, 0.0, 0])
            self.goal_weights = torch.Tensor([1.0, 10.0, 1, 1])
            self.ctrl_penalty = 0.0001
            self.mpc_eps = 1e-5
            self.linesearch_decay = 0.2
            self.max_linesearch_iter = 10
            self.nx = env.observation_space.shape[0]
            self.nu = env.action_space.shape[0]
            self.bsz = 1

            self.u_lower = torch.tensor(
                env.action_space.low, dtype=torch.float32
            ).double()
            self.u_upper = torch.tensor(
                env.action_space.high, dtype=torch.float32
            ).double()

            self.qp_iter = 20
            self.u_init = torch.zeros(self.T, self.bsz, self.nu).double()
            self.q = torch.cat(
                (self.goal_weights, self.ctrl_penalty * torch.ones(self.nu))
            )
            self.px = -torch.sqrt(self.goal_weights) * self.goal_state
            self.p = torch.cat((self.px, torch.zeros(self.nu)))
            self.Q = (
                torch.diag(self.q)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(self.T, self.bsz, 1, 1)
                .double()
            )
            self.p = self.p.unsqueeze(0).repeat(self.T, self.bsz, 1).double()

            self.ctrl = mpc.MPC(
                self.nx,
                self.nu,
                self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,  # .double(),
                qp_iter=self.qp_iter,
                exit_unconverged=False,
                eps=1e-5,
                n_batch=self.bsz,
                backprop=False,
                verbose=0,
                u_init=self.u_init,  # .double(),
                grad_method=mpc.GradMethods.AUTO_DIFF,
                solver_type="dense",
                add_goal_constraint=False,
                x_goal = torch.stack([env.goal]*self.bsz, dim=0),
            )
            self.cost = mpc.QuadCost(self.Q, self.p)

    def optimize_action(self, state):
        """Solve the MPC problem for the given state."""
        # ipdb.set_trace()
        F, f = self.ctrl.linearize_dynamics(torch.stack([env.goal]*self.bsz, dim=0).unsqueeze(0).repeat(self.T, 1, 1), torch.zeros_like(self.u_init), env.dynamics, diff=False)
        dx = mpc.LinDx(F, f)
        # ipdb.set_trace()
        nominal_states, nominal_actions = self.ctrl(
            state.double(), self.cost, dx, env.dynamics
        )
        # ipdb.set_trace()
        # u = torch.clamp(nominal_actions[0], self.u_lower, self.u_upper)
        return nominal_states, nominal_actions  # Return the first action in the optimal sequence

    def energy_shaping_action(self, state):
        """Compute the energy shaping action for the given state."""
        th = state[:, 0]
        # th = angle_normalize(th)
        thdot = state[:, 1]
        # ipdb.set_trace()
        E_tilde = 0.5 * thdot**2 - 10 * (1 + torch.cos(th + np.pi))
        u = -0.5 * thdot * E_tilde
        u = torch.clamp(u, self.u_lower, self.u_upper)
        return u.unsqueeze(1)


def get_1lcartpole_expert_traj_mpc(env, num_traj):
    """
    Get expert trajectories for cartpole environment using MPC for trajectory optimization.
    Args:
        env: The PendulumEnv environment.
        mpc_controller: The CartpoleExpert.
        num_traj: Number of trajectories to save.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples.
    """
    mpc_controller = CartpoleExpert(env)
    trajectories = []
    for _ in range(num_traj):
        state = env.reset()  # Reset environment to a new initial state
        traj = []
        done = False
        while not done:
            action, actions = mpc_controller.optimize_action(
                torch.tensor(state, dtype=torch.float32).view(1, -1)
            )
            # ipdb.set_trace()
            print(state, action)
            next_state, _, done, _ = env.step(action)
            traj.append((state, action.numpy()[0]))
            state = next_state*1
            # print(state, action)
            # if len(traj) > 100:
            #     ipdb.set_trace()
            print(len(traj))
            actions = torch.cat([actions[1:], actions[-1].unsqueeze(0)], dim=0)
            mpc_controller.ctrl.u_init = actions
        print(f"Trajectory length: {len(traj)}")
        trajectories.append(traj)
    return trajectories


def get_1lcartpole_expert_traj_ppo(env, num_traj):
    """
    Get expert trajectories for cartpole environment using the saved PPO checkpoint."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lr = 0.0003
    gamma = 0.99
    K_epochs = 80
    eps_clip = 0.2
    has_continuous_action_space = True
    action_std = 0.45
    ppo_policy = PPO(
        state_dim,
        action_dim,
        lr,
        lr,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std,
    )
    ppo_policy.load("PPO_preTrained/Pendulum-v0/PPO_Pendulum-v0_0_0.pth")
    trajectories = []
    reward_trajs = []
    for _ in range(num_traj):
        state = env.reset()
        traj = []
        done = False
        reward_traj = 0
        while not done:
            action = (
                ppo_policy.policy.actor(
                    torch.tensor(state, dtype=torch.float32).to(
                        ppo_policy.policy.action_var.device
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
            next_state, reward, done, _ = env.step(action)
            traj.append((state, action))
            state = next_state
            reward_traj += reward
        print(f"Trajectory length: {len(traj)}, reward: {reward_traj}")
        trajectories.append(traj)
        reward_trajs.append(reward_traj)
    print(
        f"Average reward: {np.mean(reward_trajs)}, Avg traj length: {np.mean([len(traj) for traj in trajectories])}"
    )
    ipdb.set_trace()
    return trajectories


def get_1lcartpole_expert_traj_sac(env, num_traj):
    """
    Get expert trajectories for cartpole environment using the saved PPO checkpoint."""
    device = torch.device("cuda" if False else "cpu")
    policy = GaussianPolicy(
        env.observation_space.shape[0], env.action_space.shape[0], 256, env.action_space
    ).to(device)
    # checkpoint = torch.load("/home/sgurumur/locuslab/pytorch-soft-actor-critic/checkpoints/sac_checkpoint_Pendulum-v0_bestT200")
    checkpoint = torch.load("ckpts/sac/sac_checkpoint_Pendulum-v0_bestT200")
    policy.load_state_dict(checkpoint["policy_state_dict"])
    trajectories = []
    reward_trajs = []
    for _ in range(num_traj):
        state = env.reset()
        traj = []
        done = False
        reward_traj = 0
        while not done:
            action = (
                policy.sample(
                    torch.tensor(state, dtype=torch.float32).to(device)[None]
                )[2]
                .detach()
                .cpu()
                .numpy()[0]
            )
            next_state, reward, done, _ = env.step(action)
            traj.append((state, action))
            state = next_state
            reward_traj += reward
        print(f"Trajectory length: {len(traj)}, reward: {reward_traj}")
        trajectories.append(traj)
        reward_trajs.append(reward_traj)
    print(
        f"Average reward: {np.mean(reward_trajs)}, Avg traj length: {np.mean([len(traj) for traj in trajectories])}"
    )
    # ipdb.set_trace()
    return trajectories


def save_expert_traj(env, num_traj, type="mpc"):
    """Save expert trajectories to a file.

    Args:
        env: gym environment
        num_traj: number of trajectories to save
        type: type of expert to use
    """

    ## use env name to choose which function to use to get expert trajectories
    if env.spec_id == "OneLinkCartpole-v0" or env.spec_id == "OneLinkCartpole-v0-stabilize":
        if type == "mpc":
            expert_traj = get_1lcartpole_expert_traj_mpc(env, num_traj)
        elif type == "ppo":
            expert_traj = get_1lcartpole_expert_traj_ppo(env, num_traj)
        elif type == "sac":
            expert_traj = get_1lcartpole_expert_traj_sac(env, num_traj)
    else:
        raise NotImplementedError            

    ## save expert trajectories to a file in data folder
    if os.path.exists("data") == False:
        os.makedirs("data")

    with open(f"data/expert_traj_{type}-{env.spec_id}_new.pkl", "wb") as f:
        pickle.dump(expert_traj, f)


def get_gt_data(args, env, type="mpc"):
    """
    Get ground truth data for imitation learning.
    Args:
        args: The arguments for the training script.
        env: The environment.
        type: The type of controller to use. Can be 'mpc' or 'ppo' or 'sac'.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples.
    """
    with open('data/expert_traj_mpc-OneLinkCartpole-v0-stabilize_new.pkl', 'rb') as f:#f'data/expert_traj_{type}-{env.spec_id}.pkl', 'rb') as f:
    # with open(f"data/expert_traj_{type}-{env.spec_id}_new.pkl", "rb") as f:
        gt_trajs = pickle.load(f)
    # ipdb.set_trace()
    return gt_trajs


def merge_gt_data(gt_trajs):
    """
    Merge ground truth data for imitation learning.
    Args:
        gt_trajs: A list of trajectories, each trajectory is a list of (state, action) tuples.
    Returns:
        A list of (state, action) tuples.
    """
    merged_gt_traj = {"state": [], "action": [], "mask": []}
    for traj in gt_trajs:
        for state, action in traj:
            merged_gt_traj["state"].append(state)
            merged_gt_traj["action"].append(action)
            merged_gt_traj["mask"].append(1)
        merged_gt_traj["mask"][-1] = 0  # mask = 0 at the end of each trajectory
    # ipdb.set_trace()
    merged_gt_traj["state"] = torch.tensor(
        np.array(merged_gt_traj["state"]), dtype=torch.float32
    )
    merged_gt_traj["action"] = torch.tensor(
        np.array(merged_gt_traj["action"]), dtype=torch.float32
    )
    merged_gt_traj["mask"] = torch.tensor(
        np.array(merged_gt_traj["mask"]), dtype=torch.float32
    )
    return merged_gt_traj


def sample_trajectory(gt_trajs, bsz, T):
    """
    Sample a batch of trajectories from the ground truth data.
    Args:
        gt_trajs: A dictionary of "state", "action" and "mask" tensors with concatenated trajectories.
        bsz: Batch size.
        T: Time horizon.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples with length T.
    """
    idxs = np.random.randint(0, len(gt_trajs["state"]), bsz)
    trajs = {"state": [], "action": [], "mask": []}
    for i in range(bsz):
        if idxs[i] + T < len(gt_trajs["state"]):
            trajs["state"].append(gt_trajs["state"][idxs[i] : idxs[i] + T])
            trajs["action"].append(gt_trajs["action"][idxs[i] : idxs[i] + T])
            trajs["mask"].append(gt_trajs["mask"][idxs[i] : idxs[i] + T])
        else:
            padding = idxs[i] + T - len(gt_trajs["state"])
            trajs["state"].append(
                torch.cat(
                    [gt_trajs["state"][idxs[i] :], gt_trajs["state"][:padding] * 0.0],
                    dim=0,
                )
            )
            trajs["action"].append(
                torch.cat(
                    [gt_trajs["action"][idxs[i] :], gt_trajs["action"][:padding] * 0.0],
                    dim=0,
                )
            )
            trajs["mask"].append(
                torch.cat(
                    [gt_trajs["mask"][idxs[i] :], gt_trajs["mask"][:padding] * 0], dim=0
                )
            )
    trajs["state"] = torch.stack(trajs["state"])
    trajs["action"] = torch.stack(trajs["action"])
    trajs["mask"] = torch.stack(trajs["mask"])
    for i in reversed(range(T)):
        trajs["mask"][:, i] = torch.prod(trajs["mask"][:, :i], dim=1)
    return trajs


def test_qp_mpc(env):
    mpc_controller = CartpoleExpert(env)
    trajectories = []
    state = env.reset()  # Reset environment to a new initial state
    # state = np.array([0.0, np.pi+0.1, 0.0, 0.0])
    traj = []
    done = False
    states, actions = mpc_controller.optimize_action(
        torch.tensor(state, dtype=torch.float32).view(1, -1)
    )
    # ipdb.set_trace()
    states = (states-mpc_controller.goal_state).squeeze().detach().numpy()
    actions = actions.detach().numpy()
    traj = []
    trajectories = []
    ipdb.set_trace()
    # for i in range(len(states)):
    #     traj.append((states[i], actions[i][0]))
    # trajectories.append(traj)
    # with open(f"data/expert_traj_mpc-{env.spec_id}.pkl", "wb") as f:
    #     pickle.dump(trajectories, f)

if __name__ == "__main__":
    print("Starting!")
    # ipdb.set_trace()
    env = OneLinkCartpoleEnv(stabilization=True)
    # env = IntegratorEnv()
    # save_expert_traj(env, 1, "mpc")
    test_qp_mpc(env)
