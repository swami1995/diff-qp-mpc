import math
import time

import numpy as np
import torch
import torch.autograd as autograd
import sys
sys.path.insert(0, '/home/swaminathan/Workspace/qpth/')
import qpth.qp_wrapper as mpc
import ipdb
from envs import PendulumEnv, PendulumDynamics
from datagen import get_gt_data, merge_gt_data, sample_trajectory
from policies import NNMPCPolicy, DEQPolicy, DEQMPCPolicy

## example task : hard pendulum with weird coordinates to make sure direct target tracking is difficult

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=1)
    parser.add_argument("--T", type=int, default=10)
    # parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument("--qp_iter", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--warm_start", type=bool, default=True)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deq", action='store_true')
    parser.add_argument("--hdim", type=int, default=256)
    parser.add_argument("--deq_iter", type=int, default=6)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PendulumEnv(stabilization=False)
    gt_trajs = get_gt_data(args, env, "mpc")
    gt_trajs = merge_gt_data(gt_trajs)
    args.Q = torch.Tensor([10.0, 1]).to(args.device)
    args.R = torch.Tensor([1.0]).to(args.device)
    if args.deq:
        policy = DEQMPCPolicy(args, env).to(args.device)
    else:
        policy = NNMPCPolicy(args, env).to(args.device)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-4)
    losses = []
    losses_end = []

    # run imitation learning using gt_trajs
    for i in range(10000):
        # sample bsz random trajectories from gt_trajs and a random time step for each
        traj_sample = sample_trajectory(gt_trajs, args.bsz, args.T)
        traj_sample = {k: v.to(args.device) for k, v in traj_sample.items()}
        # ipdb.set_trace()
        # traj_sample["state"] = torch.cat(
        #     [
        #         traj_sample["state"],
        #         PendulumDynamics()(
        #             traj_sample["state"][:, -1], traj_sample["action"][:, -1]
        #         )[:, None],
        #     ],
        #     dim=1,
        # )
        # idxs = np.random.randint(0, len(gt_trajs), args.bsz)
        # t_idxs = np.random.randint(0, len(gt_trajs[0]), args.bsz)
        # x_init = gt_trajs[idxs, t_idxs]
        # x_gt = gt_trajs[idxs, t_idxs:t_idxs+args.T]

        # ipdb.set_trace()
        if args.deq:
            loss = 0.0
            trajs = policy(traj_sample["state"][:, 0])
            # ipdb.set_trace()
            for j, (nominal_states, nominal_actions) in enumerate(trajs):
                loss_j = torch.abs(
                            (nominal_states.transpose(0, 1) - traj_sample["state"])#[:, 1:])
                            * traj_sample["mask"][:, :, None]
                        ).sum(dim=-1).mean()
                loss += loss_j
            loss_end = torch.abs(
                        (nominal_states.transpose(0, 1) - traj_sample["state"])#[:, 1:])
                        * traj_sample["mask"][:, :, None]
                    ).sum(dim=-1).mean()
        else:
            nominal_states, nominal_actions = policy(traj_sample["state"][:, 0])
            loss = (
                torch.abs(
                    (nominal_states.transpose(0, 1) - traj_sample["state"])#[:, 1:])
                    * traj_sample["mask"][:, :, None]
                ).sum(dim=-1).mean()
            )  # + torch.norm(nominal_actions)
        # ipdb.set_trace()
        # loss += (nominal_actions[0] - traj_sample["action"][:, 0]).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        losses_end.append(loss_end.item())
        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 4)
        optimizer.step()
        if i % 10 == 0:
            print("iter: ", i)
            print(
                "grad norm: ",
                torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1000),
            )
            print("loss: ", np.mean(losses)/args.deq_iter, "loss_end: ", np.mean(losses_end))
            losses = []
            # print('nominal states: ', nominal_states)
            # print('nominal actions: ', nominal_actions)


if __name__ == "__main__":
    main()
