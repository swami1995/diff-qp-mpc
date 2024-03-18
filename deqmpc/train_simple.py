import math
import time

import numpy as np
import torch
import torch.autograd as autograd
import sys, os, time
sys.path.insert(0, '/home/sgurumur/locuslab/diff-qp-mpc/')
import qpth.qp_wrapper as mpc
import ipdb
from envs import PendulumEnv, PendulumDynamics, IntegratorEnv, IntegratorDynamics
from datagen import get_gt_data, merge_gt_data, sample_trajectory
from policies import NNMPCPolicy, DEQPolicy, DEQMPCPolicy, NNPolicy

# import tensorboard from pytorch
from torch.utils.tensorboard import SummaryWriter

## example task : hard pendulum with weird coordinates to make sure direct target tracking is difficult


def seeding(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=1)
    parser.add_argument("--T", type=int, default=5)
    # parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument("--qp_iter", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warm_start", type=bool, default=True)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--deq", action="store_true")
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--deq_iter", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--layer_type", type=str, default='mlp')
    parser.add_argument("--kernel_width", type=int, default=3)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--lastqp_solve", action="store_true")
    parser.add_argument("--qp_solve", action="store_true")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--solver_type", type=str, default='al')

    args = parser.parse_args()
    seeding(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device if args.device is None else args.device
    if args.save:
        if not os.path.exists("./logs/" + args.name):
            os.makedirs("./logs/" + args.name)
        args.name = args.name + f"_T{args.T}_bsz{args.bsz}_deq_iter{args.deq_iter}_np{args.np}"
        writer = SummaryWriter("./logs/" + args.name)

    env = PendulumEnv(stabilization=False)
    # env = IntegratorEnv()

    # gt_trajs = get_gt_data(args, env, "mpc")
    gt_trajs = get_gt_data(args, env, "sac")
    gt_trajs = merge_gt_data(gt_trajs)
    args.Q = torch.Tensor([10.0, 1.00]).to(args.device)
    args.R = torch.Tensor([0.001]).to(args.device)
    if args.deq:
        policy = DEQMPCPolicy(args, env).to(args.device)
        # save arguments
        if args.save:
            torch.save(args, "./logs/" + args.name + "/args")
    else:
        # policy = NNMPCPolicy(args, env).to(args.device)
        policy = NNPolicy(args, env).to(args.device)
        # save arguments
        torch.save(args, "./model/bc_sac_pen_args")
        
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=args.lr)
    losses = []
    losses_end = []
    time_diffs = []

    # run imitation learning using gt_trajs
    for i in range(5000):
        # sample bsz random trajectories from gt_trajs and a random time step for each
        traj_sample = sample_trajectory(gt_trajs, args.bsz, args.T)
        traj_sample = {k: v.to(args.device) for k, v in traj_sample.items()}

        traj_sample["state"] = unnormalize_states(traj_sample["state"])
        iter_qp_solve = True#False if (i < 1000 or not args.pretrain) else True
        qp_solve = iter_qp_solve and args.qp_solve # warm start only after 1000 iterations
        lastqp_solve = args.lastqp_solve and iter_qp_solve
        if args.deq:
            loss = 0.0
            start = time.time()
            trajs = policy(traj_sample["state"][:, 0], traj_sample["state"], iter=i, qp_solve=qp_solve, lastqp_solve=lastqp_solve)
            end = time.time()
            for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
                loss_j = (
                    torch.abs(
                        (
                            nominal_states.transpose(0, 1) - traj_sample["state"]
                        )
                        * traj_sample["mask"][:, :, None]
                    )
                    .sum(dim=-1)
                    .mean()
                )
                loss += loss_j
            loss_end = (
                torch.abs(
                    (nominal_states.transpose(0, 1) - traj_sample["state"])
                    * traj_sample["mask"][:, :, None]
                )
                .sum(dim=-1)
                .mean()
            )
        else:
            loss = 0.0
            nominal_states, nominal_actions = policy(traj_sample["state"][:, 0])
            if policy.output_type == 0 or policy.output_type == 2:
                loss += (
                    torch.abs(
                        (nominal_actions - traj_sample["action"])
                        * traj_sample["mask"][:, :, None]
                    )
                    .sum(dim=-1)
                    .mean()
                )
            if policy.output_type == 1 or policy.output_type == 2 or policy.output_type == 3:
                loss += (
                    torch.abs(
                        (nominal_states - traj_sample["state"])#[:, :, : policy.np]
                        * traj_sample["mask"][:, :, None]
                    )
                    .sum(dim=-1)
                    .mean()
                )
            loss_end = torch.Tensor([0.0])
        time_diffs.append(end-start)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        losses_end.append(loss_end.item())
        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 4)
        optimizer.step()
        if i % 100 == 0:
            print("iter: ", i)
            print(
                "grad norm: ",
                torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1000),
            )
            print(
                "loss: ",
                np.mean(losses) / args.deq_iter,
                "loss_end: ",
                np.mean(losses_end),
                "avg time: ",
                np.mean(time_diffs)
            )
            if args.save:
                torch.save(policy.state_dict(), "./model/" + args.name)
                writer.add_scalar("losses/loss_avg", np.mean(losses) / args.deq_iter, i)
                writer.add_scalar("losses/loss_end", np.mean(losses_end), i)

            
            losses = []
            losses_end = []
            time_diffs = []
            # print('nominal states: ', nominal_states)
            # print('nominal actions: ', nominal_actions)   

    # torch.save(policy.state_dict(), "./model/bc_sac_pen")

def unnormalize_states(nominal_states):
    # ipdb.set_trace()
    # check theta of the first state in nominal_states[:, 0][0] and make sure all the nominal_states are in the same phase (i.e in terms of angle normalization)
    angle_0 = nominal_states[:, 0, 0]
    prev_angle = angle_0
    # ipdb.set_trace()
    for i in range(nominal_states.shape[1]):
        mask = torch.abs(nominal_states[:, i, 0] - prev_angle) > np.pi / 2
        mask_sign = torch.sign(nominal_states[:, i, 0])
        if mask.any():
            nominal_states[mask, i, 0] = (
                nominal_states[mask, i, 0] - mask_sign[mask] * 2 * np.pi
            )
        prev_angle = nominal_states[:, i, 0]
    return nominal_states


if __name__ == "__main__":
    main()
