import math
import time

import numpy as np
import torch
import torch.autograd
import qpth.qp_wrapper as mpc
import ipdb

## example task : hard pendulum with weird coordinates to make sure direct target tracking is difficult

class PendulumDynamics(torch.nn.Module):
        def forward(self, state, action):
            th = state[:, 0].view(-1, 1)
            thdot = state[:, 1].view(-1, 1)

            g = 10
            m = 1
            l = 1
            dt = 0.05

            u = action
            u = torch.clamp(u, -2, 2)

            newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
            newth = th + newthdot * dt
            newthdot = torch.clamp(newthdot, -8, 8)

            state = torch.cat((angle_normalize(newth), newthdot), dim=1)
            return state

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class FFDNetwork(torch.nn.Module):
    def __init__(self, args):
        self.args = args
        self.nu = args.nu
        self.nx = args.nx
        self.np = args.np
        self.T = args.T


        ## define the network layers : 
        self.fc1 = torch.nn.Linear(self.nx, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, self.np*self.T)
        self.net = torch.nn.Sequential(self.fc1, self.relu, self.fc2, self.relu, self.fc3)

    def forward(self, x):
        """
        compute the policy output for the given state x
        """
        x_ref = self.net(x)
        x_ref = x_ref.view(-1, self.T, self.np)
        return x_ref


class Tracking_MPC:
    def __init__(self, args):
        self.args = args
        self.nu = args.nu
        self.nx = args.nx
        self.dt = args.dt
        self.T = args.T

        self.u_upper = args.u_upper
        self.u_lower = args.u_lower
        self.max_iter = args.max_iter
        self.eps = args.eps
        self.warm_start = args.warm_start
        self.bsz = args.bsz
        self.device = args.device

        self.Q = args.Q
        self.R = args.R
        # self.Qf = args.Qf
        if args.Q is None:
            self.Q = torch.ones(self.nx, dtype=torch.float32, device=args.device)
            # self.Qf = torch.ones(self.nx, dtype=torch.float32, device=args.device)
            self.R = torch.ones(self.nu, dtype=torch.float32, device=args.device)
        self.Q = torch.cat([self.Q, self.R], dim=0)
        self.Q = torch.diag(self.Q).repeat(self.T, self.bsz, 1, 1)

        self.u_init = torch.zeros(self.T, self.bsz, self.nu, dtype=torch.float32, device=self.device)

        self.ctrl = mpc.MPC(self.nx, self.nu, self.T, u_lower=self.u_lower, u_upper=self.u_upper, qp_iter=self.max_iter,
                        exit_unconverged=False, eps=1e-2,
                        n_batch=self.bsz, backprop=False, verbose=0, u_init=self.u_init,
                        grad_method=mpc.GradMethods.AUTO_DIFF, solver_type='dense')

    def forward(self, x_init, x_ref):
        """
        compute the mpc output for the given state x and reference x_ref
        """

        self.compute_p(x_ref)
        cost = mpc.QuadCost(self.Q, self.p)
        self.ctrl.u_init = self.u_init
        state = x_init.unsqueeze(0).repeat(self.bsz, 1)
        nominal_states, nominal_actions = self.ctrl(state, cost, PendulumDynamics())
        return nominal_states, nominal_actions

    def compute_p(self, x_ref):
        """
        compute the p for the quadratic objective using self.Q as the diagonal matrix and the reference x_ref at each time without a for loop
        """
        self.p = torch.zeros(self.T, self.bsz, self.nx+self.nu, dtype=torch.float32, device=self.device)
        self.p[:, :, :self.nx] = -(self.Q[:, :, :self.nx, :self.nx] * x_ref.unsqueeze(-2)).sum(dim=-1)
        return self.p



class DiffMPCPolicy:
    def __init__(self, args):
        self.args = args
        self.nu = args.nu
        self.nx = args.nx
        self.np = args.np
        self.T = args.T
        self.dt = args.dt
        self.device = args.device
        self.model = FFDNetwork(args)
        self.model.to(self.device)
        self.tracking_mpc = Tracking_MPC(args)
    
    def forward(self, x):
        """
        compute the policy output for the given state x
        """
        x_ref = self.model(x)
        x_ref = x_ref.view(-1, self.np)
        nom_states, nom_actions = self.tracking_mpc(x, x_ref)
        return nominal_states, nominal_actions
    
# class DEQPolicy:

# class DEQMPCPolicy:
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nu', type=int, default=1)
    parser.add_argument('--nx', type=int, default=2)
    parser.add_argument('--np', type=int, default=2)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--u_upper', type=float, default=2)
    parser.add_argument('--u_lower', type=float, default=-2)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--eps', type=float, default=1e-2)
    parser.add_argument('--warm_start', type=bool, default=True)
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--Q', type=float, default=None)
    parser.add_argument('--R', type=float, default=None)
    args = parser.parse_args()

    gt_trajs = get_data(args)

    policy = DiffMPCPolicy(args)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-3)
    

    # run imitation learning using gt_trajs
    for i in range(100):
        # sample bsz random trajectories from gt_trajs and a random time step for each
        idxs = np.random.randint(0, len(gt_trajs), args.bsz)
        t_idxs = np.random.randint(0, len(gt_trajs[0]), args.bsz)
        x_init = gt_trajs[idxs, t_idxs]
        x_gt = gt_trajs[idxs, t_idxs:t_idxs+args.T]
        nominal_states, nominal_actions = policy.forward(x_init)
        loss = torch.norm(nominal_states - x_gt) + torch.norm(nominal_actions)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('loss: ', loss)
        print('nominal states: ', nominal_states)
        print('nominal actions: ', nominal_actions)





if __name__ == '__main__':
    main()