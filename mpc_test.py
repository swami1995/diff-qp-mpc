import math
import time

import numpy as np
import torch
import torch.autograd
import qpth.qp_wrapper as mpc
import ipdb

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 4  # T
    N_BATCH = 1
    LQR_ITER = 5
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0


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

    class DoubleIntegrator(torch.nn.Module):
        def forward(self, state, action):
            q = state[:, 0].view(-1, 1)
            qdot = state[:, 1].view(-1, 1)

            m = 1
            dt = 0.05

            u = action
            # u = torch.clamp(u, -2, 2)

            newqdot = qdot + u * dt
            newq = q + newqdot * dt
            # newthdot = torch.clamp(newthdot, -8, 8)

            state = torch.cat((newq, newqdot), dim=1)
            return state


    state = [np.pi, 0]

    nx = 2
    nu = 1

    u_init = None
    goal_weights = torch.tensor((1., 0.1))  # nx
    goal_state = torch.tensor((0., 0.))  # nx
    ctrl_penalty = 0.001
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(nu)
    ))  # nx + nu
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(nu)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

    # run MPC
    # total_reward = 0
    # for i in range(run_iter):
        # state = env.state.copy()
    state = torch.tensor(state).view(1, -1).requires_grad_(True)
    # recreate controller using updated u_init (kind of wasteful right?)
    ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW, u_upper=ACTION_HIGH, qp_iter=LQR_ITER,
                    exit_unconverged=False, eps=1e-2,
                    n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                    grad_method=mpc.GradMethods.AUTO_DIFF, solver_type='dense')

    # compute action based on current state, dynamics, and cost
    nominal_states, nominal_actions = ctrl(state, cost, DoubleIntegrator())# PendulumDynamics())
    action = nominal_actions[0]  # take first planned action
    u_init = torch.cat((nominal_actions[1:].cpu(), torch.zeros(1, N_BATCH, nu)), dim=0)
    rollout_states = ctrl.rollout(state, nominal_actions.cpu(), DoubleIntegrator())# PendulumDynamics())
    print(torch.autograd.grad(nominal_states.norm(), state, retain_graph=True))
    # ipdb.set_trace()