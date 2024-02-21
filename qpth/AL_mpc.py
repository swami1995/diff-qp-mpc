import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr

from collections import namedtuple

from enum import Enum

import sys

from . import util
# from .pnqp import pnqp
# from .lqr_step import LQRStep
# from .dynamics import CtrlPassthroughDynamics
from . import qp
import ipdb

QuadCost = namedtuple('QuadCost', 'C c')
LinDx = namedtuple('LinDx', 'F f')

# https://stackoverflow.com/questions/11351032
QuadCost.__new__.__defaults__ = (None,) * len(QuadCost._fields)
LinDx.__new__.__defaults__ = (None,) * len(LinDx._fields)


class GradMethods(Enum):
    AUTO_DIFF = 1
    FINITE_DIFF = 2
    ANALYTIC = 3
    ANALYTIC_CHECK = 4


class SlewRateCost(Module):
    """Hacky way of adding the slew rate penalty to costs."""
    # TODO: It would be cleaner to update this to just use the slew
    # rate penalty instead of # slew_C
    def __init__(self, cost, slew_C, n_state, n_ctrl):
        super().__init__()
        self.cost = cost
        self.slew_C = slew_C
        self.n_state = n_state
        self.n_ctrl = n_ctrl

    def forward(self, tau):
        true_tau = tau[:, self.n_ctrl:]
        true_cost = self.cost(true_tau)
        # The slew constraints are time-invariant.
        slew_cost = 0.5 * util.bquad(tau, self.slew_C[0])
        return true_cost + slew_cost

    def grad_input(self, x, u):
        raise NotImplementedError("Implement grad_input")


class MPC(Module):
    """A differentiable box-constrained iLQR solver.

    This provides a differentiable solver for the following box-constrained
    control problem with a quadratic cost (defined by C and c) and
    non-linear dynamics (defined by f):

        min_{tau={x,u}} sum_t 0.5 tau_t^T C_t tau_t + c_t^T tau_t
                        s.t. x_{t+1} = f(x_t, u_t)
                            x_0 = x_init
                            u_lower <= u <= u_upper

    This implements the Control-Limited Differential Dynamic Programming
    paper with a first-order approximation to the non-linear dynamics:
    https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

    Some of the notation here is from Sergey Levine's notes:
    http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_8_model_based_planning.pdf

    Required Args:
        n_state, n_ctrl, T

    Optional Args:
        u_lower, u_upper: The lower- and upper-bounds on the controls.
            These can either be floats or shaped as [T, n_batch, n_ctrl]
        u_init: The initial control sequence, useful for warm-starting:
            [T, n_batch, n_ctrl]
        qp_iter: The number of QP iterations to perform.
        grad_method: The method to compute the Jacobian of the dynamics.
            GradMethods.ANALYTIC: Use a manually-defined Jacobian.
                + Fast and accurate, use this if possible
            GradMethods.AUTO_DIFF: Use PyTorch's autograd.
                + Slow
            GradMethods.FINITE_DIFF: Use naive finite differences
                + Inaccurate
        delta_u (float): The amount each component of the controls
            is allowed to change in each LQR iteration.
        verbose (int):
            -1: No output or warnings
             0: Warnings
            1+: Detailed iteration info
        eps: Termination threshold, on the norm of the full control
             step (without line search)
        back_eps: `eps` value to use in the backwards pass.
        n_batch: May be necessary for now if it can't be inferred.
                 TODO: Infer, potentially remove this.
        linesearch_decay (float): Multiplicative decay factor for the
            line search.
        max_linesearch_iter (int): Can be used to disable the line search
            if 1 is used for some problems the line search can
            be harmful.
        exit_unconverged: Assert False if a fixed point is not reached.
        detach_unconverged: Detach examples from the graph that do
            not hit a fixed point so they are not differentiated through.
        backprop: Allow the solver to be differentiated through.
        slew_rate_penalty (float): Penalty term applied to
            ||u_t - u_{t+1}||_2^2 in the objective.
        prev_ctrl: The previous nominal control sequence to initialize
            the solver with.
        not_improved_lim: The number of iterations to allow that don't
            improve the objective before returning early.
        best_cost_eps: Absolute threshold for the best cost
            to be updated.
    """

    def __init__(
            self, n_state, n_ctrl, T,
            u_lower=None, u_upper=None,
            u_zero_I=None,
            u_init=None,
            x_init=None,
            qp_iter=10,
            grad_method=GradMethods.ANALYTIC,
            delta_u=None,
            verbose=0,
            eps=1e-7,
            back_eps=1e-7,
            n_batch=None,
            linesearch_decay=0.2,
            max_linesearch_iter=10,
            exit_unconverged=True,
            detach_unconverged=True,
            backprop=True,
            slew_rate_penalty=None,
            prev_ctrl=None,
            not_improved_lim=5,
            best_cost_eps=1e-4,
            solver_type='dense',
            single_qp_solve=False,
            add_goal_constraint=False,
            x_goal=None,
            diag_cost=False, 
            ineqG=None,
            ineqh=None,
    ):
        super().__init__()

        assert (u_lower is None) == (u_upper is None)
        assert max_linesearch_iter > 0

        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.u_lower = u_lower
        self.u_upper = u_upper
        self.x_upper = None
        self.x_lower = None
        self.x_goal = x_goal
        self.ineqG = ineqG
        self.ineqh = ineqh        

        if not isinstance(u_lower, float):
            self.u_lower = util.detach_maybe(self.u_lower)

        if not isinstance(u_upper, float):
            self.u_upper = util.detach_maybe(self.u_upper)

        self.u_zero_I = util.detach_maybe(u_zero_I)
        self.u_init = util.detach_maybe(u_init)
        self.x_init = util.detach_maybe(x_init)
        self.qp_iter = qp_iter
        self.grad_method = grad_method
        self.delta_u = delta_u
        self.verbose = verbose
        self.eps = eps
        self.back_eps = back_eps
        self.n_batch = n_batch
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.exit_unconverged = exit_unconverged
        self.detach_unconverged = detach_unconverged
        self.backprop = backprop
        self.not_improved_lim = not_improved_lim
        self.best_cost_eps = best_cost_eps

        self.slew_rate_penalty = slew_rate_penalty
        self.prev_ctrl = prev_ctrl
        self.solver_type = solver_type
        self.single_qp_solve = single_qp_solve
        self.add_goal_constraint = add_goal_constraint
        self.diag_cost = True# diag_cost
        self.al_iter = 20
        self.neq = n_state*(T-1) + n_state
        self.nineq = 0
        self.dyn_res_crit = 1e-6
        self.dyn_res_factor = 10
        self.rho_prev = 1.0
        if self.add_goal_constraint:
            self.neq += n_state
        if ineqG is not None:
            self.nineq = ineqG.size(1)
        if u_lower is not None:
            self.nineq += n_ctrl*T*2
        if self.x_lower is not None:
            self.nineq += n_state*T*2
        # return self.Qi, self.Gi, self.Ai

    def forward(self, x0, cost, dx, dx_true=None):
        # QuadCost.C: [T, n_batch, n_tau, n_tau]
        # QuadCost.c: [T, n_batch, n_tau]
        if dx_true is None:
            self.dx_true = dx
        else:
            self.dx_true = dx_true
        assert isinstance(cost, QuadCost) or \
            isinstance(cost, Module) or isinstance(cost, Function)
        assert isinstance(dx, LinDx) or \
            isinstance(dx, Module) or isinstance(dx, Function)

        # TODO: Clean up inferences, expansions, and assumptions made here.
        if self.n_batch is not None:
            n_batch = self.n_batch
        elif isinstance(cost, QuadCost) and cost.C.ndimension() == 4:
            n_batch = cost.C.size(1)
        else:
            print('MPC Error: Could not infer batch size, pass in as n_batch')
            sys.exit(-1)


        # if c.ndimension() == 2:
        #     c = c.unsqueeze(1).expand(self.T, n_batch, -1)

        if isinstance(cost, QuadCost):
            C, c = cost
            if C.ndimension() == 2:
                # Add the time and batch dimensions.
                C = C.unsqueeze(0).unsqueeze(0).expand(
                    self.T, n_batch, self.n_state+self.n_ctrl, -1)
            elif C.ndimension() == 3:
                # Add the batch dimension.
                C = C.unsqueeze(1).expand(
                    self.T, n_batch, self.n_state+self.n_ctrl, -1)

            if c.ndimension() == 1:
                # Add the time and batch dimensions.
                c = c.unsqueeze(0).unsqueeze(0).expand(self.T, n_batch, -1)
            elif c.ndimension() == 2:
                # Add the batch dimension.
                c = c.unsqueeze(1).expand(self.T, n_batch, -1)

            if C.ndimension() != 4 or c.ndimension() != 3:
                print('MPC Error: Unexpected QuadCost shape.')
                sys.exit(-1)
            cost = QuadCost(C, c)

        # ipdb.set_trace()
        assert x0.ndimension() == 2 and x0.size(0) == n_batch

        if self.u_init is None:
            u = torch.zeros(self.T, n_batch, self.n_ctrl).type_as(x0.data)
        else:
            u = self.u_init
            if u.ndimension() == 2:
                u = u.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        u = u.type_as(x0.data)

        if self.x_init is None:
            # x = torch.zeros(self.T, n_batch, self.n_state).type_as(x0.data)
            x = self.rollout(x0, u, dx)#[:-1]
            # ipdb.set_trace()
        else:
            x = self.x_init
            if x.ndimension() == 2:
                x = x.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        x = x.type_as(x0.data)

        if self.verbose > 0:
            print('Initial mean(cost): {:.4e}'.format(
                torch.mean(util.get_cost(
                    self.T, u, cost, dx, x_init=self.x_init
                )).item()
            ))

        best = None
        # ipdb.set_trace()
        if self.single_qp_solve:
            if self.diag_cost:
                cost = QuadCost(cost.C.diagonal(dim1=-2, dim2=-1), cost.c)
            x, u, cost_total = self.single_qp_ls(x, u, dx, x0, cost)
        else:
            x, u, cost_total = self.solve_nonlin(x, u, dx, x0, cost)
        
        return (x, u)
    
    def single_qp(self, x, u, dx, x0, cost, lamda_init=None, rho_init=None):
        x = x.transpose(0,1)
        u = u.transpose(0,1)
        if lamda_init is None:
            lamda = torch.zeros(self.n_batch, self.neq+self.nineq).type_as(x)#.reshape(self.n_batch, self.T, -1) 
        else:
            lamda = lamda_init
        if rho_init is None:
            rho = self.rho_prev
        else:
            rho = rho_init

        x_old, u_old = x, u
        dyn_res_clamp_start = self.dyn_res(torch.cat((x, u), dim=2), dx, x0, res_type='clamp')
        dyn_res_clamp_start = dyn_res_clamp_start.norm().item()
        
        # Augmented Lagrangian updates with broyden for root finding of the residual
        for i in range(self.al_iter):
            xu = torch.cat((x, u), dim=2)
            # ipdb.set_trace()
            y = lamda + rho*self.dyn_res(xu, dx, x0, res_type='clamp')
            input = torch.cat((x, u), dim=2)
            Hinv = 1/cost.C.transpose(0,1) # Might need to adaptively regularize this based on rho and solver progress/step size. 
            fn = lambda x,y,y_update=False : self.grad_res(x, y, cost, dx, x0, lamda, rho, y_update)
            dyn_fn = lambda x : self.dyn_res(x, dx, x0)
            cost_fn = lambda x : self.compute_cost(x, cost)
            merit_fn = lambda x, grad=False : self.merit_function(x, dx, x0, cost, lamda, rho, grad)
            # out = util.broyden_AL(fn, merit_fn, dyn_fn, cost_fn, xu, y, threshold=10, eps=1e-6, rho=rho, Hinv=Hinv, ysize=y.shape[-1], ls=True, idx=True)
            # out = util.LBFGS_AL(fn, merit_fn, dyn_fn, cost_fn, xu, y, threshold=80, eps=1e-6, rho=rho, Hinv=Hinv, ysize=y.shape[-1], ls=True, idx=True)
            out = util.Newton_AL(fn, merit_fn, dyn_fn, cost_fn, xu, y, threshold=1e-3, eps=1e-6, rho=rho, Hinv=Hinv, ysize=y.shape[-1], ls=True, idx=True)
            # out = util.GD_AL(fn, input, threshold=10, eps=1e-6, rho=rho, Hinv=Hinv, ysize=y.shape[-1], ls=True, dyn_fn = dyn_fn, cost_fn = cost_fn)
            # x_res, lam_res = self.grad_res(x, lamda, cost, dx, x0, lamda, rho)    

            x_new, u_new = out['result'][:,:,:self.n_state], out['result'][:,:,self.n_state:]
            delta_x = x_new - x
            delta_u = u_new - u
            delta_xu = torch.cat((delta_x, delta_u), dim=2)
            # if delta_xu.norm() < self.eps:
            #     print("converged, iter : ", i, " norm : ", delta_xu.norm())
            #     break
            # else:
            #     print("not converged, iter : ", i, " norm : ", delta_xu.norm())
            # ipdb.set_trace()
            dyn_res, dyn_res_clamp = self.dyn_res(torch.cat((x_new, u_new), dim=2), dx, x0, res_type='both')
            lamda = lamda + rho*dyn_res
            lamda[:, self.neq:] = torch.clamp(lamda[:, self.neq:], min=0)
            rho = rho*10
            x, u = x_new, u_new
            dyn_res_clamp = torch.norm(dyn_res_clamp).item()
            if  dyn_res_clamp < self.dyn_res_crit or dyn_res_clamp < dyn_res_clamp_start/self.dyn_res_factor:
                break
            ipdb.set_trace()

        delta_x = (x - x_old).transpose(0,1)
        delta_u = (u - u_old).transpose(0,1)
        return delta_x, delta_u, out['result']
    
    def merit_function(self, xu, dx, x0, cost, lamda, rho, grad=False):
        bsz = xu.size(0)
        cost_total = self.compute_cost(xu, cost)
        res, res_clamp = self.dyn_res(xu, dx, x0, res_type='both')
        # if grad:
        return cost_total + (lamda*res).view(bsz, -1).sum(dim=1) + 0.5*rho*(res_clamp*res_clamp).view(bsz, -1).sum(dim=1)
        # else:
        #     return cost_total.mean() + (lamda*res_clamp).view(bsz, -1).sum(dim=1).mean() + 0.5*rho*(res_clamp*res_clamp).view(bsz, -1).sum(dim=1).mean()

    def dyn_res_eq(self, x, u, dx, x0):
        " split x into state and control and compute dynamics residual using dx"
        # ipdb.set_trace()
        if isinstance(dx, LinDx):
            x_next = (dx.F.permute(1,0,2,3)*torch.cat((x, u), dim=2)[:,:-1,None,:]).sum(dim=-1) + dx.f.permute(1,0,2)
            # x_next = x_next[:,:-1]
        else:
            x_next = dx(x, u)[:,:-1]
            
        res = (x_next - x[:,1:,:])#.reshape(self.n_batch, -1)
        res_init = (x[:,0,:] - x0).reshape(self.n_batch, 1, -1)
        if self.add_goal_constraint:
            res_goal = (x[:,-1,:] - self.x_goal).reshape(self.n_batch, -1)
            res = torch.cat((res, res_init, res_goal), dim=1)
        else:
            res = torch.cat((res, res_init), dim=1)
        res = res.reshape(self.n_batch, -1)
        return res
    
    def dyn_res_ineq(self, x, u, dx, x0):
        # Add state constraints if self.x_lower and self.x_upper are not None
        res = None
        if self.x_lower is not None:
            res = torch.cat((
                x - self.x_upper, 
                -x + self.x_lower
                ), dim=2)

        # Add control constraints if self.u_lower and self.u_upper are not None
        if self.u_lower is not None:
            res_u = torch.cat((
                u - self.u_upper,
                -u + self.u_lower
                ), dim=2)
            if res is None:
                res = res_u
            else:
                res = torch.cat((res,res_u), dim=2)
        
        # Add other inequality constraints if self.ineqG and self.ineqh are not None
        if self.ineqG is not None:
            res_G = torch.bmm(self.ineqG, x.unsqueeze(-1)) - self.ineqh
            if res is None:
                res = res_G
            else:
                res = torch.cat((res,res_G), dim=2)
        res = res.reshape(self.n_batch, -1)
        res_clamp = torch.clamp(res, min=0)
        return res, res_clamp
    
    def dyn_res(self, x, dx, x0, res_type='clamp'):
        x = x.reshape(self.n_batch, self.T, self.n_state+self.n_ctrl)
        x, u = x[:,:,:self.n_state], x[:,:,self.n_state:]
        # Equality residuals
        res_eq = self.dyn_res_eq(x, u, dx, x0)
        # Inequality residuals
        res_ineq, res_ineq_clamp = self.dyn_res_ineq(x, u, dx, x0)
        if res_type == 'noclamp':
            return torch.cat((res_eq, res_ineq), dim=1)
        elif res_type == 'clamp':
            return torch.cat((res_eq, res_ineq_clamp), dim=1)
        else:
            return torch.cat((res_eq, res_ineq), dim=1), torch.cat((res_eq, res_ineq_clamp), dim=1)

    def grad_res(self, x, y, cost, dx, x0, lam, rho, y_update=False):
        " compute the gradient of the residual with respect to x"
        # ipdb.set_trace()
        if y_update:
            return lam + rho*self.dyn_res(x, dx, x0)
        cost_grad = self.compute_cost_gradient(x, cost)
        x.requires_grad_(True)
        dyn_res = self.dyn_res(x, dx, x0)
        grad_res = torch.autograd.grad(dyn_res, x, grad_outputs=y)[0]
        x_res = grad_res + cost_grad
        # y_res = y/rho - lam/rho - dyn_res
        # dr = torch.cat((x_res, y_res), dim=-1)
        # print("residual norms : ", dr.norm(), cost_grad.norm(), dyn_res.norm())
        # if y_update:
        #     return lam + rho*dyn_res
        return x_res

    def solve_nonlin(self, x, u, dx, x0, cost):
        best = None
        n_not_improved = 0
        xhats_qpf = torch.cat((x, u), dim=2).transpose(0,1)
        cost_total = self.compute_cost(xhats_qpf, cost)
        # ipdb.set_trace()
        # print("init", cost_total.mean().item())
        with torch.no_grad():
            for i in range(self.qp_iter):
                u_prev = u.clone()
                delta_x, delta_u, _ = self.single_qp(x, u, dx, x0, cost)
                # ipdb.set_trace()

                x, u, alpha, cost_total = self.line_search(x, u, delta_x, delta_u, dx, x0, cost)
                full_du_norm = (u - u_prev).norm()


                if best is None:
                    best = {
                        'x': list(torch.split(x, split_size_or_sections=1, dim=1)),
                        'u': list(torch.split(u, split_size_or_sections=1, dim=1)),
                        'costs': cost_total,
                    }
                else:
                    for j in range(self.n_batch):
                        if cost_total[j] <= best['costs'][j] + self.best_cost_eps:
                            n_not_improved = 0
                            best['x'][j] = x[:,j].unsqueeze(1)
                            best['u'][j] = u[:,j].unsqueeze(1)
                            best['costs'][j] = cost_total[j]

                # if self.verbose > 0:
                #     util.table_log('lqr', (
                #         ('iter', i),
                #         ('mean(cost)', torch.mean(best['costs']).item(), '{:.4e}'),
                #         ('||full_du||_max', max(full_du_norm).item(), '{:.2e}'),
                #         # ('||alpha_du||_max', max(alpha_du_norm), '{:.2e}'),
                #         # TODO: alphas, total_qp_iters here is for the current
                #         # iterate, not the best
                #         ('mean(alphas)', mean_alphas.item(), '{:.2e}'),
                #         ('total_qp_iters', n_total_qp_iter),
                #     ))
                # print(i, cost_total.mean().item(), full_du_norm)

                if full_du_norm < self.eps or \
                n_not_improved > self.not_improved_lim:
                    break
        
        x, u = torch.cat(best['x'], dim=1), torch.cat(best['u'], dim=1)
        delta_x, delta_u, _ = self.single_qp(x, u, dx, x0, cost)
        with torch.no_grad():
            _, _, alpha, cost_total = self.line_search(x, u, delta_x, delta_u, dx, x0, cost)
        x = x + delta_x * alpha
        u = u + delta_u * alpha        
        return x, u, cost_total

    def single_qp_ls(self, x, u, dx, x0, cost):
        best = None
        n_not_improved = 0
        xhats_qpf = torch.cat((x, u), dim=2).transpose(0,1)
        cost_total = self.compute_cost(xhats_qpf, cost)
        delta_x, delta_u, _ = self.single_qp(x, u, dx, x0, cost)
        # with torch.no_grad():
        #     _, _, alpha, cost_total = self.line_search(x, u, delta_x, delta_u, dx, x0, cost)
        x = x + delta_x #* alpha
        u = u + delta_u# * alpha
        xhats_qpf = torch.cat((x, u), dim=2).transpose(0,1)
        cost_total = self.compute_cost(xhats_qpf, cost)
        return x, u, cost_total


    def line_search(self, x, u, delta_x, delta_u, dx, x0, cost):
        # ipdb.set_trace()
        alpha_shape = [1, self.n_batch, 1]
        alpha = torch.ones(alpha_shape).to(x0)
        cost_total = self.compute_cost(torch.cat((x, u), dim=2).transpose(0,1), cost)
        for j in range(self.max_linesearch_iter):
            # x_new = x + delta_x * alpha
            u_new = u + delta_u * alpha
            x_new = self.rollout(x0, u_new, dx)#[:-1]
            xhats_qpf = torch.cat((x_new, u_new), dim=2).transpose(0,1)
            cost_total_new = self.compute_cost(xhats_qpf, cost)
            if (cost_total_new < cost_total).all():
                break
            else:
                mask = (cost_total_new >= cost_total).float()[None,:,None]
                alpha = alpha * self.linesearch_decay * mask + (1-mask) * alpha
            if j > self.max_linesearch_iter:
                print("line search failed")
                ipdb.set_trace()
        return x_new, u_new, alpha, cost_total_new

    def rollout(self, x, actions, dynamics):
        n_batch = x.size(0)
        x = [x]
        for t in range(self.T-1):
            xt = x[t]
            ut = actions[t]
            if isinstance(dynamics, LinDx):
                # ipdb.set_trace()
                new_x = util.bmv(dynamics.F[t], torch.cat([xt, ut], dim=-1)) + dynamics.f[t]
            else:
                new_x = dynamics(xt, ut)
            # ipdb.set_trace()
            x.append(new_x)
        return torch.stack(x, 0)


    def rollout_lin(self, x, actions, F, f):
        n_batch = x.size(0)
        x = [x]
        for t in range(self.T-1):
            xt = x[t]
            ut = actions[t]
            Ft = F[t]
            ft = f[t]
            new_x = util.bmv(Ft, torch.cat([xt, ut], dim=-1)) + ft
            x.append(new_x)
        return torch.stack(x, 0)

    def compute_cost(self, xu, cost):
        # ipdb.set_trace()
        C = cost.C.transpose(0,1)
        c = cost.c.transpose(0,1)
        if self.diag_cost:
            return 0.5*(( xu * C * xu ).sum(-1) + ( c * xu ).sum(-1)).sum(dim=-1)
        return 0.5*((xu.unsqueeze(-1)*C).sum(dim=-2)*xu).sum(dim=-1).sum(dim=-1) + (xu*c).sum(dim=-1).sum(dim=-1)
    
    def compute_cost_gradient(self, xu, cost):
        # ipdb.set_trace()
        C = cost.C.transpose(0,1)
        c = cost.c.transpose(0,1)
        if self.diag_cost:
            return C*xu + c
        return torch.cat((C*xu, c), dim=-1)