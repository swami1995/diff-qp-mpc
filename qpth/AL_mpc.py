import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.func import hessian, vmap, jacrev

import numpy as np
import numpy.random as npr

from collections import namedtuple

from enum import Enum

import sys, time

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
        device = self.u_upper.device
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
        self.al_iter = 2
        self.neq = n_state*(T-1) + n_state
        self.nineq = 0
        self.dyn_res_crit = 1e-4
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
        self.lamda_prev = torch.zeros(self.n_batch, self.neq+self.nineq).to(self.u_upper)#.type_as(x)#.reshape(self.n_batch, self.T, -1) 
        self.dyn_res_prev = 1000000
        # return self.Qi, self.Gi, self.Ai

    def forward(self, x0, cost, dx, dx_true=None, u_init=None, x_init=None):
        # QuadCost.C: [T, n_batch, n_tau, n_tau]
        # QuadCost.c: [T, n_batch, n_tau]
        if dx_true is None:
            self.dx_true = dx
        else:
            self.dx_true = dx_true

        # TODO: Clean up inferences, expansions, and assumptions made here.
        if self.n_batch is not None:
            n_batch = self.n_batch
        else:
            n_batch = cost.C.size(1)

        # ipdb.set_trace()
        assert cost.C.ndimension() == 4
        assert x0.ndimension() == 2 and x0.size(0) == n_batch

        if u_init is not None:
            u = u_init
            if u.ndimension() == 2:
                u = u.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        elif self.u_init is None:
            u = torch.zeros(self.T, n_batch, self.n_ctrl).type_as(x0.data)
        else:
            u = self.u_init
            if u.ndimension() == 2:
                u = u.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        u = u.type_as(x0.data)

        if x_init is not None:
            x = x_init
            if x.ndimension() == 2:
                x = x.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        elif self.x_init is None:
            # x = torch.zeros(self.T, n_batch, self.n_state).type_as(x0.data)
            x = self.rollout(x0, u, dx)#[:-1]
        else:
            x = self.x_init
            if x.ndimension() == 2:
                x = x.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        x = x.type_as(x0.data)

        # if self.verbose > 0:
        #     print('Initial mean(cost): {:.4e}'.format(
        #         torch.mean(util.get_cost(
        #             self.T, u, cost, dx, x_init=self.x_init
        #         )).item()
        #     ))

        best = None
        # ipdb.set_trace()
        if self.single_qp_solve:
            if self.diag_cost:
                cost = QuadCost(cost.C.diagonal(dim1=-2, dim2=-1), cost.c)
            x, u, cost_total = self.single_qp_ls(x, u, dx, x0, cost)
        else:
            x, u, cost_total = self.solve_nonlin(x, u, dx, x0, cost)
        
        self.x_init = x
        self.u_init = u
        return (x, u)
    
    def single_qp(self, x, u, dx, x0, cost, lamda_init=None, rho_init=None):
        x = x.transpose(0,1).double()
        u = u.transpose(0,1).double()
        x0 = x0.double()
        if lamda_init is None:
            lamda = self.lamda_prev.double()
        else:
            lamda = lamda_init
        if rho_init is None:
            rho = self.rho_prev
        else:
            rho = rho_init

        x_old, u_old = x, u
        with torch.no_grad():
            dyn_res_clamp_start = self.dyn_res(torch.cat((x, u), dim=2), dx, x0, res_type='clamp').view(self.n_batch, -1)
            cost_start = self.compute_cost(torch.cat((x, u), dim=2), cost.C.double(), cost.c.double())
            dyn_res_clamp = dyn_res_clamp_start = dyn_res_clamp_start.norm(dim=-1)
            if not self.just_initialized:
                cost_lam_hist = self.cost_lam_hist
                cost_hist = torch.stack(cost_lam_hist[0][::-1], dim=0)
                # ipdb.set_trace()
                cost_sim_idx = torch.max(cost_hist < cost_start[None], dim=0)[1]
                lamda_hist = torch.stack(cost_lam_hist[1][::-1], dim=0)
                rho_hist = torch.stack(cost_lam_hist[2][::-1], dim=0)
                batch_idx = torch.arange(self.n_batch, device=x.device, dtype=torch.long)
                lamda_hist = lamda_hist[cost_sim_idx, batch_idx]
                lamda = lamda*(lamda_hist.norm(dim=-1)/lamda.norm(dim=-1)).unsqueeze(-1)
                rho = rho_hist[cost_sim_idx, batch_idx]
        
        # if dyn_res_clamp_start > self.dyn_res_prev*2:
        #     dyn_res_ratio = dyn_res_clamp_start/self.dyn_res_prev
        #     rho = rho/dyn_res_ratio
        rho_init = rho
        # ipdb.set_trace()
        # print(dyn_res_clamp_start)
        Q, q = cost.C.double(), cost.c.double()
        cost_lam_hist = [[cost_start], [lamda], [rho]]
        # self.verbose = 1
        # Augmented Lagrangian updates with broyden for root finding of the residual
        for i in range(self.al_iter):
            xu = torch.cat((x, u), dim=2).detach().clone()
            # ipdb.set_trace()
            with torch.no_grad():
                y = lamda + rho*self.dyn_res(xu, dx, x0, res_type='clamp')
            # input = torch.cat((x, u), dim=2)
            # Hinv = 1/cost.C.transpose(0,1) # Might need to adaptively regularize this based on rho and solver progress/step size. 
            fn = lambda xi, yi, Qi, qi, y_update=False : self.grad_res(xi, yi, Qi, qi, dx, x0, lamda, rho, y_update)
            dyn_fn = lambda xi : self.dyn_res(xi, dx, x0)
            cost_fn = lambda xi, Qi, qi : self.compute_cost(xi, Qi, qi)
            merit_fn = lambda xi, Qi, qi, yi, x0i=x0, rhoi=rho, grad=False : self.merit_function(xi, Qi, qi, dx, x0i, yi, rhoi, grad)
            # out = util.broyden_AL(fn, merit_fn, dyn_fn, cost_fn, xu, y, threshold=10, eps=1e-6, rho=rho, Hinv=Hinv, ysize=y.shape[-1], ls=True, idx=True)
            # out = util.LBFGS_AL(fn, merit_fn, dyn_fn, cost_fn, xu, y, threshold=80, eps=1e-6, rho=rho, Hinv=Hinv, ysize=y.shape[-1], ls=True, idx=True)
            # out = util.Newton_AL(fn, merit_fn, dyn_fn, cost_fn, xu, y, threshold=1e-3, eps=1e-6, ls=True)
            newton_lamda = lambda Qi: util.NewtonAL.apply(fn, merit_fn, dyn_fn, cost_fn, xu, x0, y, lamda, rho, Qi, q, 1e-3, 1e-6, True, self.verbose)[0]
            # ls_lamda = lambda qi: util.CholeskySolver.apply(torch.diag_embed(Q.squeeze()), -qi.squeeze().unsqueeze(-1))
            # q = torch.randn_like(q)
            # util.check_fd_grads(ls_lamda, q, eps=1e-10)
            # util.check_fd_grads(newton_lamda, Q, eps=1e-10)
            # util.check_grads(newton_lamda, q, eps=1e-6)
            merit_hess = lambda xi : self.merit_hessian(xi, Q, q, dx, x0, lamda, rho)
            Hess = None#merit_hess(xu)
            out = util.NewtonAL.apply(fn, merit_fn, dyn_fn, cost_fn, merit_hess, xu, x0, y, lamda, rho, Q, q, Hess, 1e-3, 1e-6, True, self.verbose)
            # out = util.GD_AL(fn, input, threshold=10, eps=1e-6, rho=rho, Hinv=Hinv, ysize=y.shape[-1], ls=True, dyn_fn = dyn_fn, cost_fn = cost_fn)
            # x_res, lam_res = self.grad_res(x, lamda, cost, dx, x0, lamda, rho)    

            x_new, u_new = out[0][:,:,:self.n_state], out[0][:,:,self.n_state:]
            status = out[2]
            delta_x = x_new - x
            delta_u = u_new - u
            x, u = x_new, u_new
            delta_xu = torch.cat((delta_x, delta_u), dim=2)
            with torch.no_grad():
                dyn_res, dyn_res_clamp = self.dyn_res(torch.cat((x_new, u_new), dim=2), dx, x0, res_type='both')
                lamda = lamda + rho*dyn_res
                lamda = torch.cat([lamda[:, :self.neq], torch.clamp(lamda[:, self.neq:], min=0)], dim=1)
                cost_res = self.compute_cost(out[0], Q, q)
                dyn_res_clamp = torch.norm(dyn_res_clamp.view(self.n_batch, -1), dim=-1)
                if self.verbose > 0:
                    print("iter :", i, dyn_res_clamp.mean().item(), rho.mean().item(), cost_res.mean().item())
                # print("iter :", i, dyn_res_clamp.mean().item(), rho.mean().item(), cost_res.mean().item())
                
                # rho = torch.minimum(rho*10*status[:,None] + rho*(1-status[:,None]), rho_init*100)
                # rho = torch.minimum(rho*10, rho*0 + 100)
                rho = rho*10
                cost_lam_hist[0].append(cost_res)
                cost_lam_hist[1].append(lamda)
                cost_lam_hist[2].append(rho)
                # if  torch.logical_or(dyn_res_clamp < self.dyn_res_crit, dyn_res_clamp < dyn_res_clamp_start/self.dyn_res_factor).all():
                #     break
        # ipdb.set_trace()
        self.cost_lam_hist = cost_lam_hist
        self.lamda_prev = lamda
        self.rho_prev = rho
        self.dyn_res_prev = dyn_res_clamp
        self.just_initialized = False

        # delta_x = (x - x_old).transpose(0,1).float()
        # delta_u = (u - u_old).transpose(0,1).float()
        # return delta_x, delta_u, None
        return x.transpose(0,1).float(), u.transpose(0,1).float(), None
    
    def merit_function(self, xu, Q, q, dx, x0, lamda, rho, grad=False):
        bsz = xu.size(0)
        cost_total = self.compute_cost(xu, Q, q)
        res, res_clamp = self.dyn_res(xu, dx, x0, res_type='both')
        # ipdb.set_trace()
        if grad:
            return cost_total +  0.5*rho[:,0]*(res_clamp*res_clamp).view(bsz, -1).sum(dim=1)
        else:
            return cost_total + (lamda*res).view(bsz, -1).sum(dim=1) + 0.5*rho[:,0]*(res_clamp*res_clamp).view(bsz, -1).sum(dim=1)
        # else:
        #     return cost_total.mean() + (lamda*res_clamp).view(bsz, -1).sum(dim=1).mean() + 0.5*rho*(res_clamp*res_clamp).view(bsz, -1).sum(dim=1).mean()
    
    def merit_hessian(self, xu, Q, q, dx, x0, lamda, rho):
        bsz = xu.size(0)
        xi_shape = (1, xu.shape[1], xu.shape[2])
        res, res_clamp = self.dyn_res(xu, dx, x0, res_type='both')
        dyn_res_fn_clamp = lambda xi, xi0 : self.dyn_res(xi.view(xi_shape), dx, xi0.view(1, x0.shape[1]), res_type='clamp')[0]
        dyn_res_fn = lambda xi, xi0 : self.dyn_res(xi.view(xi_shape), dx, xi0.view(1, x0.shape[1]), res_type='noclamp')[0]
        # constraint jacobian
        constraint_jac_clamp = vmap(jacrev(dyn_res_fn_clamp))(xu.view(bsz, -1), x0)
        constraint_jac = vmap(jacrev(dyn_res_fn))(xu.view(bsz, -1), x0)
        constraint_hess = torch.bmm(constraint_jac_clamp.permute(0,2,1), constraint_jac_clamp)
        if self.diag_cost:
            Qfull = torch.diag_embed(Q.reshape((bsz, -1)))
        Hess = Qfull + rho[:,:,None]*constraint_hess
        Hess_clip = Qfull + torch.clamp(rho[:,:,None], max=100)*constraint_hess
        # print("ineqjac :", constraint_jac_clamp[:, 6:].norm().item(), constraint_jac[:, 6:].norm().item(), res_clamp[:,6:].norm().item(), res[:,6:].norm().item())
        # print(constraint_hess.norm().item(), Qfull.norm().item(), constraint_jac_clamp.norm().item(), constraint_jac.norm().item())
        # ipdb.set_trace()
        return Hess, Hess#_clip
    def dyn_res_eq(self, x, u, dx, x0):
        " split x into state and control and compute dynamics residual using dx"
        # ipdb.set_trace()
        bsz = x.size(0)
        if isinstance(dx, LinDx):
            x_next = (dx.F.permute(1,0,2,3)*torch.cat((x, u), dim=2)[:,:-1,None,:]).sum(dim=-1) + dx.f.permute(1,0,2)
        else:
            x_next = dx(x, u)[:,:-1]
            
        res = (x_next - x[:,1:,:])#.reshape(self.n_batch, -1)
        res_init = (x[:,0,:] - x0).reshape(bsz, 1, -1)
        # print(res.shape, res_init.shape, x0.shape, x.shape)
        if self.add_goal_constraint:
            res_goal = (x[:,-1,:] - self.x_goal).reshape(bsz, -1)
            res = torch.cat((res, res_init, res_goal), dim=1)
        else:
            res = torch.cat((res, res_init), dim=1)
        res = res.reshape(bsz, -1)
        return res
    
    def dyn_res_ineq(self, x, u, dx, x0):
        # Add state constraints if self.x_lower and self.x_upper are not None
        bsz = x.size(0)
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
        res = res.reshape(bsz, -1)
        res_clamp = torch.clamp(res, min=0)
        return res, res_clamp
    
    def dyn_res(self, x, dx, x0, res_type='clamp'):
        x = x.reshape(-1, self.T, self.n_state+self.n_ctrl)
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

    def grad_res(self, x, y, Q, q, dx, x0, lam, rho, y_update=False):
        " compute the gradient of the residual with respect to x"
        # ipdb.set_trace()
        if y_update:
            return lam + rho*self.dyn_res(x, dx, x0)
        cost_grad = self.compute_cost_gradient(x, Q, q)
        x.requires_grad_(True)
        with torch.enable_grad():
            dyn_res = self.dyn_res(x, dx, x0)
            grad_res = torch.autograd.grad(dyn_res, x, grad_outputs=y)[0]
        x_res = grad_res + cost_grad
        # y_res = y/rho - lam/rho - dyn_res
        # dr = torch.cat((x_res, y_res), dim=-1)
        # print("residual norms : ", dr.norm(), cost_grad.norm(), dyn_res.norm())
        # if y_update:
        #     return lam + rho*dyn_res
        return x_res

    def solve_nonlin(self, x, u, dx, x0, cost, lamda_init=None, rho_init=None):
        x = x.transpose(0,1)
        u = u.transpose(0,1)
        if lamda_init is None:
            lamda = self.lamda_prev
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
            dyn_res_clamp = torch.norm(dyn_res_clamp).item()
            if  dyn_res_clamp < self.dyn_res_crit:
                break

            rho = rho*10
            x, u = x_new, u_new
        
        self.lamda_prev = lamda
        self.rho_prev = rho

        delta_x = (x - x_old).transpose(0,1)
        delta_u = (u - u_old).transpose(0,1)
        return delta_x, delta_u, out['result']

    def single_qp_ls(self, x, u, dx, x0, cost):
        best = None
        n_not_improved = 0
        # xhats_qpf = torch.cat((x, u), dim=2).transpose(0,1)
        # cost_total = self.compute_cost(xhats_qpf, cost.C, cost.c)
        x, u, _ = self.single_qp(x, u, dx, x0, cost)
        # with torch.no_grad():
        #     _, _, alpha, cost_total = self.line_search(x, u, delta_x, delta_u, dx, x0, cost)
        # x = x + delta_x #* alpha
        # u = u + delta_u# * alpha
        # xhats_qpf = torch.cat((x, u), dim=2).transpose(0,1)
        # cost_total = self.compute_cost(xhats_qpf, cost.C, cost.c)
        return x, u, 0


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

    def compute_cost(self, xu, Q, q):
        # ipdb.set_trace()
        C = Q.transpose(0,1)
        c = q.transpose(0,1)
        if self.diag_cost:
            return (0.5*( xu * C * xu ).sum(-1) + ( c * xu ).sum(-1)).sum(dim=-1)
        return 0.5*((xu.unsqueeze(-1)*C).sum(dim=-2)*xu).sum(dim=-1).sum(dim=-1) + (xu*c).sum(dim=-1).sum(dim=-1)
    
    def compute_cost_gradient(self, xu, Q, q):
        # ipdb.set_trace()
        C = Q.transpose(0,1)
        c = q.transpose(0,1)
        if self.diag_cost:
            return C*xu + c
        return torch.cat((C*xu, c), dim=-1)

    def reinitialize(self, x):
        self.u_init = None
        self.x_init = None
        self.rho_prev = torch.ones((self.n_batch,1), device=x.device, dtype=x.dtype)
        self.lamda_prev = torch.zeros(self.n_batch, self.neq+self.nineq, device=x.device, dtype=x.dtype)
        self.dyn_res_prev = 1000000
        self.just_initialized = True