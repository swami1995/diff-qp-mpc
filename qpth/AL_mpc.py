import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.func import hessian, vmap, jacrev

import numpy as np
import numpy.random as npr

from enum import Enum

import sys, time

# from .pnqp import pnqp
# from .lqr_step import LQRStep
# from .dynamics import CtrlPassthroughDynamics
from . import qp
import ipdb
from . import al_utils
from . import util

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
            u_init=None,
            x_init=None,
            al_iter=20,
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
            solver_type='dense',
            add_goal_constraint=False,
            x_goal=None,
            diag_cost=True, 
            ineqG=None,
            ineqh=None,
            dtype=torch.float64
    ):
        super().__init__()

        assert (u_lower is None) == (u_upper is None)
        assert max_linesearch_iter > 0

        self.dtype = dtype
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.u_lower = u_lower.to(self.dtype)
        self.u_upper = u_upper.to(self.dtype)
        self.x_upper = None
        self.x_lower = None
        self.x_goal = x_goal
        self.ineqG = ineqG
        self.ineqh = ineqh        

        if not isinstance(u_lower, float):
            self.u_lower = util.detach_maybe(self.u_lower)

        if not isinstance(u_upper, float):
            self.u_upper = util.detach_maybe(self.u_upper)

        self.u_init = util.detach_maybe(u_init)
        self.x_init = util.detach_maybe(x_init)
        self.verbose = verbose
        self.eps = eps
        self.back_eps = back_eps
        self.n_batch = n_batch
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.exit_unconverged = exit_unconverged
        self.detach_unconverged = detach_unconverged
        self.backprop = backprop

        self.slew_rate_penalty = slew_rate_penalty
        self.solver_type = solver_type
        self.add_goal_constraint = add_goal_constraint
        self.diag_cost = diag_cost
        self.al_iter = al_iter
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
        self.mask = torch.ones(self.n_batch, self.T, 1).to(self.u_upper)
        # return self.Qi, self.Gi, self.Ai

    def forward(self, x0, cost, dx, dx_jac, u_init=None, x_init=None):
        # QuadCost.C: [T, n_batch, n_tau, n_tau]
        # QuadCost.c: [T, n_batch, n_tau]

        # TODO: Clean up inferences, expansions, and assumptions made here.
        if self.n_batch is not None:
            n_batch = self.n_batch
        else:
            n_batch = cost.C.size(0)

        # ipdb.set_trace()
        assert cost.C.ndimension() == 4
        assert x0.ndimension() == 2 and x0.size(0) == n_batch

        if u_init is not None:
            u = u_init
            if u.ndimension() == 2:
                u = u.unsqueeze(0).expand(n_batch, self.T, -1).clone()
        elif self.u_init is None:
            u = torch.zeros(n_batch, self.T, self.n_ctrl).type_as(x0.data)
        else:
            u = self.u_init
            if u.ndimension() == 2:
                u = u.unsqueeze(0).expand(n_batch, self.T, -1).clone()
        u = u.type_as(x0.data)

        if x_init is not None:
            x = x_init
            if x.ndimension() == 2:
                x = x.unsqueeze(0).expand(n_batch, self.T, -1).clone()
        elif self.x_init is None:
            # x = torch.zeros(self.T, n_batch, self.n_state).type_as(x0.data)
            x = self.rollout(x0, u, dx)#[:-1]
        else:
            x = self.x_init
            if x.ndimension() == 2:
                x = x.unsqueeze(0).expand(n_batch, self.T, -1).clone()
        x = x.type_as(x0.data)

        if self.verbose > 0:
            print('Initial mean(cost): {:.4e}'.format(
                torch.mean(util.get_cost(
                    self.T, u, cost, dx, x_init=self.x_init
                )).item()
            ))

        best = None
        if self.diag_cost:
            cost = al_utils.QuadCost(cost.C.diagonal(dim1=-2, dim2=-1), cost.c)
        x, u = self.al_solve(x, u, dx, dx_jac, x0, cost)
        
        self.x_init = x
        self.u_init = u
        return (x, u)
    
    def al_solve(self, x, u, dx, dx_jac, x0, cost, lamda_init=None, rho_init=None):
        # dx_jac = [torch.zeros(self.n_batch, self.T-1, self.n_state, self.n_state).to(x), torch.zeros(self.n_batch, self.T-1, self.n_state, self.n_ctrl).to(x)]
        x = x.to(self.dtype)
        u = u.to(self.dtype)
        x0 = x0.to(self.dtype)
        # start1 = time.time()
        if lamda_init is None:
            lamda = self.lamda_prev.to(self.dtype)
        else:
            lamda = lamda_init
        if rho_init is None:
            rho = self.rho_prev
        else:
            rho = rho_init
        # ipdb.set_trace()
        x_old, u_old = x, u
        with torch.no_grad():
            dyn_res_clamp_start = self.dyn_res(torch.cat((x, u), dim=2), dx, x0, res_type='clamp').view(self.n_batch, -1)
            cost_start = self.compute_cost(torch.cat((x, u), dim=2), cost.C.double(), cost.c.double())
            dyn_res_clamp = dyn_res_clamp_start = dyn_res_clamp_start.norm(dim=-1)
            if not self.just_initialized:
                cost_hist = torch.stack(self.cost_lam_hist[0][::-1], dim=0)
                lamda_hist = torch.stack(self.cost_lam_hist[1][::-1], dim=0)
                rho_hist = torch.stack(self.cost_lam_hist[2][::-1], dim=0)
                lamda, rho = al_utils.warm_start_al(x, lamda, rho, cost_start, cost_hist, lamda_hist, rho_hist)        
        rho_init = rho
        Q = cost.C.to(self.dtype)
        q = cost.c.to(self.dtype)
        cost_lam_hist = [[cost_start], [lamda], [rho]]
        # end1 = time.time()
        # Augmented Lagrangian updates
        for i in range(self.al_iter):
            # start2 = time.time()
            xu = torch.cat((x, u), dim=2).detach().clone()
            dyn_fn = lambda xi : self.dyn_res(xi, dx, x0)
            cost_fn = lambda xi, Qi, qi : self.compute_cost(xi, Qi, qi)
            merit_fn = lambda xi, Qi, qi, yi, x0i=x0, rhoi=rho, grad=False : self.merit_function(xi, Qi, qi, dx, x0i, yi, rhoi, grad)
            merit_grad_hess = lambda xi, Q, q, lamda : self.merit_grad_hess(xi, Q, q, dx, dx_jac, x0, lamda, rho)
            
            out = al_utils.NewtonAL.apply(merit_fn, dyn_fn, cost_fn, merit_grad_hess, xu, x0, lamda, rho, Q, q, 1e-3, 1e-6, True)
            x_new, u_new = out[0][:,:,:self.n_state], out[0][:,:,self.n_state:]
            # end2 = time.time()
            status = out[1]
            x, u = x_new, u_new
            with torch.no_grad():
                dyn_res, dyn_res_clamp = self.dyn_res(torch.cat((x_new, u_new), dim=2), dx, x0, res_type='both')
                lamda = lamda + rho*dyn_res
                lamda = torch.cat([lamda[:, :self.neq], torch.clamp(lamda[:, self.neq:], min=0)], dim=1)
                cost_res = self.compute_cost(out[0], Q, q)
                dyn_res_clamp = torch.norm(dyn_res_clamp.view(self.n_batch, -1), dim=-1)
                # print("iter :", i, dyn_res_clamp.mean().item(), rho.mean().item(), cost_res.mean().item())
                
                rho = torch.minimum(rho*10*status[:,None] + rho*(1-status[:,None]), rho_init*100)
                # rho = rho * 10
                cost_lam_hist[0].append(cost_res)
                cost_lam_hist[1].append(lamda)
                cost_lam_hist[2].append(rho)
            # end3 = time.time()
            # print("outer time: ", end1 - start1, end2 - start2, end3 - end2)
        # ipdb.set_trace()
        self.cost_lam_hist = cost_lam_hist
        self.lamda_prev = lamda
        self.rho_prev = rho
        self.dyn_res_prev = dyn_res_clamp
        self.just_initialized = False
        x = x.float()
        u = u.float()
        return x, u
    
    def merit_function(self, xu, Q, q, dx, x0, lamda, rho, grad=False):
        return al_utils.merit_function(xu, Q, q, dx, x0, lamda, rho, self.x_lower, self.x_upper, self.u_lower, self.u_upper, self.diag_cost)
    def merit_hessian(self, xu, Q, q, dx, dx_jac, x0, lamda, rho):
        return al_utils.merit_hessian(xu, Q, q, dx, dx_jac, x0, lamda, rho, self.x_lower, self.x_upper, self.u_lower, self.u_upper, self.diag_cost)

    def merit_grad_hess(self, xu, Q, q, dx, dx_jac, x0, lamda, rho):
        return al_utils.merit_grad_hessian(xu, Q, q, dx, dx_jac, x0, lamda, rho, self.x_lower, self.x_upper, self.u_lower, self.u_upper, self.diag_cost)
    def dyn_res_eq(self, x, u, dx, x0, mask=None):
        " split x into state and control and compute dynamics residual using dx"
        # ipdb.set_trace()
        bsz, T, nx = x.shape
        nu = u.shape[-1]
        if isinstance(dx, al_utils.LinDx):
            x_next = (dx.F.permute(1,0,2,3)*torch.cat((x, u), dim=2)[:,:-1,None,:]).sum(dim=-1) + dx.f.permute(1,0,2)
        else:
            x_next = dx(x.reshape(-1, nx), u.reshape(-1, nu)).view(bsz, T-1, nx)
        
        # if mask is None:
        #     mask = self.mask
        res = (x_next - x[:,1:,:])#*mask[:,:-1]#.reshape(self.n_batch, -1)
        # ipdb.set_trace()
        res_init = (x[:,0,:] - x0).reshape(bsz, 1, -1)
        # print(res.shape, res_init.shape, x0.shape, x.shape)
        if self.add_goal_constraint:
            res_goal = (x[:,-1,:] - self.x_goal).reshape(bsz, -1)
            res = torch.cat((res, res_init, res_goal), dim=1)
        else:
            try:
                res = torch.cat((res, res_init), dim=1)
            except:
                ipdb.set_trace()
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
    
    def dyn_res(self, xu, dx, x0, res_type='clamp'):
        res, res_clamp = al_utils.dyn_res(xu, dx, x0, self.x_lower, self.x_upper, self.u_lower, self.u_upper)
        if res_type == 'noclamp':
            return res
        elif res_type == 'clamp':
            return res_clamp
        else:
            return res, res_clamp

    def rollout(self, x, actions, dynamics):
        n_batch = x.size(0)
        x = [x]
        for t in range(self.T-1):
            xt = x[t]
            ut = actions[:,t]
            if isinstance(dynamics, al_utils.LinDx):
                # ipdb.set_trace()
                new_x = util.bmv(dynamics.F[:,t], torch.cat([xt, ut], dim=-1)) + dynamics.f[:,t]
            else:
                new_x = dynamics(xt, ut)
            # ipdb.set_trace()
            x.append(new_x)
        return torch.stack(x, 1)


    def rollout_lin(self, x, actions, F, f):
        n_batch = x.size(0)
        x = [x]
        for t in range(self.T-1):
            xt = x[t]
            ut = actions[:,t]
            Ft = F[:,t]
            ft = f[:,t]
            new_x = util.bmv(Ft, torch.cat([xt, ut], dim=-1)) + ft
            x.append(new_x)
        return torch.stack(x, 1)

    def compute_cost(self, xu, Q, q):
        return al_utils.compute_cost(xu, Q, q, self.diag_cost)
    
    def compute_cost_gradient(self, xu, Q, q):
        return al_utils.compute_cost_gradient(xu, Q, q, self.diag_cost)

    def reinitialize(self, x, mask):
        self.u_init = None
        self.x_init = None
        self.rho_prev = torch.ones((self.n_batch,1), device=x.device, dtype=x.dtype)
        self.lamda_prev = torch.zeros(self.n_batch, self.neq+self.nineq, device=x.device, dtype=x.dtype)
        self.dyn_res_prev = 1000000
        self.just_initialized = True
        self.mask = mask