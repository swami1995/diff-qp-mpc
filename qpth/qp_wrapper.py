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
from .solvers.pdipm import SparseStructure as ss
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
        lqr_iter: The number of LQR iterations to perform.
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
            lqr_iter=10,
    ):
        super().__init__()

        assert (u_lower is None) == (u_upper is None)
        assert max_linesearch_iter > 0

        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.u_lower = u_lower
        self.u_upper = u_upper

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
        self.lqr_iter = lqr_iter

        if solver_type == 'dense':
            idxs_1 = torch.arange(n_state + n_ctrl)
            idxs_0 = torch.arange(n_state)
            idxs_0, idxs_1 = torch.meshgrid(idxs_0, idxs_1, indexing='ij')
            self.A_slices_xu1 = torch.cat([idxs_1 + (n_state+n_ctrl)*i for i in range(T-1)], dim=0).view(-1)
            self.A_slices_xu0 = torch.cat([idxs_0 + (n_state)*i for i in range(T-1)], dim=0).view(-1)
            idxs_1 = torch.arange(n_state + n_ctrl)
            idxs_0 = torch.arange(n_state)
            self.A_slices_xx1 = torch.cat([torch.arange(n_state) + (n_state+n_ctrl)*(i+1) for i in range(T-1)], dim=0).view(-1)
            self.A_slices_xx0 = torch.cat([torch.arange(n_state) + (n_state)*i for i in range(T-1)], dim=0).view(-1)

            idx = torch.arange(n_state + n_ctrl)
            idx_0, idx_1 = torch.meshgrid(idx, idx, indexing='ij')
            self.Q_slices_xu1 = torch.cat([idx_1 + (n_state+n_ctrl)*i for i in range(T)], dim=0).view(-1)
            self.Q_slices_xu0 = torch.cat([idx_0 + (n_state+n_ctrl)*i for i in range(T)], dim=0).view(-1)

            self.G_slices_uu1 = torch.cat([torch.arange(n_ctrl) + n_state + (n_state+n_ctrl)*(i) for i in range(T)], dim=0).view(-1)
            self.G_slices_uu0 = torch.cat([torch.arange(n_ctrl) + (n_ctrl)*i for i in range(T)], dim=0).view(-1)
        elif solver_type == 'sparse':            
            Qrw_ptr = torch.arange(0, (n_state+n_ctrl)*(T)+1)
            Qcol_ind = torch.arange(0, (n_state+n_ctrl)*(T))
            Acol_ind = []
            Arw_ptr = [0]
            
            for i in range(n_state*(T-1)):
                i_ = i // n_state
                j = i % n_state
                Acol_ind.append(torch.arange(n_state + n_ctrl) + (n_state+n_ctrl)*(i_))
                Acol_ind.append(torch.Tensor([j]).long() + (n_state+n_ctrl)*(i_+1))
                Arw_ptr.append(Acol_ind[-2].shape[0] + Acol_ind[-1].shape[0] + Arw_ptr[-1])
            Acol_ind.append(torch.arange(n_state))
            Arw_ptr += list(Arw_ptr[-1] + np.arange(n_state) + 1)
            # ipdb.set_trace()
            Arw_ptr = torch.Tensor(Arw_ptr).long()
            Acol_ind = torch.cat(Acol_ind, dim=0)
            
            if u_upper is None:
                Grw_ptr = torch.arange(0, n_ctrl+1)
                Gcol_ind = torch.arange(n_ctrl) + n_state + (n_state+n_ctrl)*(T-1)
            else:
                Grw_ptr = torch.arange(0, n_ctrl*(T)*2+1)
                Gcol_ind = torch.cat([torch.arange(n_ctrl) + n_state + (n_state+n_ctrl)*(i) for i in range(T)], dim=0).view(-1)
                Gcol_ind = torch.cat([Gcol_ind, Gcol_ind], dim=0)
                # Gcol_ind = torch.arange(n_ctrl) + n_state + (n_state+n_ctrl)*(T-1)

            self.Qi = ss.SparseStructure(Qrw_ptr, Qcol_ind)
            self.Ai = ss.SparseStructure(Arw_ptr, Acol_ind, value=None, num_rows=n_state*T, num_cols=(n_state+n_ctrl)*T)
            self.Gi = ss.SparseStructure(Grw_ptr, Gcol_ind)
            self.QP = qp.SparseQPFunction(self.Qi, self.Gi, self.Ai, bsz=n_batch)
        # return self.Qi, self.Gi, self.Ai

    def forward(self, x0, cost, dx):
        # QuadCost.C: [T, n_batch, n_tau, n_tau]
        # QuadCost.c: [T, n_batch, n_tau]
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

        assert x0.ndimension() == 2 and x0.size(0) == n_batch

        if self.u_init is None:
            u = torch.zeros(self.T, n_batch, self.n_ctrl).type_as(x0.data)
        else:
            u = self.u_init
            if u.ndimension() == 2:
                u = u.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        u = u.type_as(x0.data)

        if self.x_init is None:
            x = torch.zeros(self.T, n_batch, self.n_state).type_as(x0.data)
        else:
            x = self.x_init
            if x.ndimension() == 2:
                x = x.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        x = x.type_as(x0.data)

        if self.verbose > 0:
            print('Initial mean(cost): {:.4e}'.format(
                torch.mean(util.get_cost(
                    self.T, u, cost, dx, x_init=x_init
                )).item()
            ))

        best = None
        # ipdb.set_trace()
        # x, u, cost_total = self.single_qp(x, u, dx, x0, cost)
        x, u, cost_total = self.solve_nonlin(x, u, dx, x0, cost)
        
        return (x, u)
    
    def single_qp(self, x, u, dx, x0, cost):
        if isinstance(dx, LinDx):
            F, f = dx.F, dx.f
            if f is None:
                f = torch.zeros((self.T-1, self.n_batch, self.n_state)).to(x0)
        else:
            F, f = self.linearize_dynamics(
                x, util.detach_maybe(u), dx, diff=False)
        

        # ipdb.set_trace()
        if self.solver_type == 'dense':
            Q, q = self.compute_Qq_dense(cost.C, cost.c)
            A, b = self.compute_Ab_dense(F, f, x0)
            G, h = self.compute_Gh_dense(x0)
            # xhats_qpf = qp.QPFunction()(Q, q, G, h, A, b)
            xhats_qpf = qp.DenseQPFunction()(Q, q, G, h, A, b)
        else:
            Qv, q = self.compute_Qq_sparse(cost.C, cost.c)
            Av, b = self.compute_Ab_sparse(F, f, x0)
            Gv, h = self.compute_Gh_sparse(x0)
            xhats_qpf = self.QP(Qv, q, Gv, h, Av, b)
        xhats_qpf = xhats_qpf.reshape(self.n_batch, self.T, -1)
        x = xhats_qpf[:, :, :self.n_state].transpose(0,1)
        u = xhats_qpf[:, :, self.n_state:].transpose(0,1)
        cost_total = self.compute_cost(xhats_qpf, cost)
        return x, u, cost_total

    def solve_nonlin(self, x, u, dx, x0, cost):
        best = None
        n_not_improved = 0
        with torch.no_grad():
            for i in range(self.lqr_iter):
                u_prev = u.clone()
                x, u, cost_total = self.single_qp(x, u, dx, x0, cost)
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
                print(i, cost_total.mean().item(), full_du_norm)

                if full_du_norm < self.eps or \
                n_not_improved > self.not_improved_lim:
                    break
        
        x, u = torch.cat(best['x'], dim=1), torch.cat(best['u'], dim=1)
        x, u, cost_total = self.single_qp(x, u, dx, x0, cost)
        return x, u, cost_total


    def approximate_cost(self, x, u, Cf, diff=True):
        with torch.enable_grad():
            tau = torch.cat((x, u), dim=2).data
            tau = Variable(tau, requires_grad=True)
            if self.slew_rate_penalty is not None:
                print("""
MPC Error: Using a non-convex cost with a slew rate penalty is not yet implemented.
The current implementation does not correctly do a line search.
More details: https://github.com/locuslab/mpc.pytorch/issues/12
""")
                sys.exit(-1)
                differences = tau[1:, :, -self.n_ctrl:] - tau[:-1, :, -self.n_ctrl:]
                slew_penalty = (self.slew_rate_penalty * differences.pow(2)).sum(-1)
            costs = list()
            hessians = list()
            grads = list()
            for t in range(self.T):
                tau_t = tau[t]
                if self.slew_rate_penalty is not None:
                    cost = Cf(tau_t) + (slew_penalty[t-1] if t > 0 else 0)
                else:
                    cost = Cf(tau_t)

                grad = torch.autograd.grad(cost.sum(), tau_t,
                                           create_graph=True, retain_graph=True)[0]
                hessian = list()
                for v_i in range(tau.shape[2]):
                    hessian.append(
                        torch.autograd.grad(grad[:, v_i].sum(), tau_t,
                                            retain_graph=True)[0]
                    )
                hessian = torch.stack(hessian, dim=-1)
                costs.append(cost)
                grads.append(grad - util.bmv(hessian, tau_t))
                hessians.append(hessian)
            costs = torch.stack(costs, dim=0)
            grads = torch.stack(grads, dim=0)
            hessians = torch.stack(hessians, dim=0)
            if not diff:
                return hessians.data, grads.data, costs.data
            return hessians, grads, costs

    # @profile
    def linearize_dynamics(self, x, u, dynamics, diff):
        # TODO: Cleanup variable usage.

        n_batch = x[0].size(0)

        if self.grad_method == GradMethods.ANALYTIC:
            _u = Variable(u[:-1].view(-1, self.n_ctrl), requires_grad=True)
            _x = Variable(x[:-1].contiguous().view(-1, self.n_state),
                          requires_grad=True)

            # This inefficiently calls dynamics again, but is worth it because
            # we can efficiently compute grad_input for every time step at once.
            _new_x = dynamics(_x, _u)

            # This check is a little expensive and should only be done if
            # modifying this code.
            # assert torch.abs(_new_x.data - torch.cat(x[1:])).max() <= 1e-6

            if not diff:
                _new_x = _new_x.data
                _x = _x.data
                _u = _u.data

            R, S = dynamics.grad_input(_x, _u)

            f = _new_x - util.bmv(R, _x) - util.bmv(S, _u)
            f = f.view(self.T-1, n_batch, self.n_state)

            R = R.contiguous().view(self.T-1, n_batch, self.n_state, self.n_state)
            S = S.contiguous().view(self.T-1, n_batch, self.n_state, self.n_ctrl)
            F = torch.cat((R, S), 3)

            if not diff:
                F, f = list(map(Variable, [F, f]))
            return F, f
        else:
            # TODO: This is inefficient and confusing.
            x_init = x[0]
            x = [x_init]
            F, f = [], []
            with torch.enable_grad():
                for t in range(self.T):
                    if t < self.T-1:
                        xt = Variable(x[t], requires_grad=True)
                        ut = Variable(u[t], requires_grad=True)
                        xut = torch.cat((xt, ut), 1)
                        new_x = dynamics(xt, ut)

                        # Linear dynamics approximation.
                        if self.grad_method in [GradMethods.AUTO_DIFF,
                                                GradMethods.ANALYTIC_CHECK]:
                            Rt, St = [], []
                            for j in range(self.n_state):
                                Rj, Sj = torch.autograd.grad(
                                    new_x[:,j].sum(), [xt, ut],
                                    retain_graph=True)
                                if not diff:
                                    Rj, Sj = Rj.data, Sj.data
                                Rt.append(Rj)
                                St.append(Sj)
                            Rt = torch.stack(Rt, dim=1)
                            St = torch.stack(St, dim=1)
                            if torch.isnan(Rt).any() or torch.isnan(St).any():
                                ipdb.set_trace()

                            if self.grad_method == GradMethods.ANALYTIC_CHECK:
                                assert False # Not updated
                                Rt_autograd, St_autograd = Rt, St
                                Rt, St = dynamics.grad_input(xt, ut)
                                eps = 1e-8
                                if torch.max(torch.abs(Rt-Rt_autograd)).data[0] > eps or \
                                torch.max(torch.abs(St-St_autograd)).data[0] > eps:
                                    print('''
            nmpc.ANALYTIC_CHECK error: The analytic derivative of the dynamics function may be off.
                                    ''')
                                else:
                                    print('''
            nmpc.ANALYTIC_CHECK: The analytic derivative of the dynamics function seems correct.
            Re-run with GradMethods.ANALYTIC to continue.
                                    ''')
                                sys.exit(0)
                        elif self.grad_method == GradMethods.FINITE_DIFF:
                            Rt, St = [], []
                            for i in range(n_batch):
                                Ri = util.jacobian(
                                    lambda s: dynamics(s, ut[i]), xt[i], 1e-4
                                )
                                Si = util.jacobian(
                                    lambda a : dynamics(xt[i], a), ut[i], 1e-4
                                )
                                if not diff:
                                    Ri, Si = Ri.data, Si.data
                                Rt.append(Ri)
                                St.append(Si)
                            Rt = torch.stack(Rt)
                            St = torch.stack(St)
                        else:
                            assert False

                        Ft = torch.cat((Rt, St), 2)
                        F.append(Ft)

                        if not diff:
                            xt, ut, new_x = xt.data, ut.data, new_x.data
                        ft = new_x - util.bmv(Rt, xt) - util.bmv(St, ut)
                        f.append(ft)

                    if t < self.T-1:
                        x.append(util.detach_maybe(new_x))

            F = torch.stack(F, 0)
            f = torch.stack(f, 0)
            if not diff:
                F, f = list(map(Variable, [F, f]))
            return F, f

    def rollout(self, x, actions, dynamics):
        n_batch = x.size(0)
        x = [x]
        for t in range(self.T):
            xt = x[t]
            ut = actions[t]
            new_x = dynamics(xt, ut)
            x.append(new_x)
        return torch.stack(x, 0)
    
    def compute_Ab_dense(self, F, f, x0):
        T, n_batch, n_state, n_tau = F.size()
        A = torch.zeros(n_batch, (T+1)*n_state, (T+1)*n_tau).to(F)
        b = torch.zeros(n_batch, (T+1)*n_state).to(F)
        A[:, self.A_slices_xu0, self.A_slices_xu1] = F.transpose(0,1).contiguous().view(n_batch, -1)
        A[:, self.A_slices_xx0, self.A_slices_xx1] = -1
        A[:, T*n_state:, :n_state] += torch.eye(n_state).unsqueeze(0).to(F)#.expand(n_batch, n_state, n_state)
        b[:, :T*n_state] = -f.transpose(0,1).contiguous().view(n_batch, -1)
        b[:, T*n_state:] = x0
        return A, b

    def compute_Qq_dense(self, C, c):
        T, n_batch, n_tau, n_tau = C.size()
        Q = torch.zeros(n_batch, (T)*n_tau, (T)*n_tau).to(C)
        Q[:, self.Q_slices_xu0, self.Q_slices_xu1] = C.transpose(0,1).contiguous().view(n_batch, -1)
        q = c.transpose(0,1).contiguous().view(n_batch, -1)
        return Q, q
    
    def compute_Gh_dense(self, x0):
        T, n_batch, n_state, n_ctrl = self.T, self.n_batch, self.n_state, self.n_ctrl
        n_tau = n_state + n_ctrl
        if self.u_upper is None:
            G = torch.zeros(n_batch, n_ctrl, (T)*n_tau).to(x0)
            h = torch.ones(n_batch, n_ctrl).to(x0)
            G[:, torch.arange(n_ctrl), torch.arange(n_ctrl)+(T-1)*n_tau+n_state] = 1
            # h[:, :] *= self.u_upper
        else:
            G = torch.zeros(n_batch, 2*T*n_ctrl, (T)*n_tau).to(x0)
            h = torch.ones(n_batch, 2*T*n_ctrl).to(x0)
            G[:, self.G_slices_uu0, self.G_slices_uu1] = 1.0
            G[:, self.G_slices_uu0+T*n_ctrl, self.G_slices_uu1] = -1.0
            h[:, :T*n_ctrl] *= self.u_upper
            h[:, T*n_ctrl:] = -self.u_lower
        return G, h

    # def compute_Gh_dense(self):
    #     T, n_batch, n_state, n_ctrl = self.T, self.n_batch, self.n_state, self.n_ctrl
    #     n_tau = n_state + n_ctrl
    #     G = torch.zeros(n_batch, n_ctrl, (T)*n_tau)
    #     h = torch.ones(n_batch, n_ctrl)
    #     G[:, torch.arange(n_ctrl), torch.arange(n_ctrl)+(T-1)*n_tau+n_state] = 1
    #     h[:, :] *= self.u_upper
    #     return G, h

    def compute_Ab_sparse(self, F, f, x0):
        T, n_batch, n_state, n_tau = F.size()
        F = torch.cat([F, -torch.ones(T, n_batch, n_state, 1).to(F)], dim=-1)
        Av = F.transpose(0,1).contiguous().view(n_batch, -1) 
        Av = torch.cat([Av, torch.ones(n_batch, n_state).to(F)], dim=-1)
        b = -f.transpose(0,1).contiguous().view(n_batch, -1)
        b = torch.cat([b, x0], dim=-1)
        # ipdb.set_trace()
        return Av, b

    def compute_Qq_sparse(self, C, c):
        # ipdb.set_trace()
        T, n_batch, n_tau, n_tau = C.size()
        Qv = C.transpose(0,1).contiguous().diagonal(dim1=-2, dim2=-1).reshape(n_batch, -1)
        q = c.transpose(0,1).contiguous().reshape(n_batch, -1)
        return Qv, q

    def compute_Gh_sparse(self, x0):
        T, n_batch, n_state, n_ctrl = self.T, self.n_batch, self.n_state, self.n_ctrl
        n_tau = n_state + n_ctrl
        if self.u_upper is not None:
            Gv = torch.ones((n_batch, 2*T*n_ctrl)).to(x0)
            Gv[:, T*n_ctrl:] *= -1
            h = torch.ones((n_batch, 2*T*n_ctrl)).to(x0)*self.u_upper
            h[:, T*n_ctrl:] = -self.u_lower
        else:
            Gv = torch.ones((n_batch, n_ctrl)).to(x0)
            h = torch.ones((n_batch, n_ctrl)).to(x0)
        return Gv, h

    def compute_cost(self, xu, cost):
        C = cost.C.transpose(0,1)
        c = cost.c.transpose(0,1)
        return 0.5*((xu.unsqueeze(-1)*C).sum(dim=-2)*xu).sum(dim=-1).sum(dim=-1) + (xu*c).sum(dim=-1).sum(dim=-1)