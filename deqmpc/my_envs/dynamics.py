import torch
from torch.autograd import Function
import ipdb

class DynamicsFunction(Function):
    @staticmethod
    def forward(q_in, qdot_in, tau_in, h_in, my_func):
        output = my_func(q_in, qdot_in, tau_in, h_in)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

class Dynamics(torch.nn.Module):
    def __init__(self, nx=None, dt=0.01, kwargs=None):
        super().__init__()
        assert nx is not None
        self.nx = nx  # number of states
        self.nu = 1  # number of inputs
        self.nq = nx // 2  # number of generalized coordinates
        self.dt = dt  # time step
        self.kwargs = kwargs  # the arguments

        # remember to use the package for each child environment

    def forward(self, state, action):
        """
        Computes the next state given the current state and action
        Args:
            state (torch.Tensor bsz x nx): The current state.
            action (torch.Tensor bsz x nu): The action to apply.
        Returns:
            torch.Tensor bsz x nx: The next state.
        """
        # check sizes
        # ipdb.set_trace()
        reshape_flag = False
        if state.dim() == 3:
            ipdb.set_trace()
            reshape_flag = True
            state = state.reshape(-1, self.nx)
            action = action.reshape(-1, self.nu)
        assert state.dim() == 2
        assert state.size(1) == self.nx
        assert action.dim() == 2
        assert action.size(1) == self.nu
        # TODO the package only supports double precision for now
        assert state.dtype == torch.float64

        bsz = state.size(0)

        # action only on the first joint
        tau = torch.zeros_like(state[:, : self.nq])
        tau[:, 0] = action[:, 0]
        q = state[:, : self.nq].contiguous()
        qdot = state[:, self.nq:].contiguous()
        h = torch.full((bsz, 1), self.dt, **self.kwargs)
        # ipdb.set_trace()
        next_state = DynamicsFunction.apply(
            q, qdot, tau, h, self.package.dynamics)
        next_state = torch.cat(next_state, dim=-1)
        if reshape_flag:
            return next_state.reshape(bsz, -1, self.nx)
        return next_state

    def derivatives(self, state, action):
        """
        Computes the derivatives of the dynamics with respect to the states and actions
        Args:
            state (torch.Tensor bsz x nx): The current state.
            action (torch.Tensor bsz x nu): The action to apply.
        Returns:
            Tuple of torch.Tensor: The derivatives of the dynamics with respect to the states and actions.
        """
        # check sizes
        reshape_flag = False
        if state.dim() == 3:
            reshape_flag = True
            state = state.reshape(-1, self.nx)
            action = action.reshape(-1, self.nu)
        assert state.dim() == 2
        assert state.size(1) == self.nx
        assert action.dim() == 2
        assert action.size(1) == self.nu
        assert state.dtype == torch.float64

        bsz = state.size(0)

        # action only on the first joint
        tau = torch.zeros((bsz, self.nq), **self.kwargs)
        tau[:, 0] = action.squeeze(1)
        q = state[:, : self.nq].contiguous()
        qdot = state[:, self.nq:].contiguous()
        h = torch.full((bsz, 1), self.dt, **self.kwargs)

        q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau = (
            DynamicsFunction.apply(q, qdot, tau, h, self.package.derivatives)
        )

        # concat jacobians to get dx_dx and dx_du
        q_jac_x = torch.cat((q_jac_q, q_jac_qdot), dim=-2)
        qdot_jac_x = torch.cat((qdot_jac_q, qdot_jac_qdot), dim=-2)
        x_jac_x = torch.cat((q_jac_x, qdot_jac_x), dim=-1)
        x_jac_u = torch.cat((q_jac_tau, qdot_jac_tau), dim=-1)[:, :1, :]
        if reshape_flag:
            return x_jac_x.reshape(bsz, -1, self.nx).tranpose(-1, -2), x_jac_u.reshape(bsz, -1, self.nu)
        return x_jac_x.transpose(-1, -2), x_jac_u.transpose(-1, -2)

    @torch.jit.script
    def _finite_diff_pre_processing(nq: int, q, qdot, tau, h, q_id, qdot_id, tau_id):
        # ipdb.set_trace()
        nqdot = nq
        ntau = nq
        nqt = nq + nqdot + ntau

        # parallelize the input
        q_plus = q[:, None] + q_id
        q_minus = q[:, None] - q_id
        q_zero = q[:, None].repeat(1, nqdot + ntau, 1)
        q_total = torch.cat((q_minus, q_zero, q_plus, q_zero), dim=1)
        qdot_plus = qdot[:, None] + qdot_id
        qdot_minus = qdot[:, None] - qdot_id
        qdot_zero_q = qdot[:, None].repeat(1, nq, 1)
        qdot_zero_tau = qdot[:, None].repeat(1, ntau, 1)
        qdot_total = torch.cat(
            (
                qdot_zero_q,
                qdot_minus,
                qdot_zero_tau,
                qdot_zero_q,
                qdot_plus,
                qdot_zero_tau,
            ),
            dim=1,
        )
        tau_plus = tau[:, None] + tau_id
        tau_minus = tau[:, None] - tau_id
        tau_zero = tau[:, None].repeat(1, nq + nqdot, 1)
        tau_total = torch.cat((tau_zero, tau_minus, tau_zero, tau_plus), dim=1)
        h_total = h[:, None].repeat(1, 2 * (nq + nqdot + ntau), 1)

        return (
            q_total.reshape(-1, nq),
            qdot_total.reshape(-1, nqdot),
            tau_total.reshape(-1, ntau),
            h_total.reshape(-1, nqt),
        )

    @torch.jit.script
    def _finite_diff_post_processing(nq: int, q_out, qdot_out, eps: float):
        nqdot = nq
        ntau = nq
        nqt = nq + nqdot + ntau

        q_out = q_out.reshape(-1, 2 * nqt, nq)
        qdot_out = qdot_out.reshape(-1, 2 * nqt, nqdot)

        # compute the jacobians
        q_jac_q = -(q_out[:, :nq] - q_out[:, nqt: nq + nqt]) / (2 * eps)
        q_jac_qdot = -(
            q_out[:, nq: nq + nqdot] - q_out[:, nqt + nq: nqt + nq + nqdot]
        ) / (2 * eps)
        q_jac_tau = -(q_out[:, nq + nqdot: nqt] - q_out[:, nqt + nq + nqdot:]) / (
            2 * eps
        )
        qdot_jac_q = -(qdot_out[:, :nq] -
                       qdot_out[:, nqt: nq + nqt]) / (2 * eps)
        qdot_jac_qdot = -(
            qdot_out[:, nq: nq + nqdot] -
            qdot_out[:, nqt + nq: nqt + nq + nqdot]
        ) / (2 * eps)
        qdot_jac_tau = -(
            qdot_out[:, nq + nqdot: nqt] - qdot_out[:, nqt + nq + nqdot:]
        ) / (2 * eps)

        return q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau

    def _finite_diff_derivatives(self, q, qdot, tau, h, eps=1e-8):
        nq = self.nq
        nqdot = nq
        ntau = nq
        bsz = q.size(0)

        # compute epsilon deltas
        q_id = eps * torch.eye(nq, **self.kwargs)[None].repeat(bsz, 1, 1)
        qdot_id = eps * torch.eye(nqdot, **self.kwargs)[None].repeat(bsz, 1, 1)
        tau_id = eps * torch.eye(ntau, **self.kwargs)[None].repeat(bsz, 1, 1)

        # parallelize the input
        q_total, qdot_total, tau_total, h_total = self._finite_diff_pre_processing(
            nq, q, qdot, tau, h, q_id, qdot_id, tau_id
        )

        # evaluate the function
        q_out, qdot_out = self.package.dynamics(
            q_total, qdot_total, tau_total, h_total)

        # compute the jacobians
        q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau = (
            self._finite_diff_post_processing(nq, q_out, qdot_out, eps)
        )

        return q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau

    def finite_diff_derivatives(self, state, action, eps=1e-8):
        """
        Computes the derivatives of the dynamics with respect to the states and actions using finite differences
        Args:
            state (torch.Tensor bsz x nx): The current state.
            action (torch.Tensor bsz x nu): The action to apply.
        Returns:
            Tuple of torch.Tensor: The derivatives of the dynamics with respect to the states and actions.
        """
        # check sizes
        reshape_flag = False
        if state.dim() == 3:
            reshape_flag = True
            state = state.reshape(-1, self.nx)
            action = action.reshape(-1, self.nu)
        assert state.dim() == 2
        assert state.size(1) == self.nx
        assert action.dim() == 2
        assert action.size(1) == self.nu
        assert state.dtype == torch.float64

        bsz = state.size(0)

        # action only on the first joint
        tau = torch.zeros((bsz, self.nq), **self.kwargs)
        tau[:, 0] = action.squeeze(1)
        q = state[:, : self.nq].contiguous()
        qdot = state[:, self.nq:].contiguous()
        h = torch.full((bsz, 1), self.dt, **self.kwargs)

        q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau = (
            self._finite_diff_derivatives(q, qdot, tau, h, eps)
        )

        # concat jacobians to get dx_dx and dx_du
        q_jac_x = torch.cat((q_jac_q, q_jac_qdot), dim=-2)
        qdot_jac_x = torch.cat((qdot_jac_q, qdot_jac_qdot), dim=-2)
        x_jac_x = torch.cat((q_jac_x, qdot_jac_x), dim=-1)
        x_jac_u = torch.cat((q_jac_tau, qdot_jac_tau), dim=-1)[:, :1, :]
        if reshape_flag:
            return x_jac_x.reshape(bsz, -1, self.nx).tranpose(-1, -2), x_jac_u.reshape(bsz, -1, self.nu)
        return x_jac_x.transpose(-1, -2), x_jac_u.transpose(-1, -2)

    def dynamics_derivatives(self, state, action):
        """
        Computes the dynamics and its derivatives with respect to the states and actions
        Args:
            state (torch.Tensor bsz x nx): The current state.
            action (torch.Tensor bsz x nu): The action to apply.
        Returns:
            Tuple of torch.Tensor: The next state and the derivatives of the dynamics with respect to the states and actions.
        """
        return self.forward(state, action), self.derivatives(state, action)
        # return self.forward(state, action), self.finite_diff_derivatives(state, action)

