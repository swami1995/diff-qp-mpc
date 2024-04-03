import torch
from torch.autograd import Function
import numpy as np
import ipdb
import time
import sys

import cartpole2l

sys.path.insert(0, "/home/khai/diff-qp-mpc/deqmpc")
from utils import *




class DynamicsFunction(Function):
    @staticmethod
    def forward(q_in, qdot_in, tau_in, h_in, my_func):
        output = my_func(q_in, qdot_in, tau_in, h_in)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def vmap(info, in_dims, q_in, qdot_in, tau_in, h_in, my_func):
        # ipdb.set_trace()
        q_in_bdim, qdot_in_bdim, tau_in_bdim, _, _ = in_dims
        if (q_in.dim() == 3):
            # repeat h to match shape of (q_in.shape[0], q_in.shape[1], 1)
            h_in = h_in.repeat(q_in.shape[0], 1, 1)

        q_in = q_in.movedim(q_in_bdim, 0)
        qdot_in = qdot_in.movedim(qdot_in_bdim, 0)
        tau_in = tau_in.movedim(tau_in_bdim, 0)

        next_state = DynamicsFunction.apply(
            q_in, qdot_in, tau_in, h_in, my_func)
        return next_state, 0


class CartpoleDynamics(torch.nn.Module):
    def __init__(self, nx=None, dt=0.01, package=None, kwargs=None):
        super().__init__()
        assert nx is not None
        assert package is not None
        self.nx = nx  # number of states
        self.nu = 1  # number of inputs
        self.nq = nx // 2  # number of generalized coordinates
        self.dt = dt  # time step
        self.package = package  # the generated package
        self.kwargs = kwargs  # the arguments

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
        # TODO the package only supports double pr  ecision for now
        assert state.dtype == torch.float64

        bsz = state.size(0)

        # action only on the first joint
        tau = torch.zeros_like(state[:, : self.nq])
        tau[:, 0] = action[:, 0]
        q = state[:, : self.nq].contiguous()
        qdot = state[:, self.nq:].contiguous()
        h = torch.full((bsz, 1), self.dt, **self.kwargs)

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
        q_jac_x = torch.cat((q_jac_q, q_jac_qdot), dim=1)
        qdot_jac_x = torch.cat((qdot_jac_q, qdot_jac_qdot), dim=1)
        x_jac_x = torch.cat((q_jac_x, qdot_jac_x), dim=2)
        x_jac_u = torch.cat((q_jac_tau, qdot_jac_tau), dim=1)[:, :, :1]
        if reshape_flag:
            return x_jac_x.reshape(bsz, -1, self.nx).tranpose(-1,-2), x_jac_u.reshape(bsz, -1, self.nu)
        return x_jac_x.transpose(-1, -2), x_jac_u

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

    def _finite_diff_derivatives(self, q, qdot, tau, h, eps=1e-8, kwargs=None):
        nq = self.nq
        nqdot = nq
        ntau = nq
        bsz = state.size(0)

        # compute epsilon deltas
        q_id = eps * torch.eye(nq, **kwargs)[None].repeat(bsz, 1, 1)
        qdot_id = eps * torch.eye(nqdot, **kwargs)[None].repeat(bsz, 1, 1)
        tau_id = eps * torch.eye(ntau, **kwargs)[None].repeat(bsz, 1, 1)

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

    def finite_diff_derivatives(self, state, action, eps=1e-8, kwargs=None):
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
        h = torch.full((bsz, 1), dt, **self.kwargs)

        q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau = (
            self._finite_diff_derivatives(q, qdot, tau, h, eps, kwargs)
        )

        # concat jacobians to get dx_dx and dx_du
        q_jac_x = torch.cat((q_jac_q, q_jac_qdot), dim=1)
        qdot_jac_x = torch.cat((qdot_jac_q, qdot_jac_qdot), dim=1)
        x_jac_x = torch.cat((q_jac_x, qdot_jac_x), dim=2)
        x_jac_u = torch.cat((q_jac_tau, qdot_jac_tau), dim=1)[:, :, 0]
        if reshape_flag:
            return x_jac_x.reshape(bsz, -1, self.nx).tranpose(-1,-2), x_jac_u.reshape(bsz, -1, self.nu)
        return x_jac_x.transpose(-1,-2), x_jac_u


class CartpoleEnv(torch.nn.Module):
    def __init__(self, nx=None, dt=0.01, stabilization=False, kwargs=None):
        super().__init__()
        assert nx is not None
        if nx == 6:
            self.package = cartpole2l
        else:
            raise NotImplementedError
        self.nx = nx
        self.dynamics = CartpoleDynamics(
            nx=nx, dt=dt, package=self.package, kwargs=kwargs
        )
        self.spec_id = "Cartpole{}l-v0{}".format(
            nx // 2 - 1, "-stabilize" if stabilization else ""
        )
        self.nq = self.dynamics.nq
        self.nu = self.dynamics.nu
        self.dt = dt
        self.kwargs = kwargs
        self.stabilization = stabilization
        self.num_successes = 0
        self.u_bounds = 10.0
        # create observation space based on nx, position and velocity
        high = np.concatenate(
            (np.full(self.nq, np.pi), np.full(self.nq, np.pi * 5)))
        self.observation_space = Spaces(-high, high, (self.nx, 2))
        self.action_space = Spaces(
            np.full(self.nu, -self.u_bounds),
            np.full(self.nu, self.u_bounds),
            (self.nu, 2),
        )
        self.stabilization = stabilization

    def action_clip(self, action):
        return torch.clamp(action, -self.u_bounds, self.u_bounds)

    def state_clip(self, state):
        state[..., 1: self.nq] = angle_normalize_2pi(state[..., 1: self.nq])
        return state

    def seed(self, seed):
        """
        Seeds the environment to produce deterministic results.
        Args:
            seed (int): The seed to use.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        """
        Resets the environment to an initial state, which is a random angle and angular velocity.
        Returns:
            numpy.ndarray: The initial state.
        """
        if self.stabilization:
            high = np.concatenate(
                (np.full(self.nq, 0.05), np.full(self.nq, 0.05)))
            high[0], high[1] = 0.1, 0.1  # cart
            offset = torch.tensor([np.pi, 0.0] * self.nq, **self.kwargs)
            offset[0], offset[1] = 0.0, 0.0  # cart
            self.state = (
                torch.tensor(
                    np.random.uniform(low=-high, high=high), **self.kwargs
                )
                + offset
            )
        else:
            high = np.concatenate(
                (np.full(self.nq, np.pi), np.full(self.nq, np.pi * 5))
            )
            high[0], high[1] = 1.0, 1.0  # cart
            self.state = torch.tensor(
                np.random.uniform(low=-high, high=high), **self.kwargs
            )

        self.state = torch.tensor(
            [0.0, np.pi / 2 + 0.01, 0.0, 0.0, 0.0, 0.0], **self.kwargs)  # fixed

        self.num_successes = 0
        return to_numpy(self.state)

    def step(self, action):
        """
        Applies an action to the environment and steps it forward by one timestep.
        Args:
            action (np array): The action to apply.
        Returns:
            tuple: A tuple containing the new state, reward, done flag, and info dict.
        """
        # action = torch.tensor([action], dtype=torch.float64)
        action = torch.tensor(action, **self.kwargs)
        action = self.action_clip(action)
        # ipdb.set_trace()
        self.state = self.dynamics(self.state, action)
        self.state = self.state_clip(self.state)
        # ipdb.set_trace()
        done = self.is_done()
        reward = self.get_reward(
            action
        )  # Define your reward function based on the state and action
        return to_numpy(self.state), reward, done, {}

    def is_done(self):
        """
        Determines whether the episode is done (e.g., if the pendulum is upright).
        Returns:
            bool: True if the episode is finished, otherwise False.
        """
        x = self.state[..., 0]
        theta = self.state[..., 1:2]
        desired_theta = torch.tensor([torch.pi, 0], **self.kwargs)
        success = torch.norm(
            theta - desired_theta) < 0.05 and (torch.abs(x) < 0.05)
        self.num_successes = 0 if not success else self.num_successes + 1
        return self.num_successes >= 10

    def get_reward(self, action):
        """
        Calculates the reward for the current state and action.
        Args:
            action (float): The action taken.
        Returns:
            float: The calculated reward.
        """
        # Define your reward function; for simplicity, let's use the negative square of the angle
        # as a reward, so the closer to upright (0 rad), the higher the reward.
        # theta, _ = self.state.unbind()
        # theta, _ = self.state[0][0], self.state[0][1]
        x = self.state[..., 0]
        theta = self.state[..., 1:2]
        desired_theta = torch.tensor([torch.pi, 0], **self.kwargs)
        rw = torch.norm(theta - desired_theta) + (torch.abs(x))
        return -rw

    def close(self):
        """
        Closes the environment.
        """
        pass


# if this is main then run the test
if __name__ == "__main__":
    # create the dynamics model
    kwargs = {
        "dtype": torch.float64,
        "device": torch.device("cuda"),
        "requires_grad": False,
    }
    nx = 6
    dt = 0.05
    package = cartpole2l
    dynamics = CartpoleDynamics(nx=nx, dt=dt, package=package, kwargs=kwargs)

    # create some random states and actions
    bsz = 1
    # state = torch.randn((bsz, nx), **kwargs)
    # action = torch.randn((bsz, 1), **kwargs)

    state = torch.tensor([[0.5, 0.5, 0.3, 0.7, 2.2, 1.0]], **kwargs)
    action = torch.tensor([[3.6]], **kwargs)

    next_state = dynamics(state, action)
    jacobians = dynamics.derivatives(state, action)
    jacobians_fd = dynamics.finite_diff_derivatives(
        state, action, eps=1e-5, kwargs=kwargs
    )
    next_state, jacobians = dynamics.dynamics_derivatives(state, action)

    print("next_state:", next_state)
    print("jacobians[0]:", jacobians[0])
    print("jacobians[1]:", jacobians[1])
    print("jacobians_fd[0]:", jacobians_fd[0])
    print("jacobians_fd[1]:", jacobians_fd[1])

    # calculate the error between jacobians and jacobians_fd
    error = np.zeros(2)
    for i in range(len(jacobians)):
        error[i] = torch.norm(jacobians[i] - jacobians_fd[i]) / torch.norm(jacobians[i])
    print("error:", error)

    # create the environment
    # env = CartpoleEnv(nx=nx, dt=dt, stabilization=False, kwargs=kwargs)
    # env.state = state
    # next_state2 = env.step(to_numpy(action))
    # print("next_state:", next_state2)

    #############################
    # Test vmap
    #############################
    # ls = 10
    # T = 5
    # bsz = 2
    # state = torch.randn((ls, bsz, T, nx), **kwargs)
    # action = torch.randn((ls, bsz, T, 1), **kwargs)
    # dx = dynamics

    # def merit(x, u): return dx(
    #     x[:, :-1].reshape(-1, nx), u[:, :-1].reshape(-1, 1)).view(bsz, T - 1, nx)
    
    # print("state:", merit(state, action).shape)
    # my_vmap = torch.vmap(merit)
    # next_state = my_vmap(state, action)
    # print("next_state:", next_state.shape)
