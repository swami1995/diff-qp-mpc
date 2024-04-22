import torch
import numpy as np
import ipdb

class PendulumDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dt = 0.05
        self.max_torque = 3.0        
        self.g = 10.
        self.m = 1.
        self.l = 1.
        self.nx = 2
        self.nu = 1

    def forward(self, state, action):
        """
        Computes the next state given the current state and action
        """
        state = self.semi_implicit_euler(state, action)
        return state
    
    def semi_implicit_euler(self, state, action):
        # semi-implicit euler integration
        thdot, thdotdot = self.dynamics(state, action)
        newthdot = thdot + thdotdot * self.dt
        newth = state[..., 0] + newthdot * self.dt

        # state = torch.stack((angle_normalize(newth), newthdot), dim=-1)
        state = torch.stack((newth, newthdot), dim=-1)
        return state

    def dynamics(self, state, action):
        """
        Computes pendulum cont. dynamics with external torque input
        theta is the angle from upright, anti-clockwise is positive
        """
        th = state[..., 0]
        thdot = state[..., 1]

        u = action.squeeze(-1)
        # u = torch.clamp(u, -self.max_torque, self.max_torque)

        newthdotdot = (u + self.m * self.g * self.l * torch.sin(th)) / (self.m * self.l ** 2)
        newthdot = thdot

        return newthdot, newthdotdot
    
    def action_clip(self, action):
        return torch.clamp(action, -self.max_torque, self.max_torque)
    
    def state_clip(self, state):
        state[..., 0] = angle_normalize(state[..., 0])
        return state
    
def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class Spaces:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape
    
    def sample(self):
        return np.random.uniform(self.low, self.high)

class PendulumDynamics_jac(PendulumDynamics):
    def __init__(self,):
        super(PendulumDynamics_jac, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.identity = torch.eye(self.nx).to(device)
    
    def forward(self, x, u):
        ## use vmap to compute jacobian using autograd.grad
        x = x.unsqueeze(-2).repeat(1, self.nx, 1)
        u = u.unsqueeze(-2).repeat(1, self.nx, 1)
        out_rk4 = self.semi_implicit_euler(x, u)
        out = out_rk4*self.identity[None]
        jac_out = torch.autograd.grad([out.sum()], [x, u])
        
        return out_rk4[:, 0], jac_out

class PendulumEnv:
    def __init__(self, stabilization=False):
        self.dynamics = PendulumDynamics()
        self.dynamics_derivatives = PendulumDynamics_jac()
        self.dynamics = torch.jit.script(self.dynamics)
        self.dynamics_derivatives = torch.jit.script(self.dynamics_derivatives)
        self.spec_id = 'Pendulum-v0{}'.format('-stabilize' if stabilization else '')
        self.state = None  # Will be initialized in reset
        self.nx = self.dynamics.nx
        self.nu = self.dynamics.nu
        self.max_torque = self.dynamics.max_torque
        self.dt = self.dynamics.dt
        self.num_successes = 0
        self.observation_space = Spaces(-np.array([np.pi, np.inf]), np.array([np.pi, np.inf]), (self.nx, 2)) # np.array([[-np.pi, np.pi], [-8, 8]])        
        self.action_space = Spaces(-np.array([self.max_torque]), np.array([self.max_torque]), (self.nu, 2)) #np.array([[-2, 2]])                
        self.stabilization = stabilization
        self.Qlqr = torch.Tensor([10.0, 1.00])
        self.Rlqr = torch.Tensor([0.01])

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
            high = np.array([0.05, 0.5])
        else:
            high = np.array([np.pi, 1])
        # self.state = torch.tensor(np.array([np.pi, 0]), dtype=torch.float32)
        self.state = torch.tensor(np.random.uniform(low=-high, high=high), dtype=torch.float32)
        self.num_successes = 0
        return self.state.numpy()

    def step(self, action):
        """
        Applies an action to the environment and steps it forward by one timestep.
        Args:
            action (float): The action to apply.
        Returns:
            tuple: A tuple containing the new state, reward, done flag, and info dict.
        """
        # action = torch.tensor([action], dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        action = self.dynamics.action_clip(action)
        self.state = self.dynamics(self.state, action)
        self.state = self.dynamics.state_clip(self.state)
        done = self.is_done()
        reward = self.get_reward(action)  # Define your reward function based on the state and action
        return self.state.numpy(), reward, done, {}

    def is_done(self):
        """
        Determines whether the episode is done (e.g., if the pendulum is upright).
        Returns:
            bool: True if the episode is finished, otherwise False.
        """
        # Implement your logic for ending an episode, e.g., a time limit or reaching a goal state
        # For demonstration, let's say an episode ends if the pendulum is upright within a small threshold
        # ipdb.set_trace()
        # theta, _ = self.state.unbind()
        # theta, _ = self.state[0][0], self.state[0][1]
        theta, _ = self.state[...,0], self.state[...,1]
        success = abs(angle_normalize(theta)) < 0.05
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
        theta, _ = self.state[...,0], self.state[...,1]
        return -float(angle_normalize(theta) ** 2)

    def close(self):
        """
        Closes the environment.
        """
        pass

class IntegratorDynamics(torch.nn.Module):
    def __init__(self, nx=2, nu=1, dt=0.1, max_acc=1, max_vel=1):
        super().__init__()
        self.dt = dt
        self.max_acc = max_acc     
        self.max_vel = max_vel
        self.nx = nx
        self.nu = nu
        self.np = int(self.nx / 2)


    def semi_implicit_euler(self, state, action):
        this_shape = state.shape
        pos = state[..., :self.np]
        vel = state[..., self.np:]
        # ipdb.set_trace()
        vel_n = vel + action * self.dt
        pos_n = pos + vel_n * self.dt
        state = torch.stack((pos_n, vel_n), dim=-1)
        return state.reshape(this_shape)

    def forward(self, state, action):
        """
        Computes the next state given the current state and action
        Args:
            state (torch.Tensor bsz x nx): The current state.
            action (torch.Tensor bsz x nu): The action to apply.
        Returns:
            torch.Tensor bsz x nx: The next state.
        """
        # semi-implicit euler integration
        return self.semi_implicit_euler(state, action)

    def action_clip(self, action):
        return torch.clamp(action, -self.max_acc, self.max_acc)


class IntegratorDynamics_jac(IntegratorDynamics):
    def __init__(self, nx=2, nu=1, dt=0.1, max_acc=1, max_vel=1):
        super(IntegratorDynamics_jac, self).__init__( nx, nu, dt, max_acc, max_vel)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.identity = torch.eye(self.nx).to(device)
    
    def forward(self, x, u):
        ## use vmap to compute jacobian using autograd.grad
        x = x.unsqueeze(-2).repeat(1, self.nx, 1)
        u = u.unsqueeze(-2).repeat(1, self.nx, 1)
        out_rk4 = self.semi_implicit_euler(x, u)
        out = out_rk4*self.identity[None]
        jac_out = torch.autograd.grad([out.sum()], [x, u])
        
        return out_rk4[:, 0], jac_out


class Spaces:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape
    
    def sample(self):
        return np.random.uniform(self.low, self.high)


class IntegratorEnv:
    def __init__(self, nx=2, nu=1, dt=0.1, max_acc=1, max_vel=1):
        self.dynamics = IntegratorDynamics(nx, nu, dt, max_acc, max_vel)
        self.dynamics_derivatives = IntegratorDynamics_jac(nx, nu, dt, max_acc, max_vel)
        self.dynamics = torch.jit.script(self.dynamics)
        self.dynamics_derivatives = torch.jit.script(self.dynamics_derivatives)
        self.spec_id = 'Integrator-v0'
        self.state = None  # Will be initialized in reset
        self.nx = self.dynamics.nx
        self.nu = self.dynamics.nu
        self.np = self.dynamics.np  
        self.max_acc = self.dynamics.max_acc
        self.max_vel = self.dynamics.max_vel
        self.dt = self.dynamics.dt
        self.num_successes = 0

        # create observation space based on nx
        low = np.concatenate((np.full(self.np, -np.inf), np.full(self.np, -self.max_vel)))
        self.observation_space = Spaces(low, -low, (self.nx, 2))
        self.action_space = Spaces(-np.full(self.nu, self.max_acc), np.full(self.nu, self.max_acc), (self.nu, 2))

        self.Qlqr = torch.Tensor([10.0, 1.00])
        self.Rlqr = torch.Tensor([0.00001])

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
        low = np.concatenate((np.full(self.np, -2.0), np.full(self.np, -self.max_vel)))
        # self.state = torch.tensor(np.array([2.0, 0]), dtype=torch.float32)
        self.state = torch.tensor(np.random.uniform(low=low, high=-low), dtype=torch.float32)
        self.num_successes = 0
        return self.state.numpy()

    def step(self, action):
        """
        Applies an action to the environment and steps it forward by one timestep.
        Args:
            action (float): The action to apply.
        Returns:
            tuple: A tuple containing the new state, reward, done flag, and info dict.
        """
        # action = torch.tensor([action], dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        action = self.dynamics.action_clip(action)
        self.state = self.dynamics(self.state, action)
        # self.state = self.dynamics.state_clip(self.state)
        done = self.is_done()
        reward = self.get_reward(action)  # Define your reward function based on the state and action
        return self.state.numpy(), reward, done, {}

    def is_done(self):
        """
        Determines whether the episode is done (e.g., if the pendulum is upright).
        Returns:
            bool: True if the episode is finished, otherwise False.
        """
        # Implement your logic for ending an episode, e.g., a time limit or reaching a goal state
        # For demonstration, let's say an episode ends if the pendulum is upright within a small threshold
        # ipdb.set_trace()
        # theta, _ = self.state.unbind()
        # theta, _ = self.state[0][0], self.state[0][1]
        pos = self.state[..., : self.np]
        success = torch.norm(pos) < 0.01
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
        pos, vel = self.state[..., : self.np], self.state[..., self.np:]
        reward = -torch.norm(pos) - torch.norm(vel) - torch.norm(action)
        return reward

    def close(self):
        """
        Closes the environment.
        """
        pass

class BatchPendulumEnv:
    def __init__(self, batch_size=1):
        self.dynamics = PendulumDynamics()
        self.batch_size = batch_size
        self.state = None  # Will be initialized in reset
        self.num_successes = torch.zeros(batch_size)

    def reset(self):
        """
        Resets the environment to an initial state for each batch, which is a random angle and angular velocity.
        Returns:
            torch.Tensor: The initial states for the batch.
        """
        high = np.array([np.pi, 1])
        self.state = torch.tensor(np.random.uniform(low=-high, high=high, size=(self.batch_size, 2)), dtype=torch.float32)
        return self.state

    def step(self, actions):
        """
        Applies actions to the environment and steps it forward by one timestep for each batch.
        Args:
            actions (torch.Tensor): The actions to apply for each batch.
        Returns:
            tuple: A tuple containing the new states, rewards, done flags.
        """
        self.state = self.dynamics(self.state, actions)
        done = self.is_done(self.state)
        reward = self.get_reward(actions)  # Batched reward calculation
        return self.state, reward, done

    def is_done(self, states):
        """
        Determines whether the episode is done for each batch.
        Args:
            states (torch.Tensor): The current states for each batch.
        Returns:
            torch.Tensor: A tensor of booleans indicating if the episode is finished for each batch.
        """
        theta = states[:, 0]

        success = (torch.abs(angle_normalize(theta)) < 0.05).float()
        self.num_successes = success*(self.num_successes + 1)
        return self.num_successes >= 5
        # return torch.abs(angle_normalize(theta)) < 0.1

    def get_reward(self, actions):
        """
        Calculates the reward for the current states and actions for each batch.
        Args:
            actions (torch.Tensor): The actions taken for each batch.
        Returns:
            torch.Tensor: The calculated rewards for each batch.
        """
        theta = self.state[:, 0]
        return -torch.pow(angle_normalize(theta), 2)
