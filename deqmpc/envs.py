import torch
import numpy as np
import ipdb

class PendulumDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, state, action):
        th = state[..., 0]#.view(-1, 1)
        thdot = state[..., 1]#.view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = action.squeeze(-1)
        u = torch.clamp(u, -11, 11)

        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        # newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.stack((angle_normalize(newth), newthdot), dim=-1)
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
    


class PendulumEnv:
    def __init__(self, stabilization=False):
        self.dynamics = PendulumDynamics()
        self.spec_id = 'Pendulum-v0{}'.format('-stabilize' if stabilization else '')
        self.state = None  # Will be initialized in reset
        self.nx = 2
        self.nu = 1
        self.num_successes = 0
        self.observation_space = Spaces(-np.array([np.pi, np.inf]), np.array([np.pi, np.inf]), (self.nx, 2)) # np.array([[-np.pi, np.pi], [-8, 8]])
        self.max_torque = 11.0
        self.action_space = Spaces(-np.array([self.max_torque]), np.array([self.max_torque]), (self.nu, 2)) #np.array([[-2, 2]])        
        self.dt = 0.05
        self.stabilization = stabilization

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
        self.state = torch.tensor(np.random.uniform(low=high, high=high), dtype=torch.float32)
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
        self.state = self.dynamics(self.state, action)
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
        theta, _ = self.state[0][0], self.state[0][1]
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
        theta, _ = self.state[0][0], self.state[0][1]
        return -float(angle_normalize(theta) ** 2)

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
