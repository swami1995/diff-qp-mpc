import torch
# import sympy, sympytorch
import numpy as np
# import scipy
# from scipy import linalg
# from sympy.physics import vector
# from sympy.physics import mechanics
import ipdb


def angle_normalize_2pi(x):
    return (((x) % (2*np.pi)))

class Spaces:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape
    
    def sample(self):
        return np.random.uniform(self.low, self.high)


###########################
# 1-link cartpole dynamics
###########################

class OneLinkCartpoleDynamics(torch.nn.Module):
    def __init__(self):
        '''
        Initializes the cartpole dynamics
        '''
        super().__init__()
        self.dt = 0.01
        self.max_force = 5.0        
        self.g = -9.81
        self.M = 0.5
        self.m = 0.2
        self.l = 0.3 
        self.n = 1
        self.nx = 2*self.n + 2
        self.nu = 1
        self.np = self.nx // 2

        # Generate the dynamics

    def forward(self, state, action):
        """
        Computes the next state given the current state and action
        """
        
        # semi-implicit euler integration
        k1 = self.dynamics(state, action)
        # import pdb; pdb.set_trace()
        k2 = self.dynamics(state + k1 * self.dt/2, action)
        k3 = self.dynamics(state + k2 * self.dt/2, action)
        k4 = self.dynamics(state + k3 * self.dt, action)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
    
    def dynamics(self, state, action):
        """
        Computes pendulum cont. dynamics with external torque input
        angle from downward, anti-clockwise is positive
        """

        M, m, l, g = self.M, self.m, self.l, self.g
        x, theta, dx, theta_dot = state[..., 0], state[..., 1], state[..., 2], state[..., 3]

        state_shape = state.shape
        u = action.squeeze(-1)

        x_ddot = (u + m * l * theta_dot**2 * torch.sin(theta) - m * g * torch.sin(theta) * torch.cos(theta)) / (M + m * torch.sin(theta)**2)
        theta_ddot = (-u * torch.cos(theta) - m * l * theta_dot**2 * torch.sin(theta) * torch.cos(theta) + (M + m) * g * torch.sin(theta)) / (l * (M + m * torch.sin(theta)**2))

        # ipdb.set_trace()
        
        derivatives = torch.stack((
            dx,
            theta_dot,
            x_ddot,
            theta_ddot
        ), dim=-1)

        # ipdb.set_trace()

        return derivatives.reshape(state_shape)
    
    def action_clip(self, action):
        return torch.clamp(action, -self.max_force, self.max_force)
    
    def state_clip(self, state):
        state[..., 1:self.np] = angle_normalize_2pi(state[..., 1:self.np])
        return state
    

class OneLinkCartpoleEnv:
    def __init__(self, bsz=1, max_steps=200, stabilization=False):
        self.dynamics = OneLinkCartpoleDynamics()
        self.spec_id = 'OneLinkCartpole-v0{}'.format('-stabilize' if stabilization else '')
        self.state = None  # Will be initialized in reset
        self.nx = self.dynamics.nx
        self.nu = self.dynamics.nu
        self.np = self.dynamics.np  
        self.max_force = self.dynamics.max_force
        self.dt = self.dynamics.dt
        self.T = max_steps
        self.bsz = bsz
        self.num_successes = 0
        self.goal = torch.Tensor([0.0, np.pi, 0.0, 0.0])
        # create observation space based on nx
        low = np.concatenate((np.full(self.np, -np.pi), np.full(self.np, -np.pi*5)))
        self.observation_space = Spaces(low, -low, (self.nx, 2))
        self.action_space = Spaces(-np.full(self.nu, self.max_force), np.full(self.nu, self.max_force), (self.nu, 2))   
        self.stabilization = stabilization

    def seed(self, seed):
        """
        Seeds the environment to produce deterministic results.
        Args:
            seed (int): The seed to use.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self, reset_idxs=None):
        """
        Resets the environment to an initial state, which is a random angle and angular velocity.
        Returns:
            numpy.ndarray: The initial state.
        """
        if reset_idxs is None:
            if self.stabilization:
                high = np.concatenate((np.full(self.np, 0.1), np.full(self.np, 0.1)))[None].repeat(self.bsz, axis=0)
                high[:,0], high[:,2] = 0.1, 0.1  # cart
                offset = torch.tensor([0, np.pi, 0, 0.0], dtype=torch.float32)[None, :]
                self.state = torch.tensor(np.random.uniform(low=-high, high=high), dtype=torch.float32) + offset
                self.state = self.state
            else:
                high = np.concatenate((np.full(self.np, np.pi), np.full(self.np, np.pi*5)))[None].repeat(self.bsz, axis=0)
                high[:,0], high[:,2] = 1.0, 1.0  # cart
                self.state = torch.tensor(np.random.uniform(low=-high, high=high), dtype=torch.float32)
            
            self.state = torch.Tensor([0.0, np.pi-0.1, 0.0, 0.0])[None].repeat(self.bsz, 1)
            
            self.num_successes = torch.zeros(self.bsz)
            self.num_steps = torch.zeros(self.bsz)
            return self.state.numpy()
        else:
            if self.stabilization:
                high = np.concatenate((np.full(self.np, 0.1), np.full(self.np, 0.1)))[None].repeat(len(reset_idxs), axis=0)
                high[:,0], high[:,2] = 0.1, 0.1  # cart
                offset = torch.tensor([0, np.pi, 0, 0.0], dtype=torch.float32)[None, :]
                self.state[reset_idxs] = torch.tensor(np.random.uniform(low=-high, high=high), dtype=torch.float32) + offset
            else:
                high = np.concatenate((np.full(self.np, np.pi), np.full(self.np, np.pi*5)))[None].repeat(len(reset_idxs), axis=0)
                high[:,0], high[:,2] = 1.0, 1.0  # cart
                self.state[reset_idxs] = torch.tensor(np.random.uniform(low=-high, high=high), dtype=torch.float32)

            self.state[reset_idxs] = torch.Tensor([0.0, np.pi-0.1, 0.0, 0.0])[None].repeat(len(reset_idxs), 1)
            self.num_successes[reset_idxs] = 0
            self.num_steps[reset_idxs] = 0
            return self.state[reset_idxs].numpy()

    def step(self, action):
        """
        Applies an action to the environment and steps it forward by one timestep.
        Args:
            action (float): The action to apply.
        Returns:
            tuple: A tuple containing the new state, reward, done flag, and info dict.
        """
        # action = torch.tensor([action], dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)[..., 0]
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
        x = self.state[..., 0]
        theta = self.state[..., 1]
        desired_theta = torch.pi
        success = torch.logical_and(torch.abs(theta - desired_theta) < 0.05, (torch.abs(x) < 0.05)).float()
        self.num_successes = (self.num_successes + 1)*success
        return torch.logical_and(self.num_successes >= 10, self.num_steps >= self.T)

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
        theta = self.state[..., 1]
        desired_theta = torch.pi
        rw = torch.abs(theta - desired_theta) + (torch.abs(x))
        return -rw

    def close(self):
        """
        Closes the environment.
        """
        pass


###########################
# 2-link cartpole dynamics
###########################
    
# https://openocl.github.io/tutorials/tutorial-01-modeling-double-cartpole/
    
class TwoLinkCartpoleDynamics(torch.nn.Module):
    def __init__(self):
        '''
        Initializes the cartpole dynamics
        '''
        super().__init__()
        self.dt = 0.05
        self.max_force = 5.0        
        self.g = 9.81
        self.M = 5.
        self.m1 = 1.
        self.m2 = 1.
        self.l1 = 1. 
        self.l2 = 1. 
        self.n = 2
        self.nx = 2*self.n + 2
        self.nu = 1
        self.np = self.nx // 2

        # Generate the dynamics

    def forward(self, state, action):
        """
        Computes the next state given the current state and action
        """
        
        # semi-implicit euler integration
        k1 = self.dynamics(state, action)
        # import pdb; pdb.set_trace()
        k2 = self.dynamics(state + k1 * self.dt/2, action)
        k3 = self.dynamics(state + k2 * self.dt/2, action)
        k4 = self.dynamics(state + k3 * self.dt, action)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
    
    def dynamics(self, state, action):
        """
        Computes pendulum cont. dynamics with external torque input
        angle from downward, anti-clockwise is positive
        """

        q_0, q_1, q_2, qdot_0, qdot_1, qdot_2 = state[..., 0], state[..., 1], state[..., 2], state[..., 3], state[..., 4], state[..., 5]
        f = -action.squeeze(-1)

        qddot_0 = (4.0*f*torch.cos(2.0*q_2)-6.0*f+
                    4.0*qdot_1**2*torch.cos(q_1)+
                    qdot_1**2*torch.cos(q_1-q_2)-
                    qdot_1**2*torch.cos(q_1+2.0*q_2)+
                    2.0*qdot_1*qdot_2*torch.cos(q_1-q_2)+
                    qdot_2**2*torch.cos(q_1-q_2)-
                    29.43*torch.sin(2.0*q_1)+
                    9.81*torch.sin(2.0*q_1+2.0*q_2)) / \
                    (3.0*torch.cos(2.0*q_1)-22.0*torch.cos(2.0*q_2)-
                    torch.cos(2.0*q_1+2.0*q_2)+34.0)
        qddot_1 = (-8.0*f*torch.sin(q_1)+4.0*f*torch.sin(q_1+2.0*q_2)+
                    3.0*qdot_1**2*torch.sin(2.0*q_1)+
                    23.0*qdot_1**2*torch.sin(q_2)+
                    22.0*qdot_1**2*torch.sin(2.0*q_2)+
                    qdot_1**2*torch.sin(2.0*q_1+q_2)+
                    46.0*qdot_1*qdot_2*torch.sin(q_2)+
                    2.0*qdot_1*qdot_2*torch.sin(2.0*q_1+q_2)+
                    23.0*qdot_2**2*torch.sin(q_2)+
                    qdot_2**2*torch.sin(2.0*q_1+q_2)-
                    490.5*torch.cos(q_1)+
                    215.82*torch.cos(q_1+2.0*q_2)) / \
                    (3.0*torch.cos(2.0*q_1)-
                    22.0*torch.cos(2.0*q_2)-
                    torch.cos(2.0*q_1+2.0*q_2)+34.0)
        qddot_2 = -((100.0*qdot_1**2*torch.sin(q_2)+
                    981.0*torch.cos(q_1+q_2)) *
                    (-(3.0*torch.sin(q_1)+
                        torch.sin(q_1+q_2))**2+
                    28.0*torch.cos(q_2)+42.0)+
                    0.5*(200.0*qdot_1*qdot_2*torch.sin(q_2)+
                        100.0*qdot_2**2*torch.sin(q_2)-
                        2943.0*torch.cos(q_1)-
                        981.0*torch.cos(q_1+q_2))*
                    (25.0*torch.cos(q_2)+3.0*torch.cos(2.0*q_1+q_2)+
                    torch.cos(2.0*q_1+2.0*q_2)+13.0)+
                    50.0*(2.0*torch.sin(q_1)+3.0*torch.sin(q_1-q_2)-
                    2.0*torch.sin(q_1+q_2)-torch.sin(q_1+2.0*q_2))*
                    (-2.0*f+3.0*qdot_1**2*torch.cos(q_1)+
                    qdot_1**2*torch.cos(q_1+q_2)+
                    2.0*qdot_1*qdot_2*torch.cos(q_1+q_2)+
                    qdot_2**2*torch.cos(q_1+q_2))) / \
                    (75.0*torch.cos(2.0*q_1)-550.0*torch.cos(2.0*q_2)-
                    25.0*torch.cos(2.0*q_1+2.0*q_2)+850.0)

        dq = torch.stack((qdot_0, qdot_1, qdot_2, qddot_0, qddot_1, qddot_2), dim=-1)
        return dq.reshape(state.shape)
    
    def action_clip(self, action):
        return torch.clamp(action, -self.max_force, self.max_force)
    
    def state_clip(self, state):
        state[..., 1:self.np] = angle_normalize_2pi(state[..., 1:self.np])
        return state

class TwoLinkCartpoleEnv:
    def __init__(self, n=1, stabilization=False):
        self.dynamics = TwoLinkCartpoleDynamics()
        self.spec_id = 'TwoLinkCartpole-v0{}'.format('-stabilize' if stabilization else '')
        self.state = None  # Will be initialized in reset
        self.nx = self.dynamics.nx
        self.nu = self.dynamics.nu
        self.np = self.dynamics.np  
        self.max_force = self.dynamics.max_force
        self.dt = self.dynamics.dt
        self.num_successes = 0
        # create observation space based on nx
        low = np.concatenate((np.full(self.np, -np.pi), np.full(self.np, -np.pi*5)))
        self.observation_space = Spaces(low, -low, (self.nx, 2))
        self.action_space = Spaces(-np.full(self.nu, self.max_force), np.full(self.nu, self.max_force), (self.nu, 2))   
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
            high = np.concatenate((np.full(self.np, 0.05), np.full(self.np, 0.05)))
            high[0], high[1] = 0.1, 0.1  # cart
            offset = torch.tensor([np.pi, 0.0]*self.np, dtype=torch.float32)
            offset[0], offset[1] = 0.0, 0.0  # cart
            self.state = torch.tensor(np.random.uniform(low=-high, high=high), dtype=torch.float32) + offset
        else:
            high = np.concatenate((np.full(self.np, np.pi), np.full(self.np, np.pi*5)))
            high[0], high[1] = 1.0, 1.0  # cart
            self.state = torch.tensor(np.random.uniform(low=-high, high=high), dtype=torch.float32)

        self.state = torch.Tensor([0.0, np.pi/2+0.01, 0.0, 0.0, 0.0, 0.0]) # fixed
        
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
        action = torch.tensor(action, dtype=torch.float32)[0]
        action = self.dynamics.action_clip(action)
        # ipdb.set_trace()
        self.state = self.dynamics(self.state, action)
        self.state = self.dynamics.state_clip(self.state)
        # ipdb.set_trace()
        done = self.is_done()
        reward = self.get_reward(action)  # Define your reward function based on the state and action
        return self.state.numpy(), reward, done, {}

    def is_done(self):
        """
        Determines whether the episode is done (e.g., if the pendulum is upright).
        Returns:
            bool: True if the episode is finished, otherwise False.
        """
        x = self.state[..., 0]
        theta = self.state[..., 1:2]
        desired_theta = torch.Tensor([torch.pi/2, 0])
        success = (torch.norm(theta - desired_theta) < 0.05 and (torch.abs(x) < 0.05))
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
        desired_theta = torch.Tensor([np.pi/2, 0])
        rw = torch.norm(theta - desired_theta) + (torch.abs(x))
        return -rw

    def close(self):
        """
        Closes the environment.
        """
        pass



###########################
# General cartpole dynamics
###########################
    
# class CartpoleDynamics(torch.nn.Module):
#     def __init__(self, n=1):
#         '''
#         Initializes the cartpole dynamics
#         Args:
#             n (int): The number of links in the cartpole
#         '''
#         super().__init__()
#         self.dt = 0.05
#         self.max_force = 3.0        
#         self.g = 9.81
#         self.mass = 1.
#         self.m_top = 2.
#         self.l = 1. # length of the pole
#         self.radius = 0.5
#         self.inertia = 1.
#         self.n = n
#         self.nx = 2*n + 2
#         self.nu = 1
#         self.np = self.nx // 2

#         # Generate the dynamics
#         self.generate_dynamics()

#     def generate_dynamics(self):
#         n = self.n
#         g = sympy.symbols("g")
#         x = mechanics.dynamicsymbols("x")
#         xd = mechanics.dynamicsymbols("x", 1)
#         xdd = mechanics.dynamicsymbols("x", 2)
#         theta = mechanics.dynamicsymbols("theta:"+ str(n))
#         thetad = mechanics.dynamicsymbols("theta:" + str(n), 1)
#         thetadd = mechanics.dynamicsymbols("theta:" + str(n), 2)

#         lengths = sympy.symbols("l:" + str(n))
#         masses = sympy.symbols("m:" + str(n))
#         m_top = sympy.symbols("m_{top}")
#         radii = sympy.symbols("r:" + str(n))
#         inertia_vals = sympy.symbols("I:" + str(n))

#         values = {g: self.g, m_top: self.m_top}
#         o_point = {x: 0, xd: 0, xdd: 0}

#         for i in range(n):
#             o_point.update({theta[i]: np.pi, thetad[i]: 0, thetadd[i]: 0})
#             values.update({lengths[i]: self.l, masses[i]: self.mass})
#             values.update({radii[i]: self.radius, inertia_vals[i]: self.inertia})
#         lengthsum = sum([values[i] for i in lengths])

#         u = mechanics.dynamicsymbols("u")

#         T = []
#         U = []

#         N = mechanics.ReferenceFrame("N")
#         P = mechanics.Point("P")
#         P.set_vel(N, 0)

#         PIv = N.y * xd
#         PIp = N.y * x
#         PI = P.locatenew("PI",PIp)
#         PI.set_vel(N, PIv) 
#         top_pivot = mechanics.Particle('top_pivot', PI, m_top)
#         T.append(top_pivot.kinetic_energy(N))

#         pivot0_frame = mechanics.ReferenceFrame("pivot0_f")
#         pivot0_frame.orient(N, "Axis", [ theta[0], N.z])
#         pivot0_frame.set_ang_vel(N, ( thetad[0]* N.z))


#         pivot0 = PI.locatenew("pivot0", lengths[0] * pivot0_frame.x)
#         pivot0.v2pt_theory(PI, N, pivot0_frame)

#         com0 = PI.locatenew("com0", radii[0] * pivot0_frame.x)
#         com0.v2pt_theory(PI, N, pivot0_frame)

#         inertai_dyad = vector.outer(pivot0_frame.z, pivot0_frame.z) * inertia_vals[0]
#         body = mechanics.RigidBody("B", com0, pivot0_frame, masses[0], (inertai_dyad, com0))

#         U.append(com0.pos_from(P).dot(N.x) * masses[0] * -g)
#         T.append(body.kinetic_energy(N))

#         pivot_prev = pivot0

#         for i in range(1, n):
#             P_f = mechanics.ReferenceFrame("P_f")
#             P_f.orient(N, "Axis", [theta[i], N.z])
#             P_f.set_ang_vel(N, thetad[i] * N.z)

#             pivot = mechanics.Point("p")
#             pivot.set_pos(pivot_prev, lengths[i] * P_f.x)
#             pivot.v2pt_theory(pivot_prev, N, P_f)

#             com = mechanics.Point("com")
#             com.set_pos(pivot_prev, radii[i] * P_f.x)
#             com.v2pt_theory(pivot_prev, N, P_f)

#             inertai_dyad = vector.outer(P_f.z, P_f.z) * inertia_vals[i]
#             body = mechanics.RigidBody("B", com, P_f, masses[i], (inertai_dyad, com))

#             U.append(com.pos_from(P).dot(N.x) * masses[i] * -g)
#             T.append(body.kinetic_energy(N))

#             pivot_prev = pivot

#         L = sum(T) - sum(U) + u * x

#         Lagrangian = mechanics.LagrangesMethod(L, [x] + theta)
#         Lagrangian.form_lagranges_equations()

#         M = Lagrangian.mass_matrix_full
#         K = Lagrangian.forcing_full
#         M = M.subs(values)
#         K = K.subs(values)

#         # analytical dynamics
#         # M_lambd, K_lambd = sympy.lambdify(tuple([x] + theta + [xd] + thetad + [u]), M), sympy.lambdify(tuple([x] + theta + [xd] + thetad + [u]), K)
#         print(M)
        
#         array2mat = [{'ImmutableDenseMatrix': torch.tensor}, 'torch']
#         ipdb.set_trace()
#         self.M_lambd = sympy.lambdify(tuple([x] + theta + [xd] + thetad + [u]), M)
#         self.K_lambd = sympy.lambdify(tuple([x] + theta + [xd] + thetad + [u]), K)
#         # self.M_lambd = sympytorch.SymPyModule(expressions=M)
#         # self.K_lambd = sympytorch.SymPyModule(expressions=K)

#         # analytical dynamics linearization
#         A, B, inp = Lagrangian.linearize(q_ind=[x]+theta, qd_ind=[xd]+thetad, A_and_B=True, op_point = o_point)

#         # print("A matrix free vars: {} ".format(A.free_symbols))
#         # print("B matrix free vars: {}".format(B.free_symbols))
#         # print("Input is {}".format(inp))

#         # A and B matrices with model parameters
#         A = A.subs(values)
#         B = B.subs(values)

#         # print("A matrix is: {} \n B matrix is: {}".format(A,B))

#         A = np.array(A,dtype=np.float64)
#         B = np.array(B,dtype=np.float64).T[0]

#         # analytical dynamics lambdified
#         A_lambd = sympy.lambdify(tuple([x] + theta + [xd] + thetad), A)

#     def forward(self, state, action):
#         """
#         Computes the next state given the current state and action
#         """
        
#         # semi-implicit euler integration
#         k1 = self.dynamics(state, action)
#         # import pdb; pdb.set_trace()
#         k2 = self.dynamics(state + k1 * self.dt/2, action)
#         k3 = self.dynamics(state + k2 * self.dt/2, action)
#         k4 = self.dynamics(state + k3 * self.dt, action)
#         return state + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
    
#     def dynamics(self, state, action):
#         """
#         Computes pendulum cont. dynamics with external torque input
#         angle from downward, anti-clockwise is positive
#         """
#         this_shape = state.shape
#         x = state[0]
#         u = action.squeeze(-1)[0]
#         # ipdb.set_trace()
#         dxdt = np.linalg.solve(self.M_lambd(*x, u), self.K_lambd(*x, u))
#         # ipdb.set_trace()
#         return (torch.Tensor(dxdt)).reshape(this_shape)   
    
#     def action_clip(self, action):
#         return torch.clamp(action, -self.max_force, self.max_force)
    
#     def state_clip(self, state):
#         state[..., 1:self.np] = angle_normalize_2pi(state[..., 1:self.np])
#         return state


# class CartpoleEnv:
#     def __init__(self, n=1, stabilization=False):
#         self.dynamics = CartpoleDynamics(n)
#         self.spec_id = 'Cartpole-v0{}'.format('-stabilize' if stabilization else '')
#         self.state = None  # Will be initialized in reset
#         self.nx = self.dynamics.nx
#         self.nu = self.dynamics.nu
#         self.np = self.dynamics.np  
#         self.max_force = self.dynamics.max_force
#         self.dt = self.dynamics.dt
#         self.num_successes = 0
#         # create observation space based on nx
#         low = np.concatenate((np.full(self.np, -np.pi), np.full(self.np, -np.pi*5)))
#         self.observation_space = Spaces(low, -low, (self.nx, 2))
#         self.action_space = Spaces(-np.full(self.nu, self.max_force), np.full(self.nu, self.max_force), (self.nu, 2))   
#         self.stabilization = stabilization

#     def seed(self, seed):
#         """
#         Seeds the environment to produce deterministic results.
#         Args:
#             seed (int): The seed to use.
#         """
#         np.random.seed(seed)
#         torch.manual_seed(seed)

#     def reset(self):
#         """
#         Resets the environment to an initial state, which is a random angle and angular velocity.
#         Returns:
#             numpy.ndarray: The initial state.
#         """
#         if self.stabilization:
#             high = np.concatenate((np.full(self.np, 0.1), np.full(self.np, 0.1)))
#             high[0], high[1] = 1.0, 1.0  # cart
#             offset = torch.tensor([np.pi, 0.0]*self.np, dtype=torch.float32)
#             offset[0], offset[1] = 0.0, 0.0  # cart
#             self.state = torch.tensor(np.random.uniform(low=-high, high=high), dtype=torch.float32) + offset
#         else:
#             high = np.concatenate((np.full(self.np, np.pi), np.full(self.np, np.pi*5)))
#             high[0], high[1] = 1.0, 1.0  # cart
#             self.state = torch.tensor(np.random.uniform(low=-high, high=high), dtype=torch.float32)
        
#         self.num_successes = 0
#         return self.state.numpy()

#     def step(self, action):
#         """
#         Applies an action to the environment and steps it forward by one timestep.
#         Args:
#             action (float): The action to apply.
#         Returns:
#             tuple: A tuple containing the new state, reward, done flag, and info dict.
#         """
#         # action = torch.tensor([action], dtype=torch.float32)
#         action = torch.tensor(action, dtype=torch.float32)
#         action = self.dynamics.action_clip(action)
#         self.state = self.dynamics(self.state, action)
#         self.state = self.dynamics.state_clip(self.state)
#         done = self.is_done()
#         reward = self.get_reward(action)  # Define your reward function based on the state and action
#         return self.state.numpy(), reward, done, {}

#     def is_done(self):
#         """
#         Determines whether the episode is done (e.g., if the pendulum is upright).
#         Returns:
#             bool: True if the episode is finished, otherwise False.
#         """
#         # Implement your logic for ending an episode, e.g., a time limit or reaching a goal state
#         # For demonstration, let's say an episode ends if the pendulum is upright within a small threshold
#         # ipdb.set_trace()
#         # theta, _ = self.state.unbind()
#         # theta, _ = self.state[0][0], self.state[0][1]
#         angles = self.state[..., 1:self.np]
#         desired_angles = torch.full(angles.shape, np.pi)
#         success = ((np.abs(angle_normalize_2pi(angles)) - desired_angles) ** 2).mean() < 0.01
#         self.num_successes = 0 if not success else self.num_successes + 1
#         return self.num_successes >= 10

#     def get_reward(self, action):
#         """
#         Calculates the reward for the current state and action.
#         Args:
#             action (float): The action taken.
#         Returns:
#             float: The calculated reward.
#         """
#         # Define your reward function; for simplicity, let's use the negative square of the angle
#         # as a reward, so the closer to upright (0 rad), the higher the reward.
#         # theta, _ = self.state.unbind()
#         # theta, _ = self.state[0][0], self.state[0][1]
#         angles = self.state[..., 1:self.np]
#         desired_angles = torch.full(angles.shape, np.pi)
#         return -float(((np.abs(angle_normalize_2pi(angles)) - desired_angles) ** 2).mean())

#     def close(self):
#         """
#         Closes the environment.
#         """
#         pass


