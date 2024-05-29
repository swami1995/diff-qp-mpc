import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipdb


# def animate_pendulum(env, theta, torque):
#     """
#     Animate the pendulum using matplotlib
#     Args:
#         env: pendulum environment
#         theta: list of pendulum angles
#         torque: list of pendulum torques
#     """

#     # Animation function
#     def update(frame):
#         ax.clear()
        
#         # Set up pendulum parameters
#         length = 1.0  # Length of the pendulum (meters)
        
#         # Calculate pendulum position
#         angle = theta[frame] 
        
#         # Plot pendulum
#         x = [0, -length * np.sin(angle)]
#         y = [0, length * np.cos(angle)]
#         ax.plot(x, y, marker='o', markersize=10, color='blue', linewidth=4)
#         # ax.arrow(0, -1, torque[frame]/env.max_torque, 0, color='green', width=0.05)
        
#         # Set plot limits
#         ax.set_xlim(-length*1.5, length*1.5)
#         ax.set_ylim(-length*1.5, length*1.5)
        
#         # Set plot aspect ratio to be equal
#         ax.set_aspect('equal')

#     # Set up the plot
#     fig, ax = plt.subplots()
#     ani = FuncAnimation(fig, update, frames=len(theta), interval=30, repeat=True)

#     plt.title('Simple Pendulum Animation')
#     plt.xlabel('X Position (m)')
#     plt.ylabel('Y Position (m)')

#     # Display the animation
#     plt.show()

def animate_pendulum(X, nq=1):
    # get a random starting state between min state and max state
    params = {"r_1": 1.0, "r_2": 1.0, "nq": nq}


    x0 = X[:nq,1];  # first state at k = 1

    # this function is defined below
    # ipdb.set_trace()
    [p_c, p_1, p_2] = dpc_endpositions_pendulum(tuple(x0), params)
    

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_title('Cartpole {}-Link Animation'.format(nq-1))
    # plt.show()
    # plt.draw()
    # plt.pause(0.5)

    # timer_handle = plt.text(-0.3, x_max[0], '0.00 s', fontsize=15);
    cart_handle, = plt.plot(p_c[0], p_c[1], 'ks', markersize=20, linewidth=3);
    pole_one_handle, = plt.plot([p_c[0], p_1[0]], [p_c[1], p_1[1]], color=np.array([38,124,185])/255, linewidth=8);
    pole_two_handle, = plt.plot([p_1[0], p_2[0]], [p_1[1], p_2[1]], color=np.array([38,124,185])/255, linewidth=8);

    joint_zero_handle, = plt.plot(p_c[0], p_c[1], 'ko', markersize=5)
    joint_one_handle, = plt.plot(p_1[0], p_1[1], 'ko', markersize=5)
    joint_two_handle, = plt.plot(p_2[0], p_2[1], 'ko', markersize=5)

    for k in range(0, X.shape[1]):
        tic = time.time()
        x = X[:nq,k]

        [p_c, p_1, p_2] = dpc_endpositions_pendulum(tuple(x), params)

        # timer_handle.set_text('{:.2f} s'.format(tdata[k]))

        cart_handle.set_data(x[0], 0)

        pole_one_handle.set_data([p_c[0], p_1[0]], [p_c[1], p_1[1]])
        pole_two_handle.set_data([p_1[0], p_2[0]], [p_1[1], p_2[1]])

        joint_zero_handle.set_data(p_c[0], p_c[1]);
        joint_one_handle.set_data(p_1[0], p_1[1]);
        joint_two_handle.set_data(p_2[0], p_2[1]);

        time.sleep(np.max([0.1, 0]))
        plt.pause(0.0001)
    plt.close(fig)

def dpc_endpositions_pendulum(q, p):
    # Returns the positions of cart, first joint, and second joint
    # to draw the black circles
    if p["nq"] == 1:
        q_1 = q[0]
        q_2 = 0.0
    elif p["nq"] == 2:
        q_1, q_2 = q
    else:
        raise NotImplementedError
    p_c = np.array([0.0, 0]);
    p_1 = p_c + p["r_1"] * np.array([-np.sin(q_1), np.cos(q_1)])
    p_2 = p_c + p["r_1"] * np.array([-np.sin(q_1), np.cos(q_1)]) + p["r_2"] * np.array([-np.sin(q_1+q_2), np.cos(q_1+q_2)])
    return p_c, p_1, p_2

def animate_integrator(env, pos, acc):
    # animate 2d position and acceleration as arrow
    def update(frame):
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.arrow(0, 0, pos[frame, 0], pos[frame, 1], color='blue', width=0.05)
        ax.arrow(pos[frame, 0], pos[frame, 1], acc[frame, 0], acc[frame, 1], color='green', width=0.05)
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update, frames=len(pos), interval=30, repeat=True)
    plt.title('Integrator Animation')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.show()
    

####################
import time    
def animate_cartpole(X, nq=2):
    # get a random starting state between min state and max state
    params = {"r_1": 1.0, "r_2": 1.0, "nq": nq}


    x0 = X[:nq,1];  # first state at k = 1

    # this function is defined below
    # ipdb.set_trace()
    [p_c, p_1, p_2] = dpc_endpositions(tuple(x0), params)
    

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_title('Cartpole {}-Link Animation'.format(nq-1))
    # plt.show()
    # plt.draw()
    # plt.pause(0.5)

    # timer_handle = plt.text(-0.3, x_max[0], '0.00 s', fontsize=15);
    cart_handle, = plt.plot(p_c[0], p_c[1], 'ks', markersize=20, linewidth=3);
    pole_one_handle, = plt.plot([p_c[0], p_1[0]], [p_c[1], p_1[1]], color=np.array([38,124,185])/255, linewidth=8);
    pole_two_handle, = plt.plot([p_1[0], p_2[0]], [p_1[1], p_2[1]], color=np.array([38,124,185])/255, linewidth=8);

    joint_zero_handle, = plt.plot(p_c[0], p_c[1], 'ko', markersize=5)
    joint_one_handle, = plt.plot(p_1[0], p_1[1], 'ko', markersize=5)
    joint_two_handle, = plt.plot(p_2[0], p_2[1], 'ko', markersize=5)

    for k in range(0, X.shape[1]):
        tic = time.time()
        x = X[:nq,k]

        [p_c, p_1, p_2] = dpc_endpositions(tuple(x), params)

        # timer_handle.set_text('{:.2f} s'.format(tdata[k]))

        cart_handle.set_data(x[0], 0)

        pole_one_handle.set_data([p_c[0], p_1[0]], [p_c[1], p_1[1]])
        pole_two_handle.set_data([p_1[0], p_2[0]], [p_1[1], p_2[1]])

        joint_zero_handle.set_data(p_c[0], p_c[1]);
        joint_one_handle.set_data(p_1[0], p_1[1]);
        joint_two_handle.set_data(p_2[0], p_2[1]);

        time.sleep(np.max([0.1, 0]))
        plt.pause(0.0001)
    plt.close(fig)

def dpc_endpositions(q, p):
    # Returns the positions of cart, first joint, and second joint
    # to draw the black circles
    if p["nq"] == 2:
        q_0, q_1 = q
        q_2 = 0.0
    elif p["nq"] == 3:
        q_0, q_1, q_2 = q
    else:
        raise NotImplementedError
    p_c = np.array([q_0, 0]);
    p_1 = p_c + p["r_1"] * np.array([-np.sin(q_1), np.cos(q_1)])
    p_2 = p_c + p["r_1"] * np.array([-np.sin(q_1), np.cos(q_1)]) + p["r_2"] * np.array([-np.sin(q_1+q_2), np.cos(q_1+q_2)])
    return p_c, p_1, p_2
 
# def dpc_endpositions(q_0, q_1, q_2, p):
#     # Returns the positions of cart, first joint, and second joint
#     # to draw the black circles
#     p_c = np.array([q_0, 0]);
#     p_1 = p_c + p["r_1"] * np.array([np.cos(q_1), np.sin(q_1)])
#     p_2 = p_c + p["r_1"] * np.array([np.cos(q_1), np.sin(q_1)]) + p["r_2"] * np.array([np.cos(q_1+q_2), np.sin(q_1+q_2)]);
#     return p_c, p_1, p_2


### Architecture utils
import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

def angle_normalize_2pi(x):
    return (((x) % (2*np.pi)))

class Spaces:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape
    
    def sample(self):
        return np.random.uniform(self.low, self.high)

device = None

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def unnormalize_states_pendulum(nominal_states):
    # ipdb.set_trace()
    # check theta of the first state in nominal_states[:, 0][0] and make sure all the nominal_states are in the same phase (i.e in terms of angle normalization)
    angle_0 = nominal_states[:, 0, 0]
    prev_angle = angle_0
    # ipdb.set_trace()
    for i in range(nominal_states.shape[1]):
        mask = torch.abs(nominal_states[:, i, 0] - prev_angle) > np.pi / 2
        mask_sign = torch.sign(nominal_states[:, i, 0])
        if mask.any():
            nominal_states[mask, i, 0] = (
                nominal_states[mask, i, 0] - mask_sign[mask] * 2 * np.pi
            )
        prev_angle = nominal_states[:, i, 0]
    return nominal_states


def unnormalize_states_cartpole_nlink(nominal_states):
    nq = nominal_states.shape[2] // 2 + 1
    angle_0 = nominal_states[:, 0, 1:nq]
    prev_angle = angle_0
    for i in range(nominal_states.shape[1]):
        mask = torch.abs(nominal_states[:, i, 1:nq] - prev_angle) > np.pi / 2
        mask_sign = torch.sign(nominal_states[:, i, 1:nq] - prev_angle)
        if mask.any():
            # ipdb.set_trace()
            nominal_states[:, i, 1:nq] = (
                (nominal_states[:, i, 1:nq] -
                 mask_sign * 2 * np.pi)*mask.float()
                + nominal_states[:, i, 1:nq]*(1-mask.float())
            )
        prev_angle = nominal_states[:, i, 1:nq]
    return nominal_states


def unnormalize_states_flyingcartpole(nominal_states):
    nq = nominal_states.shape[2] // 2
    angle_0 = nominal_states[:, 0, nq-1]
    prev_angle = angle_0
    for i in range(nominal_states.shape[1]):
        mask = torch.abs(nominal_states[:, i, nq-1] - prev_angle) > np.pi / 2
        mask_sign = torch.sign(nominal_states[:, i, nq-1] - prev_angle)
        if mask.any():
            nominal_states[:, i, nq-1] = (
                (nominal_states[:, i, nq-1] -
                 mask_sign * 2 * np.pi)*mask.float()
                + nominal_states[:, i, nq-1]*(1-mask.float())
            )
        prev_angle = nominal_states[:, i, nq-1]
        
    return nominal_states

