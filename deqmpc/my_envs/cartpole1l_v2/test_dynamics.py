from torch.autograd import Function
import torch
import argparse
import time
import cartpole2l
import ipdb

nq = 2
nqdot = 2
ntau = 2
        
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=1000)
parser.add_argument('-r', '--runs', type=int, default=1000)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true', default=False)
parser.add_argument('-d', '--double', action='store_true', default=True)
options = parser.parse_args()
print(options)

device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

q_in = torch.randn((options.batch_size, nq), **kwargs)
qdot_in = torch.randn((options.batch_size, nqdot), **kwargs)
tau_in = torch.randn((options.batch_size, ntau), **kwargs)
h_in = torch.full((options.batch_size, nq), 0.1, **kwargs)

# q_in = torch.tensor([[1.1, 2, 3.], [1, 2, 3.]], **kwargs)
# qdot_in = torch.tensor([[1, 2, 3.], [1, 2, 3.]], **kwargs)
# tau_in = torch.tensor([[2.0, 0, 1.], [2, 0, 0.]], **kwargs)

# warm start the device
for i in range(10):
    q_out, qdot_out = cartpole2l.dynamics(q_in, qdot_in, tau_in, h_in)
    q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau = cartpole2l.derivatives(q_in, qdot_in, tau_in, h_in)

# benchmark computations
start_time = time.time()
for i in range(options.runs):
    q_out, qdot_out = cartpole2l.dynamics(q_in, qdot_in, tau_in, h_in)
    q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau = cartpole2l.derivatives(q_in, qdot_in, tau_in, h_in) 
end_time = time.time()
torch.cuda.synchronize()
print(end_time - start_time)

# print("q_in:", q_in)
# print("qdot_in:", qdot_in)
# print("tau_in:", tau_in)
# print("\n> q_out: ", q_out)
# print("> qdot_out: ", qdot_out)
# print("\n> q_jac_q: ", q_jac_q.mT)
# print("> q_jac_qdot: ", q_jac_qdot.mT)
# print("> q_jac_tau: ", q_jac_tau.mT)
# print("> qdot_jac_q: ", qdot_jac_q.mT)
# print("> qdot_jac_qdot: ", qdot_jac_qdot.mT)
# print("> qdot_jac_tau: ", qdot_jac_tau.mT)
