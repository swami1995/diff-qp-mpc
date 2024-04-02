from torch.autograd import Function

import cartpole2l

    
class Cartpole2LFunction(Function):
    @staticmethod
    def forward(q_in, qdot_in, tau_in):
        qddot_out = cartpole2l.dynamics(q_in, qdot_in, tau_in)
        return qddot_out
    
    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        q_in, qdot_in, tau_in = inputs
        qddot_out = output
        ctx.save_for_backward(q_in, qdot_in, tau_in, qddot_out)
    
    @staticmethod
    def backward(ctx, qddot_out):
        return None

    @staticmethod
    def vmap(info, in_dims, q_in, qdot_in, tau_in):
        q_in_bdim, qdot_in_bdim, tau_in_bdim  = in_dims

        q_in = q_in.movedim(q_in_bdim, 0)
        qdot_in = qdot_in.movedim(qdot_in_bdim, 0)
        tau_in = tau_in.movedim(tau_in_bdim, 0)

        qddot_out = Cartpole2LFunction.apply(q_in, qdot_in, tau_in)

        return qddot_out, 0
        