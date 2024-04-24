#include "dynamics_cpu.h"

#include <iostream>
#include <generated_dynamics.h>
#include <generated_derivatives.h>

// Cleaner wrapper for the generated dynamics
void _dynamics_cpu(const double *q_in, const double *qdot_in, const double *tau_in, const double *h_in,
                   double *q_out, double *qdot_out)
{
    const double *arg[4] = {q_in, qdot_in, tau_in, h_in};
    double *res[2] = {q_out, qdot_out};
    long long int iw[0];
    double w[0];
    eval_forward_dynamics(arg, res, iw, w, 0);
}

void _derivatives_cpu(const double *q_in, const double *qdot_in, const double *tau_in, const double *h_in,
                      double *q_jac_qout, double *q_jac_qdotout, double *q_jac_uout,
                      double *qdot_jac_qout, double *qdot_jac_qdotout, double *qdot_jac_tauout)
{
    const double *args[4] = {q_in, qdot_in, tau_in, h_in};
    double *res[6] = {q_jac_qout, q_jac_qdotout, q_jac_uout, qdot_jac_qout, qdot_jac_qdotout, qdot_jac_tauout};
    long long int iw[0];
    double w[0];
    eval_forward_derivatives(args, res, iw, w, 0);
}

// Multi-threaded CPU code
template <typename scalar_t>
void dynamics_kernel_cpu(const scalar_t *q_in_ptr, const scalar_t *qdot_in_ptr, const scalar_t *tau_in_ptr, const scalar_t *h_in_ptr,
                         scalar_t *q_out_ptr, scalar_t *qdot_out_ptr, 
                         int q_size, int batch_size)
{
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end)
                     {
        for (int64_t b=start; b<end; b++) {
            _dynamics_cpu(q_in_ptr + b*q_size, qdot_in_ptr + b*q_size, tau_in_ptr + b*q_size, h_in_ptr + b, 
                          q_out_ptr + b*q_size, qdot_out_ptr + b*q_size);
        } });
}

template <typename scalar_t>
void derivatives_kernel_cpu(const scalar_t *q_in_ptr, const scalar_t *qdot_in_ptr, const scalar_t *tau_in_ptr, const scalar_t *h_in_ptr,
                            scalar_t *q_jac_q_ptr, scalar_t *q_jac_qdot_ptr, scalar_t *q_jac_tau_ptr, 
                            scalar_t *qdot_jac_q_ptr, scalar_t *qdot_jac_qdot_ptr, scalar_t *qdot_jac_tau_ptr, 
                            int q_size, int batch_size)
{
    at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end)
                     {
        for (int64_t b=start; b<end; b++) {
            _derivatives_cpu(q_in_ptr + b*q_size, qdot_in_ptr + b*q_size, tau_in_ptr + b*q_size, h_in_ptr + b,
                             q_jac_q_ptr + b*q_size*q_size, q_jac_qdot_ptr + b*q_size*q_size, q_jac_tau_ptr + b*q_size*q_size,
                             qdot_jac_q_ptr + b*q_size*q_size, qdot_jac_qdot_ptr + b*q_size*q_size, qdot_jac_tau_ptr + b*q_size*q_size);
        } });
}

// Torch CPU wrapper
std::vector<torch::Tensor> dynamics_cpu(torch::Tensor q_in, torch::Tensor qdot_in, torch::Tensor tau_in, torch::Tensor h_in)
{
    int batch_size = q_in.size(0);
    int q_size = q_in.size(1);
    torch::Tensor q_out = torch::zeros_like(q_in);
    torch::Tensor qdot_out = torch::zeros_like(q_in);
    using scalar_t = double;

    dynamics_kernel_cpu<scalar_t>(
        q_in.data_ptr<scalar_t>(),
        qdot_in.data_ptr<scalar_t>(),
        tau_in.data_ptr<scalar_t>(),
        h_in.data_ptr<scalar_t>(),
        q_out.data_ptr<scalar_t>(),
        qdot_out.data_ptr<scalar_t>(),
        q_size,
        batch_size);

    return {q_out, qdot_out};
}

// Here just return Jacobians, Jacobian-vector product in PyTorch later
std::vector<torch::Tensor> derivatives_cpu(torch::Tensor q_in, torch::Tensor qdot_in, torch::Tensor tau_in, torch::Tensor h_in)
{
    int batch_size = q_in.size(0);
    int q_size = q_in.size(1);
    torch::Tensor q_jac_q = torch::zeros({batch_size, q_size, q_size}, q_in.options());
    torch::Tensor q_jac_qdot = torch::zeros({batch_size, q_size, q_size}, q_in.options());
    torch::Tensor q_jac_tau = torch::zeros({batch_size, q_size, q_size}, q_in.options());
    torch::Tensor qdot_jac_q = torch::zeros({batch_size, q_size, q_size}, q_in.options());
    torch::Tensor qdot_jac_qdot = torch::zeros({batch_size, q_size, q_size}, q_in.options());
    torch::Tensor qdot_jac_tau = torch::zeros({batch_size, q_size, q_size}, q_in.options());
    using scalar_t = double;

    derivatives_kernel_cpu<scalar_t>(
        q_in.data_ptr<scalar_t>(),
        qdot_in.data_ptr<scalar_t>(),
        tau_in.data_ptr<scalar_t>(),
        h_in.data_ptr<scalar_t>(),
        q_jac_q.data_ptr<scalar_t>(),
        q_jac_qdot.data_ptr<scalar_t>(),
        q_jac_tau.data_ptr<scalar_t>(),
        qdot_jac_q.data_ptr<scalar_t>(),
        qdot_jac_qdot.data_ptr<scalar_t>(),
        qdot_jac_tau.data_ptr<scalar_t>(),
        q_size,
        batch_size);

    return {q_jac_q, q_jac_qdot, q_jac_tau, qdot_jac_q, qdot_jac_qdot, qdot_jac_tau};
}