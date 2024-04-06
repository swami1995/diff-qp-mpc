#include "dynamics_gpu.h"
#include <iostream>

#define GPU_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

#define NUM_THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + NUM_THREADS - 1) / NUM_THREADS)

#define casadi_sq(x) (x * x)

// Cleaner wrapper for the generated dynamics
__device__ void _dynamics_gpu(const double *q_in, const double *qdot_in, const double *tau_in, const double *h_in,
                              double *q_out, double *qdot_out)
{
  const double *arg[4] = {q_in, qdot_in, tau_in, h_in};
  double *res[2] = {q_out, qdot_out};

  double a0, a1, a10, a11, a12, a13, a14, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][0] : 0;
  a1=arg[3]? arg[3][0] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a3=2.;
  a4=(a1/a3);
  a5=4.;
  a6=arg[2]? arg[2][0] : 0;
  a7=(a5*a6);
  a8=-2.;
  a9=9.8100000000000005e+00;
  a10=sin(a0);
  a10=(a9*a10);
  a10=(a8*a10);
  a7=(a7+a10);
  a10=(a4*a7);
  a10=(a2+a10);
  a10=(a3*a10);
  a10=(a2+a10);
  a11=(a1/a3);
  a12=(a5*a6);
  a13=(a1/a3);
  a13=(a13*a2);
  a13=(a0+a13);
  a13=sin(a13);
  a13=(a9*a13);
  a13=(a8*a13);
  a12=(a12+a13);
  a13=(a11*a12);
  a13=(a2+a13);
  a13=(a3*a13);
  a10=(a10+a13);
  a13=(a5*a6);
  a14=(a1/a3);
  a4=(a4*a7);
  a4=(a2+a4);
  a14=(a14*a4);
  a14=(a0+a14);
  a14=sin(a14);
  a14=(a9*a14);
  a14=(a8*a14);
  a13=(a13+a14);
  a14=(a1*a13);
  a14=(a2+a14);
  a10=(a10+a14);
  a14=6.;
  a10=(a10/a14);
  a10=(a1*a10);
  a10=(a0+a10);
  if (res[0]!=0) res[0][0]=a10;
  a10=(a3*a12);
  a7=(a7+a10);
  a3=(a3*a13);
  a7=(a7+a3);
  a5=(a5*a6);
  a11=(a11*a12);
  a11=(a2+a11);
  a11=(a1*a11);
  a0=(a0+a11);
  a0=sin(a0);
  a9=(a9*a0);
  a8=(a8*a9);
  a5=(a5+a8);
  a7=(a7+a5);
  a7=(a7/a14);
  a1=(a1*a7);
  a2=(a2+a1);
  if (res[1]!=0) res[1][0]=a2;
}

__device__ void _derivatives_gpu(const double *q_in, const double *qdot_in, const double *tau_in, const double *h_in,
                                 double *q_jac_qout, double *q_jac_qdotout, double *q_jac_uout,
                                 double *qdot_jac_qout, double *qdot_jac_qdotout, double *qdot_jac_tauout)
{
  const double *arg[4] = {q_in, qdot_in, tau_in, h_in};
  double *res[6] = {q_jac_qout, q_jac_qdotout, q_jac_uout, qdot_jac_qout, qdot_jac_qdotout, qdot_jac_tauout};

  double a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=1.;
  a1=arg[3]? arg[3][0] : 0;
  a2=1.6666666666666666e-01;
  a3=2.;
  a4=(a1/a3);
  a5=-2.;
  a6=9.8100000000000005e+00;
  a7=arg[0]? arg[0][0] : 0;
  a8=cos(a7);
  a8=(a6*a8);
  a8=(a5*a8);
  a9=(a4*a8);
  a9=(a3*a9);
  a10=(a1/a3);
  a11=(a1/a3);
  a12=arg[1]? arg[1][0] : 0;
  a13=(a11*a12);
  a13=(a7+a13);
  a14=cos(a13);
  a14=(a6*a14);
  a14=(a5*a14);
  a14=(a10*a14);
  a14=(a3*a14);
  a9=(a9+a14);
  a14=(a1/a3);
  a15=4.;
  a16=arg[2]? arg[2][0] : 0;
  a17=(a15*a16);
  a18=sin(a7);
  a18=(a6*a18);
  a18=(a5*a18);
  a17=(a17+a18);
  a17=(a4*a17);
  a17=(a12+a17);
  a17=(a14*a17);
  a17=(a7+a17);
  a18=cos(a17);
  a8=(a4*a8);
  a8=(a14*a8);
  a8=(a0+a8);
  a18=(a18*a8);
  a18=(a6*a18);
  a18=(a5*a18);
  a18=(a1*a18);
  a9=(a9+a18);
  a9=(a2*a9);
  a9=(a1*a9);
  a9=(a0+a9);
  if (res[0]!=0) res[0][0]=a9;
  a9=3.;
  a18=cos(a13);
  a18=(a18*a11);
  a18=(a6*a18);
  a18=(a5*a18);
  a18=(a10*a18);
  a18=(a0+a18);
  a18=(a3*a18);
  a9=(a9+a18);
  a18=cos(a17);
  a18=(a18*a14);
  a18=(a6*a18);
  a18=(a5*a18);
  a18=(a1*a18);
  a18=(a0+a18);
  a9=(a9+a18);
  a9=(a2*a9);
  a9=(a1*a9);
  if (res[1]!=0) res[1][0]=a9;
  a9=(a15*a4);
  a9=(a3*a9);
  a18=(a15*a10);
  a18=(a3*a18);
  a9=(a9+a18);
  a18=cos(a17);
  a8=(a15*a4);
  a8=(a14*a8);
  a18=(a18*a8);
  a18=(a6*a18);
  a18=(a5*a18);
  a18=(a15+a18);
  a18=(a1*a18);
  a9=(a9+a18);
  a9=(a2*a9);
  a9=(a1*a9);
  if (res[2]!=0) res[2][0]=a9;
  a9=cos(a7);
  a9=(a6*a9);
  a9=(a5*a9);
  a18=cos(a13);
  a18=(a6*a18);
  a18=(a5*a18);
  a8=(a3*a18);
  a8=(a9+a8);
  a19=cos(a17);
  a9=(a4*a9);
  a9=(a14*a9);
  a9=(a0+a9);
  a19=(a19*a9);
  a19=(a6*a19);
  a19=(a5*a19);
  a19=(a3*a19);
  a8=(a8+a19);
  a16=(a15*a16);
  a19=sin(a13);
  a19=(a6*a19);
  a19=(a5*a19);
  a16=(a16+a19);
  a16=(a10*a16);
  a12=(a12+a16);
  a12=(a1*a12);
  a7=(a7+a12);
  a12=cos(a7);
  a18=(a10*a18);
  a18=(a1*a18);
  a18=(a0+a18);
  a12=(a12*a18);
  a12=(a6*a12);
  a12=(a5*a12);
  a8=(a8+a12);
  a8=(a2*a8);
  a8=(a1*a8);
  if (res[3]!=0) res[3][0]=a8;
  a13=cos(a13);
  a13=(a13*a11);
  a13=(a6*a13);
  a13=(a5*a13);
  a11=(a3*a13);
  a8=cos(a17);
  a8=(a8*a14);
  a8=(a6*a8);
  a8=(a5*a8);
  a8=(a3*a8);
  a11=(a11+a8);
  a8=cos(a7);
  a13=(a10*a13);
  a13=(a0+a13);
  a13=(a1*a13);
  a8=(a8*a13);
  a8=(a6*a8);
  a8=(a5*a8);
  a11=(a11+a8);
  a11=(a2*a11);
  a11=(a1*a11);
  a0=(a0+a11);
  if (res[4]!=0) res[4][0]=a0;
  a0=12.;
  a17=cos(a17);
  a4=(a15*a4);
  a14=(a14*a4);
  a17=(a17*a14);
  a17=(a6*a17);
  a17=(a5*a17);
  a17=(a15+a17);
  a3=(a3*a17);
  a0=(a0+a3);
  a7=cos(a7);
  a10=(a15*a10);
  a10=(a1*a10);
  a7=(a7*a10);
  a6=(a6*a7);
  a5=(a5*a6);
  a15=(a15+a5);
  a0=(a0+a15);
  a2=(a2*a0);
  a1=(a1*a2);
  if (res[5]!=0) res[5][0]=a1;
}

// Multi-threaded GPU code
template <typename scalar_t>
__global__ void dynamics_kernel_gpu(const scalar_t *q_in_ptr, const scalar_t *qdot_in_ptr, const scalar_t *tau_in_ptr, const scalar_t *h_in_ptr,
                                    scalar_t *q_out_ptr, scalar_t *qdot_out_ptr,
                                    int q_size, int num_threads)
{
  GPU_1D_KERNEL_LOOP(b, num_threads)
  {
    _dynamics_gpu(q_in_ptr + b * q_size, qdot_in_ptr + b * q_size, tau_in_ptr + b * q_size, h_in_ptr + b,
                  q_out_ptr + b * q_size, qdot_out_ptr + b * q_size);
  }
}

template <typename scalar_t>
__global__ void derivatives_kernel_gpu(const scalar_t *q_in_ptr, const scalar_t *qdot_in_ptr, const scalar_t *tau_in_ptr, const scalar_t *h_in_ptr,
                                       scalar_t *q_jac_q_ptr, scalar_t *q_jac_qdot_ptr, scalar_t *q_jac_tau_ptr,
                                       scalar_t *qdot_jac_q_ptr, scalar_t *qdot_jac_qdot_ptr, scalar_t *qdot_jac_tau_ptr,
                                       int q_size, int num_threads)
{
  GPU_1D_KERNEL_LOOP(b, num_threads)
  {
    _derivatives_gpu(q_in_ptr + b * q_size, qdot_in_ptr + b * q_size, tau_in_ptr + b * q_size, h_in_ptr + b,
                     q_jac_q_ptr + b * q_size * q_size, q_jac_qdot_ptr + b * q_size * q_size, q_jac_tau_ptr + b * q_size * q_size,
                     qdot_jac_q_ptr + b * q_size * q_size, qdot_jac_qdot_ptr + b * q_size * q_size, qdot_jac_tau_ptr + b * q_size * q_size);
  }
}

// Torch CPU wrapper
std::vector<torch::Tensor> dynamics_gpu(torch::Tensor q_in, torch::Tensor qdot_in, torch::Tensor tau_in, torch::Tensor h_in)
{
  int batch_size = q_in.size(0);
  int q_size = q_in.size(1);
  torch::Tensor q_out = torch::zeros_like(q_in);
  torch::Tensor qdot_out = torch::zeros_like(q_in);
  using scalar_t = double;

  dynamics_kernel_gpu<scalar_t><<<NUM_BLOCKS(batch_size), NUM_THREADS>>>(
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

std::vector<torch::Tensor> derivatives_gpu(torch::Tensor q_in, torch::Tensor qdot_in, torch::Tensor tau_in, torch::Tensor h_in)
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

  derivatives_kernel_gpu<scalar_t><<<NUM_BLOCKS(batch_size), NUM_THREADS>>>(
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
