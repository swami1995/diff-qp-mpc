#!/bin/bash
echo Starting training...

# python train.py --deq --deq_iter 6 --bsz 200 --T 5 --name cartpole_detach_hdim128_lr1e3_lsx0clamp_gradnorm_Qold11_T5_hdim256_1k --solver_type al --env cartpole1link --lastqp_solve --pretrain --H 1 --hdim 256 --nq 2 --save

# python train.py --deq --deq_iter 6 --bsz 200 --T 3 --name integrator_detach_hdim128_lr1e3_lsx0clamp --solver_type al --env integrator --lastqp_solve --test --pretrain --H 1 --hdim 128 --save

# python train.py --deq --deq_iter 6 --bsz 256 --T 5 --name integrator --solver_type al --env integrator --test --pretrain --qp_solve

python train.py --deq --deq_iter 6 --bsz 200 --T 5 --solver_type al --env cartpole1link --nq 2 --qp_solve --test --pretrain --H 1 --hdim 256 --name cp1_hdim256_reg0.1_5k_detach 