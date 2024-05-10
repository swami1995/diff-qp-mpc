#!/bin/bash
echo Starting training...

python train.py --deq --deq_iter 6 --bsz 200 --T 5 --name cartpole_detach_hdim128_lr1e3_lsx0clamp_gradnorm_Qold11_T5_hdim256_5k_hinge_gradcoeffinvres_noineq_nopretrain_deltascaled --solver_type al --env cartpole1link --qp_solve --pretrain --H 1 --hdim 256 --nq 2 --test --loss_type hinge --grad_coeff
#  --load --ckpt "deqmpc_cartpole_detach_hdim128_lr1e3_lsx0clamp_gradnorm_Qold11_T5_hdim256_5k_hinge_gradcoeffinvres_noineq_nopretrain_T5_bsz200_deq_iter6"

# python train.py --deq --deq_iter 6 --bsz 200 --T 3 --name integrator_detach_hdim128_lr1e3_lsx0clamp --solver_type al --env integrator --lastqp_solve --test --pretrain --H 1 --hdim 128 --save

# python train.py --deq --deq_iter 6 --bsz 256 --T 5 --name integrator --solver_type al --env integrator --test --pretrain --qp_solve