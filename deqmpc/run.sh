#!/bin/bash
echo Starting training...

python train.py --deq --deq_iter 6 --bsz 200 --T 5 --name FlyingCartpole_hdim256_lr1e3_noineq --solver_type al --env FlyingCartpole --qp_solve --H 1 --hdim 256 --nq 7 --save --loss_type l1 --layer_type gcn

# python train.py --deq --deq_iter 6 --bsz 200 --T 5 --name cartpole_noisy_noineq_gcn_nodetach --solver_type al --env cartpole1link --H 5 --hdim 256 --nq 2 --qp_solve --test --loss_type l1 --layer_type gcn --data_noise_type 1

# python train.py --deq --deq_iter 6 --bsz 200 --T 5 --name cartpole_5k_noineq_conditer_L1simplefix_cond_expand4_gcn_nodetach_final --solver_type al --env cartpole1link --qp_solve --H 1 --hdim 256 --nq 2 --save --loss_type l1 --layer_type gcn

# python train.py --deq --deq_iter 6 --bsz 200 --T 5 --name cartpole_5k --solver_type al --env cartpole1link --qp_solve --H 1 --hdim 256 --nq 2 --save --loss_type l1 --layer_type gcn 

# python train.py --deq --deq_iter 6 --bsz 200 --T 5 --name quad_5k --solver_type al --env rexquadrotor --qp_solve --H 1 --hdim 256 --nq 7 --save --loss_type l1 --layer_type gcn

# --load --ckpt "deqmpc_cartpole_detach_hdim128_lr1e3_lsx0clamp_gradnorm_Qold11_T5_hdim256_5k_hinge_coefffixedinvres2Increments_noineq_conditer_DeltascaleupdateTxNdtfixed_T5_bsz200_deq_iter6"

# python train.py --deq --deq_iter 6 --bsz 200 --T 5 --name cartpole_detach_hdim128_lr1e3_lsx0clamp_gradnorm_Qold11_T5_hdim256_1k --solver_type al --env cartpole1link --lastqp_solve --pretrain --H 1 --hdim 256 --nq 2 --save

# python train.py --deq --deq_iter 6 --bsz 200 --T 3 --name integrator_detach_hdim128_lr1e3_lsx0clamp --solver_type al --env integrator --lastqp_solve --test --pretrain --H 1 --hdim 128 --save

# python train.py --deq --deq_iter 6 --bsz 256 --T 5 --name integrator --solver_type al --env integrator --test --pretrain --qp_solve

# python train.py --deq --deq_iter 6 --bsz 200 --T 5 --solver_type al --env cartpole1link --nq 2 --qp_solve --test --H 1 --hdim 256 --name cp1_hdim256 #--save