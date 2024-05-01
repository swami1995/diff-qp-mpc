#!/bin/bash
echo Starting training...

python train.py --deq --deq_iter 6 --bsz 200 --T 3 --name cartpole_detach_hdim128_lr1e3_lsx0clamp --solver_type al --env cartpole1link --qp_solve --pretrain --H 1 --hdim 128 --nq 2

# python train.py --deq --deq_iter 6 --bsz 200 --T 3 --name integrator_detach_hdim128_lr1e3_lsx0clamp --solver_type al --env integrator --lastqp_solve --test --pretrain --H 1 --hdim 128 --save

# python train.py --deq --deq_iter 6 --bsz 256 --T 5 --name integrator --solver_type al --env integrator --test --pretrain --qp_solve