#!/bin/bash
echo Starting training...

python train.py --deq --deq_iter 6 --bsz 128 --T 5 --name integrator --solver_type al --env integrator --lastqp_solve --test --pretrain

# python train.py --deq --deq_iter 6 --bsz 256 --T 5 --name integrator --solver_type al --env integrator --qp_solve --test --pretrain