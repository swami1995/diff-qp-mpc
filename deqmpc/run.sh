#!/bin/bash
echo Starting training...

python train.py --deq --deq_iter 6 --bsz 200 --T 3 --name integrator --solver_type al --env integrator --qp_solve --test --pretrain --H 1

# python train.py --deq --deq_iter 6 --bsz 256 --T 5 --name integrator --solver_type al --env integrator --test --pretrain --qp_solve