#!/bin/bash

uv run torchrun --standalone --nproc_per_node=1 train.py \
--wandb-entity zhuoranjzhang \
--wandb-project tiny-lms \
--wandb-run-name doc-mask \
--intra-doc-mask \
--n-kv-heads 12 \
--n-layers 12 \
--val-every 250 \
--eval-every 500 \
--checkpoint-every 1000 \
--num-iterations 18865 \
--batch-size 44 \
--grad-accum-steps 11 \
--learning-rate 6e-4 \
--learning-rate-decay-frac 0.0 \
--weight-decay 0.1 \
--warmup-iters 700 \
--wandb