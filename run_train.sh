#!/bin/bash

uv run torchrun --standalone --nproc_per_node=1 train.py \
--wandb-entity zhuoranjzhang \
--wandb-project tiny-lms \
--wandb-run-name gqa-gpt2-size \
--wandb-log-interval 10 \
--val-every 250 \
--num-iterations 2500 \
--batch-size 64 \
--grad-accum-steps 8 \
--learning-rate 6e-4 \
--learning-rate-decay-frac 0.0 \
--weight-decay 0.1 \
--warmup-iters 700

