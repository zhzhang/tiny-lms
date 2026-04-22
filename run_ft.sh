#!/bin/bash
uv run ft.py \
  --model-name Qwen/Qwen3-0.6B \
  --dataset-name allenai/Dolci-Instruct-SFT \
  --max-seq-length 2048 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 32 \
  --lora-r 64 \
  --lora-alpha 128 \
  --weight-decay 0.01 \
  --learning-rate 2e-5 \
  --wandb-project ft \
  --wandb-run-name base-full
