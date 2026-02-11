#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --num_machines 1 \
    --mixed_precision bf16 \
    inference_pretrain_legendvla.py