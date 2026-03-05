#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --num_machines 1 \
    --mixed_precision bf16 \
    inference_pretrain_legendvla.py