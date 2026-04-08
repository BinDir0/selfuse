#!/usr/bin/env bash
python -m src.utils.convert_fsdp_checkpoint \
    --checkpoint /efs-exp/zengfanlian/Projects/EgoVLA/outputs/2026.04.06/22.21_legendvla_qwen3_vl_legendvla_qwen3_vl/step_checkpoints/update_step_100000/pytorch_model_fsdp_0  \
    --output /efs-exp/zengfanlian/Projects/EgoVLA/outputs/2026.04.06/22.21_legendvla_qwen3_vl_legendvla_qwen3_vl/step_checkpoints/update_step_100000/model.pt
