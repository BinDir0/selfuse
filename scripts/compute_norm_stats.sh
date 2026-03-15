python -m src.workspace.compute_norm_stats \
    --config src/config/experiment/pretrain_legendvla_deepspeed.yaml \
    --output_dir outputs/normalizer/2026.03.15-relative \
    --num_workers 120 \
    --max_total_shards 1000 \