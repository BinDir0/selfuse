python -m src.workspace.compute_norm_stats \
    --config src/config/experiment/pretrain_legendvla_deepspeed.yaml \
    --output_dir outputs/normalizer/2026.03.16-relative-mixed \
    --num_workers 64 \
    --max_total_shards 5000 \