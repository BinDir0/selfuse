python -m src.workspace.compute_norm_stats \
    --config src/config/experiment/legendvla_qwen3_vl.yaml \
    --output_dir outputs/normalizer/2026.04.02-relative-mixed-step \
    --num_workers 64 \
    --max_total_shards 10000 \