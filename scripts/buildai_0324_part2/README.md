BuildAI 0324 part2 partial-manifest helpers

Files:
- `generate_partial_pipeline_configs.py`
  Generates per-part pipeline configs and a shell script with one `run_dataset_pipeline.py` command per part.

Default assumptions:
- Source data is the 0324 route at `/share_data/guantianrui/datasets/Egocentric-100K/processed_0324_jpg`
- Partial manifests are named like `<prefix>.part0000.jsonl`
- Each part writes to its own dataset output directory, e.g. `.../BuildAI-100k-part2/part0000`
- Build exports depth directly with `build.export_depth: true`
- Build uses the 30fps-to-30fps path with `interpolate_labels: false`

Example:

```bash
python scripts/buildai_0324_part2/generate_partial_pipeline_configs.py \
  --manifest_prefix /share_data/guantianrui/dataset_pipeline_logs/partial/buildai_part2 \
  --config_dir configs/generated_buildai_0324_part2 \
  --part_count 8
```

This will generate:
- `configs/generated_buildai_0324_part2/dataset_pipeline_buildai_100k_0324_part2_part0000.yaml`
- ...
- `configs/generated_buildai_0324_part2/dataset_pipeline_buildai_100k_0324_part2_part0007.yaml`
- `configs/generated_buildai_0324_part2/run_partial_pipeline.sh`
