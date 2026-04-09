#!/usr/bin/env python3
"""
Script to compute normalizer statistics using the pi normalizer (NormStats / JSON).

Usage:
    python src/workspace/compute_norm_stats_pi.py \
        --config src/config/experiment/legendvla_qwen3_vl.yaml \
        --output_dir outputs/normalizer
"""

import argparse
import json
import os
import pathlib
from datetime import datetime

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.vla_dataset import VLALowLevelWdsDataset
from src.model.common.normalizer_pi import Normalizer, RunningStats, save as save_norm_stats

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "now", lambda fmt: datetime.now().strftime(fmt), replace=True
)


def compute_norm_stats(dataloader_cfg, normalizer_dataset, ignore_dim_slice=None):
    """Compute NormStats per key using RunningStats, then build a Normalizer.

    Args:
        dataloader_cfg: dict with batch_size, num_workers, etc.
        normalizer_dataset: IterableDataset yielding dicts of tensors/arrays.
        ignore_dim_slice: optional slice applied to every key via ignore_dim.

    Returns:
        (normalizer, metadata) tuple.
    """
    dataloader_cfg = dict(dataloader_cfg)
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    dataloader_cfg.setdefault("pin_memory", False)
    if num_workers > 0:
        dataloader_cfg.setdefault("persistent_workers", True)
        dataloader_cfg.setdefault("prefetch_factor", 4)

    dataloader = DataLoader(
        normalizer_dataset,
        collate_fn=normalizer_dataset.get_collator(),
        **dataloader_cfg,
    )
    dataloader_iter = iter(dataloader)
    try:
        first_batch = next(dataloader_iter)
    except StopIteration as exc:
        raise ValueError("No data to calculate normalizer") from exc

    # Determine keys (same logic as normalizer_utils.get_normalizer)
    normalizer_keys = [
        key for key, value in first_batch.items()
        if not key.startswith("_") and isinstance(value, (torch.Tensor, np.ndarray))
    ]
    if not normalizer_keys:
        raise ValueError("No tensor-like batch entries found for normalizer fitting")

    metadata = {
        "normalizer_keys": list(normalizer_keys),
        "current_frames_scanned": 0,
        "effective_rows": {key: 0 for key in normalizer_keys},
    }

    # Create one RunningStats per key
    running_stats = {key: RunningStats() for key in normalizer_keys}

    import itertools
    for batch in tqdm(
        itertools.chain([first_batch], dataloader_iter),
        desc="Calculating normalizer",
    ):
        batch_num_samples = batch.get("_batch_num_samples")
        if batch_num_samples is None:
            batch_num_samples = batch[normalizer_keys[0]].shape[0]
        metadata["current_frames_scanned"] += int(batch_num_samples)

        for key in normalizer_keys:
            value = batch[key]
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            value = value.reshape(-1, value.shape[-1])
            metadata["effective_rows"][key] += int(value.shape[0])
            running_stats[key].update(value)

    # Collect NormStats
    norm_stats = {key: rs.get_statistics() for key, rs in running_stats.items()}

    # Build Normalizer and apply ignore_dim
    normalizer = Normalizer(norm_stats)
    if ignore_dim_slice is not None:
        for key in normalizer_keys:
            normalizer.ignore_dim(key, ignore_dim_slice)

    return normalizer, metadata


def build_metadata(
    config_path, output_path, args,
    use_relative_action, history_pad_mode, future_pad_mode,
    selection_metadata, fit_metadata,
):
    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "config_path": str(config_path),
        "normalizer_path": str(output_path),
        "dataloader": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": False,
        },
        "sampling": {
            "mode": "val",
            "use_relative_action": bool(use_relative_action),
            "history_pad_mode": history_pad_mode,
            "future_pad_mode": future_pad_mode,
            "max_total_shards": args.max_total_shards,
            "min_shards_per_dataset": args.min_shards_per_dataset,
            "seed": args.seed,
        },
        "coverage": {
            "available_shards_total": selection_metadata["available_shards_total"],
            "selected_shards_total": selection_metadata["selected_shards_total"],
            "full_dataset_coverage": selection_metadata["full_dataset_coverage"],
            "current_frames_scanned": fit_metadata["current_frames_scanned"],
        },
        "scan_summary": fit_metadata,
        "datasets": selection_metadata["datasets"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalizer statistics (pi format) from WebDataset"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory to save norm_stats.json")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--max_total_shards", type=int, default=None)
    parser.add_argument("--min_shards_per_dataset", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from: {config_path}")
    raw_cfg = OmegaConf.load(config_path)
    if "defaults" in raw_cfg:
        config_dir = config_path.parent.parent.resolve()
        config_name = f"{config_path.parent.name}/{config_path.stem}"
        with hydra.initialize_config_dir(
            config_dir=str(config_dir), version_base=None
        ):
            cfg = hydra.compose(config_name=config_name)
    else:
        cfg = raw_cfg

    try:
        OmegaConf.resolve(cfg)
    except Exception as exc:
        print(f"Warning: Some config values could not be resolved: {exc}")
        print("Continuing with unresolved config...")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "norm_stats.json"
    metadata_path = output_dir / "norm_stats_metadata.json"

    print("\n" + "=" * 80)
    print("Computing Normalizer Statistics (pi format)")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path}")
    print("=" * 80)

    # 1. Create dataset
    print("\n1. Creating VLALowLevelWdsDataset...")
    vla_cfg = cfg.dataset.vla_dataset
    wds_datasets = OmegaConf.to_container(vla_cfg.wds_datasets, resolve=True)
    shape_meta_cfg = cfg.data.shape_meta if "data" in cfg and "shape_meta" in cfg.data else cfg.shape_meta
    shape_meta = OmegaConf.to_container(shape_meta_cfg, resolve=True)
    use_relative_action = vla_cfg.get("use_relative_action", False)
    history_pad_mode = vla_cfg.get("history_pad_mode", "repeat")
    future_pad_mode = vla_cfg.get("future_pad_mode", "repeat")

    normalizer_dataset = VLALowLevelWdsDataset(
        wds_datasets=wds_datasets,
        shape_meta=shape_meta,
        use_relative_action=use_relative_action,
        mode="val",
        max_total_shards=args.max_total_shards,
        min_shards_per_dataset=args.min_shards_per_dataset,
        seed=args.seed,
        history_pad_mode=history_pad_mode,
        future_pad_mode=future_pad_mode,
    )
    selection_metadata = normalizer_dataset.describe_shard_selection()
    print("   Dataset created successfully")
    print(f"   - selected_shards_total: {selection_metadata['selected_shards_total']}")
    print(f"   - available_shards_total: {selection_metadata['available_shards_total']}")
    print(f"   - full_dataset_coverage: {selection_metadata['full_dataset_coverage']}")

    # 2. Compute normalizer
    print("\n2. Computing normalizer statistics...")
    dataloader_cfg = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": False,
    }
    print(f"   - batch_size: {args.batch_size}")
    print(f"   - num_workers: {args.num_workers}")

    try:
        normalizer, fit_metadata = compute_norm_stats(
            dataloader_cfg,
            normalizer_dataset,
            ignore_dim_slice=slice(6, 18),
        )
        print("\n   Normalizer computed successfully")
        print(f"   - current_frames_scanned: {fit_metadata['current_frames_scanned']}")
        print(f"   - effective_rows: {fit_metadata['effective_rows']}")
    except Exception as exc:
        print(f"   Error computing normalizer: {exc}")
        import traceback
        traceback.print_exc()
        return

    # 3. Save norm_stats.json (includes ignored_dims)
    print(f"\n3. Saving norm_stats to {output_path}...")
    try:
        save_norm_stats(output_dir, normalizer.norm_stats)

        metadata = build_metadata(
            config_path=config_path,
            output_path=output_path,
            args=args,
            use_relative_action=use_relative_action,
            history_pad_mode=history_pad_mode,
            future_pad_mode=future_pad_mode,
            selection_metadata=selection_metadata,
            fit_metadata=fit_metadata,
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("   Saved successfully")
        file_size = os.path.getsize(output_path) / 1024
        print(f"   - norm_stats.json: {file_size:.1f} KB")
        print(f"   - metadata: {metadata_path}")
    except Exception as exc:
        print(f"   Error saving: {exc}")
        import traceback
        traceback.print_exc()
        return

    # 4. Summary
    print("\n" + "=" * 80)
    print("Normalizer Statistics Summary")
    print("=" * 80)
    for key, stats in normalizer.norm_stats.items():
        params = normalizer._params[key]
        print(f"\n{key}:")
        print(f"  Mean:  {stats.mean}")
        print(f"  Std:   {stats.std}")
        print(f"  q01:   {stats.q01}")
        print(f"  q99:   {stats.q99}")
        if stats.ignored_dims is not None:
            ignored = np.where(stats.ignored_dims.astype(bool))[0]
            print(f"  Ignored dims: {ignored.tolist()}")
        print(f"  Scale: {params['scale']}")
        print(f"  Offset:{params['offset']}")

    print("\n" + "=" * 80)
    print("Done!")
    print(f"Saved to: {output_path}")
    print(f"Load with: normalizer_pi.load('{output_dir}') -> Normalizer(stats)")
    print("=" * 80)


if __name__ == "__main__":
    main()
