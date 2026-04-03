#!/usr/bin/env python3
"""
Script to compute normalizer statistics from WebDataset.

Usage:
    python src/workspace/compute_norm_stats.py \
        --config src/config/experiment/legendvla_qwen3_vl.yaml \
        --output_dir outputs/normalizer
"""

import argparse
import json
import os
import pathlib
import pickle
from datetime import datetime

import hydra
from omegaconf import OmegaConf

from src.dataset.normalizer_utils import get_normalizer
from src.dataset.vla_dataset import VLALowLevelWdsDataset

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "now", lambda fmt: datetime.now().strftime(fmt), replace=True
)


def build_output_paths(output_dir: pathlib.Path, output_name: str):
    output_path = output_dir / output_name
    metadata_path = (
        output_path.with_suffix(".json")
        if output_path.suffix
        else pathlib.Path(f"{output_path}.json")
    )
    return output_path, metadata_path


def build_metadata(
    config_path,
    output_path,
    args,
    use_relative_action,
    relative_action_mode,
    history_pad_mode,
    future_pad_mode,
    selection_metadata,
    fit_metadata,
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
            "relative_action_mode": relative_action_mode,
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
        description="Compute normalizer statistics from WebDataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to training config file "
            "(e.g., src/config/experiment/legendvla_qwen3_vl.yaml)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save normalizer.pkl",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="normalizer.pkl",
        help="Output filename (default: normalizer.pkl)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help=(
            "Batch size for the normalizer dataloader. Default: 1024. "
            "For lowdim-only streaming on large multi-core machines, 512 and 1024 "
            "are usually the first two sweet-spot values worth benchmarking."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Number of dataloader workers (default: 64)",
    )
    parser.add_argument(
        "--max_total_shards",
        type=int,
        default=None,
        help=(
            "Optional cap on the total number of shards used for fitting. "
            "Use this together with --min_shards_per_dataset to keep shard-level "
            "coverage balanced while still favoring larger datasets."
        ),
    )
    parser.add_argument(
        "--min_shards_per_dataset",
        type=int,
        default=8,
        help=(
            "Minimum number of shards reserved for each dataset when the total shard "
            "budget allows it. Default: 8"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shard-order randomization (default: 0)",
    )
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
    output_path, metadata_path = build_output_paths(output_dir, args.output_name)

    print("\n" + "=" * 80)
    print("Computing Normalizer Statistics (WebDataset)")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path}")
    print("=" * 80)

    print("\n1. Creating VLALowLevelWdsDataset...")
    vla_cfg = cfg.dataset.vla_dataset
    wds_datasets = OmegaConf.to_container(vla_cfg.wds_datasets, resolve=True)
    shape_meta_cfg = cfg.data.shape_meta if "data" in cfg and "shape_meta" in cfg.data else cfg.shape_meta
    shape_meta = OmegaConf.to_container(shape_meta_cfg, resolve=True)
    use_relative_action = vla_cfg.get("use_relative_action", False)
    relative_action_mode = vla_cfg.get("relative_action_mode", "anchor")
    history_pad_mode = vla_cfg.get("history_pad_mode", "repeat")
    future_pad_mode = vla_cfg.get("future_pad_mode", "repeat")

    normalizer_dataset = VLALowLevelWdsDataset(
        wds_datasets=wds_datasets,
        shape_meta=shape_meta,
        use_relative_action=use_relative_action,
        relative_action_mode=relative_action_mode,
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
    if args.max_total_shards is not None:
        print(f"   - max_total_shards: {args.max_total_shards}")
        print(f"   - min_shards_per_dataset: {args.min_shards_per_dataset}")
        print(f"   - shard_seed: {args.seed}")

    print("\n2. Computing normalizer statistics...")
    dataloader_cfg = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": False,
    }
    print(f"   - batch_size: {args.batch_size}")
    print(f"   - num_workers: {args.num_workers}")
    print("   - pin_memory: False")

    try:
        normalizer, fit_metadata = get_normalizer(
            dataloader_cfg,
            normalizer_dataset,
            return_metadata=True,
        )
        print("\n   Normalizer computed successfully")
        print(f"   - current_frames_scanned: {fit_metadata['current_frames_scanned']}")
        print(f"   - effective_rows: {fit_metadata['effective_rows']}")
    except Exception as exc:
        print(f"   Error computing normalizer: {exc}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n3. Saving normalizer to {output_path}...")
    try:
        with open(output_path, "wb") as file_obj:
            pickle.dump(normalizer, file_obj)

        metadata = build_metadata(
            config_path=config_path,
            output_path=output_path,
            args=args,
            use_relative_action=use_relative_action,
            relative_action_mode=relative_action_mode,
            history_pad_mode=history_pad_mode,
            future_pad_mode=future_pad_mode,
            selection_metadata=selection_metadata,
            fit_metadata=fit_metadata,
        )
        with open(metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2, ensure_ascii=False)

        print("   Normalizer saved successfully")
        print(f"   Metadata saved successfully: {metadata_path}")

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   - File size: {file_size:.2f} MB")
    except Exception as exc:
        print(f"   Error saving normalizer: {exc}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("Normalizer Statistics Summary")
    print("=" * 80)
    for key in normalizer.params_dict.keys():
        params = normalizer.params_dict[key]
        stats = params["input_stats"]
        print(f"\n{key}:")
        print(f"  Mean: {stats.get('mean', 'N/A')}")
        print(f"  Std: {stats.get('std', 'N/A')}")
        print(f"  Min: {stats.get('min', 'N/A')}")
        print(f"  Max: {stats.get('max', 'N/A')}")
        print(f"  q01: {stats.get('q01', 'N/A')}")
        print(f"  q99: {stats.get('q99', 'N/A')}")
        print(f"  Scale: {params.get('scale', 'N/A')}")
        print(f"  Offset: {params.get('offset', 'N/A')}")

    print("\n" + "=" * 80)
    print("Normalizer computation completed!")
    print(f"Saved to: {output_path}")
    print(f"Metadata: {metadata_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
