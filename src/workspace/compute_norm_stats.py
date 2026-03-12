#!/usr/bin/env python3
"""
Script to compute normalizer statistics from WebDataset.

Usage:
    python src/workspace/compute_norm_stats.py \
        --config src/config/experiment/pretrain_legendvla_deepspeed.yaml \
        --output_dir outputs/normalizer
"""

import argparse
import os
import pickle
import pathlib
from datetime import datetime

import hydra
from omegaconf import OmegaConf

from src.dataset.vla_dataset import VLALowLevelWdsDataset
from src.dataset.normalizer_utils import get_normalizer

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "now", lambda fmt: datetime.now().strftime(fmt), replace=True
)

def main():
    parser = argparse.ArgumentParser(description='Compute normalizer statistics from WebDataset')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config file (e.g., src/config/experiment/pretrain_legendvla_deepspeed.yaml)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory to save normalizer.pkl'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='normalizer.pkl',
        help='Output filename (default: normalizer.pkl)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4096,
        help='Batch size for normalizer dataloader (default: 4096)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=64,
        help='Number of dataloader workers (default: 64)'
    )
    args = parser.parse_args()

    # Load config file
    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from: {config_path}")
    raw_cfg = OmegaConf.load(config_path)
    if "defaults" in raw_cfg:
        # Compose with Hydra to resolve defaults like vla_wds_dataset_paths
        config_dir = config_path.parent.parent.resolve()  # .../src/config (absolute)
        config_name = f"{config_path.parent.name}/{config_path.stem}"
        with hydra.initialize_config_dir(
            config_dir=str(config_dir), version_base=None
        ):
            cfg = hydra.compose(config_name=config_name)
    else:
        cfg = raw_cfg

    # Resolve config
    try:
        OmegaConf.resolve(cfg)
    except Exception as e:
        print(f"Warning: Some config values could not be resolved: {e}")
        print("Continuing with unresolved config...")

    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    print("\n" + "="*80)
    print("Computing Normalizer Statistics (WebDataset)")
    print("="*80)
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print("="*80)

    # Create low-level WDS dataset for normalizer fitting
    print("\n1. Creating VLALowLevelWdsDataset...")
    vla_cfg = cfg.dataset.vla_dataset
    wds_datasets = OmegaConf.to_container(vla_cfg.wds_datasets, resolve=True)
    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    use_relative_action = vla_cfg.get("use_relative_action", False)

    normalizer_dataset = VLALowLevelWdsDataset(
        wds_datasets=wds_datasets,
        shape_meta=shape_meta,
        use_relative_action=use_relative_action,
        history_pad_mode=vla_cfg.get("history_pad_mode"),
        future_pad_mode=vla_cfg.get("future_pad_mode"),
    )
    print("   Dataset created successfully")

    # Compute normalizer
    print("\n2. Computing normalizer statistics...")
    dataloader_cfg = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    print(f"   - batch_size: {args.batch_size}")
    print(f"   - num_workers: {args.num_workers}")

    try:
        normalizer = get_normalizer(dataloader_cfg, normalizer_dataset)
        print(f"\n   Normalizer computed successfully")
    except Exception as e:
        print(f"   Error computing normalizer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save normalizer
    print(f"\n3. Saving normalizer to {output_path}...")
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(normalizer, f)
        print(f"   Normalizer saved successfully")

        # Print file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"   - File size: {file_size:.2f} MB")
    except Exception as e:
        print(f"   Error saving normalizer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print summary
    print("\n" + "="*80)
    print("Normalizer Statistics Summary")
    print("="*80)
    for key in normalizer.params_dict.keys():
        params = normalizer.params_dict[key]
        stats = params['input_stats']
        print(f"\n{key}:")
        print(f"  Mean: {stats.get('mean', 'N/A')}")
        print(f"  Std: {stats.get('std', 'N/A')}")
        print(f"  Min: {stats.get('min', 'N/A')}")
        print(f"  Max: {stats.get('max', 'N/A')}")
        print(f"  Scale: {params.get('scale', 'N/A')}")
        print(f"  Offset: {params.get('offset', 'N/A')}")

    print("\n" + "="*80)
    print("Normalizer computation completed!")
    print(f"Saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
