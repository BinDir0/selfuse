#!/usr/bin/env python3
"""
Script to compute normalizer statistics from dataset.

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
OmegaConf.register_new_resolver("eval", eval, replace=True)

def main():
    parser = argparse.ArgumentParser(description='Compute normalizer statistics from dataset')
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
    args = parser.parse_args()
    
    # Load config file
    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
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
    print("Computing Normalizer Statistics")
    print("="*80)
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print("="*80)
    
    # Create VLA dataset
    print("\n1. Creating VLA dataset...")
    try:
        vla_dataset = hydra.utils.instantiate(cfg.dataset.vla_dataset)
        print(f"   ✓ VLA Dataset created successfully")
        print(f"   - Dataset length: {len(vla_dataset)}")
        print(f"   - Number of replay buffers: {len(vla_dataset.replay_buffers)}")
        print(f"   - Sampler lengths: {vla_dataset.sampler_lens}")
    except Exception as e:
        print(f"   ✗ Error creating VLA dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compute normalizer
    print("\n2. Computing normalizer statistics...")
    print(f"   Using dataloader config:")
    print(f"   - batch_size: {cfg.dataset.vla_dataset.normalizer_dataloader_cfg.batch_size}")
    print(f"   - num_workers: {cfg.dataset.vla_dataset.normalizer_dataloader_cfg.num_workers}")
    print(f"   - shuffle: {cfg.dataset.vla_dataset.normalizer_dataloader_cfg.shuffle}")
    
    try:
        normalizer = vla_dataset.get_normalizer()
        print(f"\n   ✓ Normalizer computed successfully")
    except Exception as e:
        print(f"   ✗ Error computing normalizer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save normalizer
    print(f"\n3. Saving normalizer to {output_path}...")
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(normalizer, f)
        print(f"   ✓ Normalizer saved successfully")
        
        # Print file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"   - File size: {file_size:.2f} MB")
    except Exception as e:
        print(f"   ✗ Error saving normalizer: {e}")
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

