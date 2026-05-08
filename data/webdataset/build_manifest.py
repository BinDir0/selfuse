"""
Build a deterministic shard-order manifest for each subset.

Run this ONCE per dataset key. The output JSON is the source of truth for
all subsequent sampling — sampling becomes a pure prefix-take over this
shuffled list, so it is reproducible across machines and across scales.

Input YAML matches the project's `src/config/dataset_paths/*.yaml` schema,
e.g. `src/config/dataset_paths/vla_wds.yaml`:

    wds_base_dir: /efs-exp/.../Webdataset
    vla_wds_datasets:
      - name: egoverse
        shard_urls: /efs-exp/.../egoverse/*.tar
        weight: 263                          # ignored here
      - name: vitra
        shard_urls:
          - ${wds_base_dir}/epic_train_rechunked/shard-*.tar
          - ${wds_base_dir}/ego4d_other_train_rechunked/shard-*.tar
        weight: 96

Each entry's `shard_urls` may be a single glob pattern or a list of glob
patterns; OmegaConf interpolations like ${wds_base_dir} are resolved.
The `weight` field (if present) is ignored — sampling is driven by a
uniform ratio per subset, not by manifest-time weights.

Usage:
    python build_manifest.py \\
        --config src/config/dataset_paths/vla_wds.yaml \\
        --datasets-key vla_wds_datasets \\
        --output manifests/vla_train.json \\
        --seed 42
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Reuse the project's own glob expansion so manifest-time shard resolution
# stays in sync with what the training pipeline sees at runtime.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.dataset.wds_dataset import expand_shard_patterns  # noqa: E402


def derive_seed(global_seed: int, subset_name: str) -> int:
    """Stable per-subset seed (hashlib, not Python's randomized hash())."""
    h = hashlib.sha256(f"{global_seed}:{subset_name}".encode()).hexdigest()
    return int(h[:16], 16)


def scan_subset(name: str, shard_urls) -> list[dict]:
    """Resolve glob patterns to concrete shard paths and stat their sizes."""
    shard_paths, _ = expand_shard_patterns(shard_urls)
    if not shard_paths:
        raise ValueError(f"Subset '{name}' has no shards matching {shard_urls!r}")

    return [
        {"path": str(Path(p).resolve()), "size": Path(p).stat().st_size}
        for p in shard_paths
    ]


def build_manifest(
    config_path: str,
    datasets_key: str,
    output_path: str,
    seed: int,
) -> None:
    cfg = OmegaConf.load(config_path)
    if datasets_key not in cfg:
        raise KeyError(
            f"Key '{datasets_key}' not found in {config_path}. "
            f"Available top-level keys: {list(cfg.keys())}"
        )

    datasets = OmegaConf.to_container(cfg[datasets_key], resolve=True)
    if not isinstance(datasets, list):
        raise TypeError(
            f"Expected '{datasets_key}' to be a list, got {type(datasets).__name__}"
        )

    manifest = {
        "seed": seed,
        "source_config": str(Path(config_path).resolve()),
        "datasets_key": datasets_key,
        # Snapshot of the resolved dataset list (interpolations expanded).
        # Self-contained record so downstream tools don't have to re-read the YAML.
        "datasets": datasets,
        "subsets": {},
    }

    for entry in datasets:
        name = entry["name"]
        shard_urls = entry["shard_urls"]

        shards = scan_subset(name, shard_urls)

        # Per-subset RNG: adding/removing one subset doesn't reshuffle others.
        subset_rng = random.Random(derive_seed(seed, name))
        subset_rng.shuffle(shards)

        total_size = sum(s["size"] for s in shards)
        manifest["subsets"][name] = {
            "num_shards": len(shards),
            "total_size_bytes": total_size,
            "shards": shards,
        }

        print(f"[{name}] {len(shards)} shards, total size {total_size / 1e9:.2f} GB")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to a project dataset YAML (e.g. src/config/dataset_paths/vla_wds.yaml)")
    parser.add_argument("--datasets-key", default="vla_wds_datasets",
                        help="Top-level key in the YAML that holds the dataset list")
    parser.add_argument("--output", required=True, help="Path to output manifest JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed for per-subset shuffle (default: 42)")
    args = parser.parse_args()

    build_manifest(args.config, args.datasets_key, args.output, args.seed)


if __name__ == "__main__":
    main()
