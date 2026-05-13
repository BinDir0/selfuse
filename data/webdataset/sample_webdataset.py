"""
Sample a WebDataset subset by symlinking shards from a pre-built manifest.

Sampling is a deterministic prefix-take: for each subset in the manifest we
walk the shuffled shard list in order and accumulate until the per-subset
target size is reached. The last shard that crosses the threshold is
included (we err on the side of slightly more bytes than requested).

The same uniform `--ratio` is applied to every subset: each subset is
sampled to `ratio * subset_total_bytes`. Subsets are independent — there
is no global cross-subset weighting.

Usage:
    python sample_webdataset.py \\
        --manifest manifests/vla_train.json \\
        --ratio 0.1 \\
        --output-dir ./sample_10pct
"""

import argparse
import json
from pathlib import Path


def format_size(b: float) -> str:
    b = float(b)
    sign = "-" if b < 0 else ""
    b = abs(b)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024:
            return f"{sign}{b:.2f}{unit}"
        b /= 1024
    return f"{sign}{b:.2f}PB"


def select_shards(shards: list[dict], target_bytes: int) -> tuple[list[dict], int]:
    """
    Walk shards in manifest order; accumulate until we cross target_bytes.
    The crossing shard is included (prefer slightly more over slightly less).
    If shards run out before reaching target, return all of them.
    """
    selected: list[dict] = []
    total = 0
    for shard in shards:
        selected.append(shard)
        total += shard["size"]
        if total >= target_bytes:
            break
    return selected, total


def symlink_shards(selected: list[dict], subset_name: str, output_dir: Path) -> None:
    subset_dir = output_dir / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)
    for shard in selected:
        src = Path(shard["path"])
        dst = subset_dir / src.name
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            raise FileExistsError(
                f"Refusing to overwrite non-symlink at {dst}. "
                f"Pick a fresh --output-dir or remove it manually."
            )
        dst.symlink_to(src)


def sample(manifest_path: str, ratio: float, output_dir: str) -> None:
    if ratio <= 0:
        raise ValueError(f"--ratio must be > 0, got {ratio}")
    if ratio > 1.0:
        print(f"[warn] --ratio={ratio} > 1.0; each subset will use all shards (no oversampling here)")

    with open(manifest_path) as f:
        manifest = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_record = {
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_seed": manifest.get("seed"),
        "datasets_key": manifest.get("datasets_key"),
        "ratio": ratio,
        "subsets": {},
    }

    grand_target = 0
    grand_actual = 0
    print(f"Per-subset ratio: {ratio}\n")
    for subset_name, info in manifest["subsets"].items():
        avail = info["total_size_bytes"]
        shards = info["shards"]
        subset_target = int(avail * ratio)

        selected, actual = select_shards(shards, subset_target)
        symlink_shards(selected, subset_name, out_dir)

        sample_record["subsets"][subset_name] = {
            "ratio": ratio,
            "target_bytes": subset_target,
            "actual_bytes": actual,
            "available_bytes": avail,
            "num_shards_selected": len(selected),
            "num_shards_available": len(shards),
            "shard_names": [Path(s["path"]).name for s in selected],
        }

        grand_target += subset_target
        grand_actual += actual
        print(
            f"[{subset_name}] target {format_size(subset_target):>10}, "
            f"actual {format_size(actual):>10} "
            f"({len(selected)}/{len(shards)} shards)"
        )

    sample_record["target_total_bytes"] = grand_target
    sample_record["actual_total_bytes"] = grand_actual
    sample_record["target_total_human"] = format_size(grand_target)
    sample_record["actual_total_human"] = format_size(grand_actual)

    record_path = out_dir / "sample_manifest.json"
    with open(record_path, "w") as f:
        json.dump(sample_record, f, indent=2)

    print(
        f"\nGrand total: {format_size(grand_actual)} actual "
        f"vs {format_size(grand_target)} target "
        f"(diff {format_size(grand_actual - grand_target)})"
    )
    print(f"Output dir:      {out_dir}")
    print(f"Sample manifest: {record_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", required=True,
                        help="Path to manifest.json from build_manifest.py")
    parser.add_argument("--ratio", type=float, required=True,
                        help="Uniform per-subset sampling ratio in (0, 1]; each subset takes ratio * its_total_bytes")
    parser.add_argument("--output-dir", required=True, help="Where to put symlinked shards")
    args = parser.parse_args()

    sample(args.manifest, args.ratio, args.output_dir)


if __name__ == "__main__":
    main()
