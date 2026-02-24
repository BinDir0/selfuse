import argparse
import os
import zarr


def find_zarr_dirs(parent_dir):
    """Recursively find all .zarr directories under parent_dir."""
    zarr_dirs = []
    for root, dirs, files in os.walk(parent_dir):
        for d in dirs:
            if d.endswith(".zarr"):
                zarr_dirs.append(os.path.join(root, d))
    if parent_dir.endswith(".zarr"):
        zarr_dirs.append(parent_dir)
    return sorted(zarr_dirs)


def count_episodes(zarr_path):
    """Count episodes by reading meta/episode_ends array length."""
    try:
        root = zarr.open(zarr_path, mode="r")
        return len(root["meta/episode_ends"])
    except Exception as e:
        print(f"[WARN] Failed to read {zarr_path}: {e}")
        return -1


def main():
    parser = argparse.ArgumentParser(
        description="Filter zarr files by minimum episode count."
    )
    parser.add_argument("input_dir", help="Parent directory to search for .zarr files")
    parser.add_argument("output", help="Output txt file path")
    parser.add_argument(
        "--min_episodes", type=int, default=10,
        help="Minimum number of episodes (default: 10)"
    )
    args = parser.parse_args()

    zarr_dirs = find_zarr_dirs(args.input_dir)
    qualified = []
    insufficient = []

    for zp in zarr_dirs:
        n = count_episodes(zp)
        if n < 0:
            insufficient.append((zp, n))
        elif n >= args.min_episodes:
            qualified.append((zp, n))
        else:
            insufficient.append((zp, n))

    with open(args.output, "w") as f:
        f.write(f"# Qualified (>= {args.min_episodes} episodes): {len(qualified)}\n")
        for zp, n in qualified:
            f.write(f"{zp}\t{n}\n")
        f.write(f"\n# Insufficient (< {args.min_episodes} episodes): {len(insufficient)}\n")
        for zp, n in insufficient:
            f.write(f"{zp}\t{n}\n")

    print(f"Qualified: {len(qualified)}, Insufficient: {len(insufficient)}")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
