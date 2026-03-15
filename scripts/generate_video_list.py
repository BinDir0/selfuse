#!/usr/bin/env python3
"""
Generate video list from BuildAI-processed dataset structure.

The dataset is organized as:
/share_data/lvjianan/datasets/BuildAI-processed/factory_{id}/worker_{id}/processed/factory{id}_worker{id}_{id}_crop{id}.mp4

This script recursively finds all videos matching this pattern and outputs a list.
"""

import argparse
import sys
from pathlib import Path


def find_videos_fast(base_dir, factory_ids=None, worker_ids=None, sort=True):
    """
    Fast video finding using known directory structure.

    Structure: factory_{id}/worker_{id}/processed/*.mp4

    This is much faster than recursive glob because we only traverse 3 levels.

    Args:
        base_dir: Base directory to search
        factory_ids: Optional list of factory IDs to filter
        worker_ids: Optional list of worker IDs to filter
        sort: Whether to sort the results

    Returns:
        List of video file paths
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(1)

    if not base_path.is_dir():
        print(f"Error: Not a directory: {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Searching for videos in: {base_dir}", file=sys.stderr)

    videos = []
    factory_set = set(factory_ids) if factory_ids else None
    worker_set = set(worker_ids) if worker_ids else None

    # Level 1: Iterate through factory_* directories
    factory_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("factory_")]
    print(f"Found {len(factory_dirs)} factory directories", file=sys.stderr)

    for factory_dir in factory_dirs:
        # Extract factory ID
        try:
            factory_id = int(factory_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Filter by factory ID if specified
        if factory_set and factory_id not in factory_set:
            continue

        # Level 2: Iterate through worker_* directories
        worker_dirs = [d for d in factory_dir.iterdir() if d.is_dir() and d.name.startswith("worker_")]

        for worker_dir in worker_dirs:
            # Extract worker ID
            try:
                worker_id = int(worker_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            # Filter by worker ID if specified
            if worker_set and worker_id not in worker_set:
                continue

            # Level 3: Get videos from processed directory
            processed_dir = worker_dir / "processed"
            if processed_dir.exists() and processed_dir.is_dir():
                video_files = list(processed_dir.glob("*.mp4"))
                videos.extend(video_files)

    if sort:
        videos = sorted(videos)

    print(f"Found {len(videos)} videos", file=sys.stderr)

    return [str(v.resolve()) for v in videos]


def find_videos(base_dir, pattern="**/*.mp4", sort=True):
    """
    Find all video files in the base directory (slow recursive method).

    This is kept for backward compatibility but is much slower.
    Use find_videos_fast() instead for BuildAI dataset structure.

    Args:
        base_dir: Base directory to search
        pattern: Glob pattern for video files
        sort: Whether to sort the results

    Returns:
        List of video file paths
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(1)

    if not base_path.is_dir():
        print(f"Error: Not a directory: {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Searching for videos in: {base_dir}", file=sys.stderr)
    print(f"Pattern: {pattern}", file=sys.stderr)

    videos = list(base_path.glob(pattern))

    if sort:
        videos = sorted(videos)

    print(f"Found {len(videos)} videos", file=sys.stderr)

    return [str(v.resolve()) for v in videos]


def filter_by_factory(videos, factory_ids):
    """Filter videos by factory IDs."""
    if not factory_ids:
        return videos

    factory_set = set(factory_ids)
    filtered = []

    for video in videos:
        # Extract factory ID from path
        # Example: .../factory_1/worker_2/...
        parts = Path(video).parts
        for part in parts:
            if part.startswith("factory_"):
                try:
                    fid = int(part.split("_")[1])
                    if fid in factory_set:
                        filtered.append(video)
                        break
                except (IndexError, ValueError):
                    continue

    return filtered


def filter_by_worker(videos, worker_ids):
    """Filter videos by worker IDs."""
    if not worker_ids:
        return videos

    worker_set = set(worker_ids)
    filtered = []

    for video in videos:
        # Extract worker ID from path
        # Example: .../worker_2/processed/...
        parts = Path(video).parts
        for part in parts:
            if part.startswith("worker_"):
                try:
                    wid = int(part.split("_")[1])
                    if wid in worker_set:
                        filtered.append(video)
                        break
                except (IndexError, ValueError):
                    continue

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Generate video list from BuildAI-processed dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate list of all videos
  python scripts/generate_video_list.py \
      --base_dir /share_data/lvjianan/datasets/BuildAI-processed \
      --output videos.txt

  # Filter by specific factories
  python scripts/generate_video_list.py \\
      --base_dir /share_data/lvjianan/datasets/BuildAI-processed \\
      --factory 1 2 3 \\
      --output factory_1_2_3.txt

  # Filter by specific workers
  python scripts/generate_video_list.py \\
      --base_dir /share_data/lvjianan/datasets/BuildAI-processed \\
      --worker 1 2 \\
      --output worker_1_2.txt

  # Combine filters
  python scripts/generate_video_list.py \\
      --base_dir /share_data/lvjianan/datasets/BuildAI-processed \\
      --factory 1 \\
      --worker 1 2 3 \\
      --output factory1_workers123.txt

  # Output to stdout (for piping)
  python scripts/generate_video_list.py \\
      --base_dir /share_data/lvjianan/datasets/BuildAI-processed \\
      --no-output
        """
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory of BuildAI-processed dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Print to stdout instead of file"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.mp4",
        help="Glob pattern for video files (default: **/*.mp4)"
    )
    parser.add_argument(
        "--factory",
        type=int,
        nargs="+",
        help="Filter by factory IDs (e.g., --factory 1 2 3)"
    )
    parser.add_argument(
        "--worker",
        type=int,
        nargs="+",
        help="Filter by worker IDs (e.g., --worker 1 2)"
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort the video list"
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use slow recursive glob method instead of fast structured search"
    )

    args = parser.parse_args()

    # Use fast method by default (leverages known directory structure)
    # Use slow method only if --slow is specified or --pattern is customized
    use_fast = not args.slow and args.pattern == "**/*.mp4"

    if use_fast:
        print("Using fast structured search (factory_*/worker_*/processed/*.mp4)", file=sys.stderr)
        videos = find_videos_fast(
            args.base_dir,
            factory_ids=args.factory,
            worker_ids=args.worker,
            sort=not args.no_sort
        )
    else:
        if args.slow:
            print("Using slow recursive glob method (--slow specified)", file=sys.stderr)
        else:
            print(f"Using slow recursive glob method (custom pattern: {args.pattern})", file=sys.stderr)

        videos = find_videos(args.base_dir, args.pattern, sort=not args.no_sort)

        # Apply filters for slow method
        if args.factory:
            print(f"Filtering by factories: {args.factory}", file=sys.stderr)
            videos = filter_by_factory(videos, args.factory)
            print(f"After factory filter: {len(videos)} videos", file=sys.stderr)

        if args.worker:
            print(f"Filtering by workers: {args.worker}", file=sys.stderr)
            videos = filter_by_worker(videos, args.worker)
            print(f"After worker filter: {len(videos)} videos", file=sys.stderr)

    if not videos:
        print("No videos found!", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.no_output or not args.output:
        # Print to stdout
        for video in videos:
            print(video)
    else:
        # Write to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for video in videos:
                f.write(f"{video}\n")

        print(f"Video list written to: {args.output}", file=sys.stderr)
        print(f"Total videos: {len(videos)}", file=sys.stderr)


if __name__ == "__main__":
    main()
