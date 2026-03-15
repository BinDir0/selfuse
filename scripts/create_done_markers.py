#!/usr/bin/env python3
"""
批量为已完成的视频创建.done标记文件

这个脚本会：
1. 扫描指定目录下的所有视频文件夹
2. 检查每个stage是否真正完成（验证输出文件）
3. 只为确实完成的stage创建.done标记

用法:
    python scripts/create_done_markers.py /path/to/video/root --stages detect_track motion slam infiller
    python scripts/create_done_markers.py /path/to/video/root --stages detect_track --dry-run
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_worker import (
    is_stage_complete,
    get_track_range,
    validate_stage_output_fast,
    STAGES
)


def find_video_folders(root_dir: Path, max_depth: int = 5):
    """
    查找所有可能的视频文件夹（包含tracks_*_*目录的文件夹）

    Args:
        root_dir: 根目录
        max_depth: 最大搜索深度

    Returns:
        List of video folder paths
    """
    video_folders = []

    def search_recursive(current_dir: Path, depth: int):
        if depth > max_depth:
            return

        try:
            # 检查当前目录是否包含tracks_*_*
            has_tracks = any(
                p.is_dir() and p.name.startswith("tracks_")
                for p in current_dir.iterdir()
            )

            if has_tracks:
                video_folders.append(current_dir)
                return  # 找到视频文件夹，不再深入

            # 继续搜索子目录
            for subdir in current_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    search_recursive(subdir, depth + 1)
        except PermissionError:
            pass

    search_recursive(root_dir, 0)
    return video_folders


def create_done_markers(
    root_dir: Path,
    stages: list,
    dry_run: bool = False,
    verbose: bool = False
):
    """
    为已完成的视频创建.done标记

    Args:
        root_dir: 视频根目录
        stages: 要检查的stage列表
        dry_run: 如果为True，只检查不创建文件
        verbose: 是否显示详细信息
    """
    print(f"Scanning video folders in: {root_dir}")
    video_folders = find_video_folders(root_dir)
    print(f"Found {len(video_folders)} video folders\n")

    if len(video_folders) == 0:
        print("No video folders found!")
        return

    # 统计信息
    stats = {stage: {'total': 0, 'complete': 0, 'created': 0, 'existing': 0}
             for stage in stages}

    # 处理每个视频文件夹
    for seq_folder in tqdm(video_folders, desc="Processing videos"):
        for stage in stages:
            done_marker = seq_folder / f".{stage}.done"

            # 检查是否已有.done标记
            if done_marker.exists():
                stats[stage]['existing'] += 1
                if verbose:
                    print(f"[SKIP] {seq_folder.name}: .{stage}.done already exists")
                continue

            # 验证stage是否真正完成
            try:
                start_idx, end_idx = get_track_range(seq_folder, fast=True)
                is_complete = validate_stage_output_fast(stage, seq_folder, start_idx, end_idx)

                stats[stage]['total'] += 1

                if is_complete:
                    stats[stage]['complete'] += 1

                    if not dry_run:
                        # 创建.done标记
                        done_marker.touch()
                        stats[stage]['created'] += 1
                        if verbose:
                            print(f"[CREATE] {seq_folder.name}: .{stage}.done")
                    else:
                        if verbose:
                            print(f"[DRY-RUN] {seq_folder.name}: would create .{stage}.done")
                else:
                    if verbose:
                        print(f"[INCOMPLETE] {seq_folder.name}: {stage} not complete")

            except Exception as e:
                if verbose:
                    print(f"[ERROR] {seq_folder.name}: {stage} - {e}")
                continue

    # 打印统计信息
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)

    for stage in stages:
        s = stats[stage]
        print(f"\n{stage}:")
        print(f"  Already had .done:     {s['existing']:6d}")
        print(f"  Checked (no .done):    {s['total']:6d}")
        print(f"  Complete:              {s['complete']:6d}")
        if dry_run:
            print(f"  Would create .done:    {s['complete']:6d}")
        else:
            print(f"  Created .done:         {s['created']:6d}")

    print("\n" + "="*60)

    if dry_run:
        print("\nDRY RUN MODE - No files were created")
        print("Run without --dry-run to actually create .done markers")
    else:
        total_created = sum(s['created'] for s in stats.values())
        print(f"\nTotal .done markers created: {total_created}")


def main():
    parser = argparse.ArgumentParser(
        description="批量为已完成的视频创建.done标记文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 检查所有stage（dry-run模式）
  python scripts/create_done_markers.py /path/to/videos --dry-run

  # 只为detect_track和motion创建.done
  python scripts/create_done_markers.py /path/to/videos --stages detect_track motion

  # 创建所有stage的.done标记
  python scripts/create_done_markers.py /path/to/videos --stages detect_track motion slam infiller

  # 显示详细信息
  python scripts/create_done_markers.py /path/to/videos --stages detect_track --verbose
        """
    )

    parser.add_argument(
        'root_dir',
        type=Path,
        help='视频根目录'
    )

    parser.add_argument(
        '--stages',
        nargs='+',
        choices=STAGES,
        default=['detect_track', 'motion', 'slam', 'infiller'],
        help='要检查的stage列表（默认：所有stage）'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只检查不创建文件（用于测试）'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细信息'
    )

    parser.add_argument(
        '--max-depth',
        type=int,
        default=5,
        help='最大搜索深度（默认：5）'
    )

    args = parser.parse_args()

    # 验证根目录存在
    if not args.root_dir.exists():
        print(f"Error: Directory does not exist: {args.root_dir}")
        sys.exit(1)

    if not args.root_dir.is_dir():
        print(f"Error: Not a directory: {args.root_dir}")
        sys.exit(1)

    # 确认操作
    if not args.dry_run:
        print(f"This will create .done markers for stages: {', '.join(args.stages)}")
        print(f"Root directory: {args.root_dir}")
        response = input("\nContinue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        print()

    # 执行
    create_done_markers(
        root_dir=args.root_dir,
        stages=args.stages,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
