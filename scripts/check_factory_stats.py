"""统计 factory range 内的视频总数和总帧数。

用法:
  python scripts/check_factory_stats.py --start 1 --end 238
  python scripts/check_factory_stats.py --start 1 --end 238 --build   # 自动为缺失索引的 factory 建索引
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_or_build_index(factory_dir, build=False):
    """加载 _video_index.json，如果不存在且 build=True 则扫描建索引。"""
    index_path = os.path.join(factory_dir, "_video_index.json")

    if os.path.exists(index_path):
        try:
            with open(index_path) as f:
                return json.load(f), False  # (index, was_built)
        except (json.JSONDecodeError, OSError):
            pass

    if not build:
        return None, False

    # 检查是否有 tar 文件
    tars = [f for f in os.listdir(factory_dir) if f.endswith(".tar")]
    if not tars:
        return None, False

    from lib.pipeline.video_index import build_video_index

    print(f"  Building index for {os.path.basename(factory_dir)} ({len(tars)} shards)...")
    index = build_video_index(factory_dir)

    # 缓存
    try:
        with open(index_path, "w") as f:
            json.dump(index, f, ensure_ascii=False)
    except OSError:
        pass

    return index, True


def main():
    parser = argparse.ArgumentParser(description="统计 factory 视频数和帧数")
    parser.add_argument("--base", default="/share_data/guantianrui/datasets/Egocentric-100K/processed_v9_test_jpg")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=70)
    parser.add_argument("--build", action="store_true",
                        help="为缺少 _video_index.json 的 factory 自动扫描建索引")
    args = parser.parse_args()

    total_videos = 0
    total_frames = 0
    factory_stats = []
    skipped = []
    built = []

    for fid in range(args.start, args.end + 1):
        factory_dir = os.path.join(args.base, f"factory{fid:03d}")
        if not os.path.isdir(factory_dir):
            continue

        index, was_built = load_or_build_index(factory_dir, build=args.build)
        if index is None:
            skipped.append(fid)
            continue
        if was_built:
            built.append(fid)

        videos = index.get("videos", {})
        n_videos = len(videos)
        n_frames = sum(v.get("num_frames", len(v.get("frames", []))) for v in videos.values())

        total_videos += n_videos
        total_frames += n_frames
        factory_stats.append((fid, n_videos, n_frames))

    print(f"\nFactory range: {args.start:03d} ~ {args.end:03d}")
    print(f"{'Factory':<12} {'Videos':>8} {'Frames':>10}")
    print("-" * 32)
    for fid, nv, nf in factory_stats:
        print(f"factory{fid:03d}   {nv:>8} {nf:>10}")
    print("-" * 32)
    print(f"{'TOTAL':<12} {total_videos:>8} {total_frames:>10}")
    print(f"\n平均 {total_frames / max(total_videos, 1):.1f} 帧/视频")

    if built:
        print(f"\n新建索引: {len(built)} 个 factory: {', '.join(f'factory{f:03d}' for f in built)}")
    if skipped:
        print(f"跳过(无索引/无tar): {len(skipped)} 个 factory: {', '.join(f'factory{f:03d}' for f in skipped)}")


if __name__ == "__main__":
    main()
