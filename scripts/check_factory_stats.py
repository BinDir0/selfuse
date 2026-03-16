"""统计 factory range 内的视频总数和总帧数。

用法: python check_factory_stats.py [--start 1] [--end 70]
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="统计 factory 视频数和帧数")
    parser.add_argument("--base", default="/share_data/guantianrui/datasets/Egocentric-100K/processed_v9_test_jpg")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=70)
    args = parser.parse_args()

    total_videos = 0
    total_frames = 0
    factory_stats = []

    for fid in range(args.start, args.end + 1):
        factory_dir = os.path.join(args.base, f"factory{fid:03d}")
        index_path = os.path.join(factory_dir, "_video_index.json")

        if not os.path.exists(index_path):
            continue

        try:
            with open(index_path) as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        videos = index.get("videos", {})
        n_videos = len(videos)
        n_frames = sum(v.get("num_frames", len(v.get("frames", []))) for v in videos.values())

        total_videos += n_videos
        total_frames += n_frames
        factory_stats.append((fid, n_videos, n_frames))

    print(f"Factory range: {args.start:03d} ~ {args.end:03d}")
    print(f"{'Factory':<12} {'Videos':>8} {'Frames':>10}")
    print("-" * 32)
    for fid, nv, nf in factory_stats:
        print(f"factory{fid:03d}   {nv:>8} {nf:>10}")
    print("-" * 32)
    print(f"{'TOTAL':<12} {total_videos:>8} {total_frames:>10}")
    print(f"\n平均 {total_frames / max(total_videos, 1):.1f} 帧/视频")


if __name__ == "__main__":
    main()
