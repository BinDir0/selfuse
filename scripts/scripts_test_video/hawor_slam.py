"""Deprecated wrapper for the SLAM stage implementation."""

from lib.pipeline.stages.slam import build_metric3d_runner, hawor_slam

__all__ = ["build_metric3d_runner", "hawor_slam"]


if __name__ == "__main__":
    import argparse
    from lib.pipeline.stages.detect_track import detect_track_video

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--input_type", type=str, default="file")
    args = parser.parse_args()

    start_idx, end_idx, _, _ = detect_track_video(args)
    hawor_slam(args, start_idx, end_idx)
