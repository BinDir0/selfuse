"""Deprecated wrapper for the detect_track stage implementation."""

from lib.pipeline.stages.detect_track import detect_track_video


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--detect_batch_size", type=int, default=128)
    parser.add_argument("--detect_io_workers", type=int, default=8)
    args = parser.parse_args()

    detect_track_video(args, detect_batch_size=args.detect_batch_size, num_io_workers=args.detect_io_workers)
