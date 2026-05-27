"""Stage 1: select a subset of sequences per dataset, extract the egocentric video,
and dump aligned ground truth (gt.npz) — the shared input for both inference systems.

Layout produced under <work_dir>/<dataset>/<sanitized_seq>/:
  video.mp4   egocentric RGB fed to BOTH the fork pipeline and upstream demo.py
  gt.npz      GTSequence arrays (camera extrinsics, K, 21 world joints, validity)
  gt.npz.meta.json

Run on the PRODUCTION machine (needs the dataset files + cv2/zarr).

    python -m scripts.eval_compare.prepare_inputs --config scripts/eval_compare/config.yaml
    python -m scripts.eval_compare.prepare_inputs --config ... --dataset egoverse --limit 1
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from scripts.eval_compare.common import (
    load_config,
    sample_sequences,
    seq_workdir,
    write_video_from_frames,
    write_video_from_tar,
    write_video_from_zarr,
)
from scripts.eval_compare.gt_adapters import get_adapter


def _extract_video(gt, out_video: str, fps: float, dataset: str) -> None:
    if os.path.exists(out_video):
        return
    if dataset == "egoverse":
        write_video_from_zarr(gt.video_path, out_video, fps)
    elif gt.frame_archive and gt.frame_members:
        write_video_from_tar(gt.frame_archive, gt.frame_members, out_video, fps)
    elif gt.frame_paths:
        write_video_from_frames(gt.frame_paths, out_video, fps)
    elif gt.video_path and gt.video_path.endswith((".mp4", ".avi")):
        os.makedirs(os.path.dirname(out_video), exist_ok=True)
        if not os.path.exists(out_video):
            os.symlink(os.path.abspath(gt.video_path), out_video)
    else:
        raise RuntimeError(f"no frame source for {dataset}:{gt.seq_id}")


def prepare_dataset(name: str, cfg: dict, work_dir: str, limit: int | None, use_cuda: bool) -> None:
    dcfg = cfg["datasets"][name]
    if not dcfg.get("enabled", True):
        print(f"[{name}] disabled, skipping")
        return
    adapter = get_adapter(name)
    data_root = dcfg["data_root"]
    n = limit if limit is not None else cfg["subset_per_dataset"]

    all_seqs = adapter.list_sequences(data_root, split_file=dcfg.get("split_file"))
    seqs = sample_sequences(all_seqs, n, cfg.get("seed", 0))
    print(f"[{name}] {len(all_seqs)} sequences available -> using {len(seqs)}")

    extra = {}
    if name == "oakink2" and "ego_cam" in dcfg:
        extra["ego_cam"] = dcfg["ego_cam"]
    if name == "taco" and "trans_unit" in dcfg:
        extra["trans_unit"] = dcfg["trans_unit"]

    for sid in seqs:
        out_dir = seq_workdir(work_dir, name, sid)
        gt_path = os.path.join(out_dir, "gt.npz")
        try:
            gt = adapter.load_sequence(data_root, sid, use_cuda=use_cuda, **extra)
            gt.save_npz(gt_path)
            _extract_video(gt, os.path.join(out_dir, "video.mp4"), gt.fps, name)
            n_valid = int(gt.valid.any(axis=0).sum())
            print(f"  ok  {sid}: T={gt.valid.shape[1]} frames, {n_valid} with a hand")
        except Exception as e:  # keep going; report per-seq failures
            print(f"  FAIL {sid}: {e}")
            traceback.print_exc(file=sys.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None, help="only this dataset (default: all enabled)")
    ap.add_argument("--limit", type=int, default=None, help="override subset_per_dataset")
    ap.add_argument("--cpu", action="store_true", help="MANO FK on CPU (oakink2/taco)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work_dir = cfg["work_dir"]
    names = [args.dataset] if args.dataset else list(cfg["datasets"])
    for name in names:
        prepare_dataset(name, cfg, work_dir, args.limit, use_cuda=not args.cpu)


if __name__ == "__main__":
    main()
