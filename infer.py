#!/usr/bin/env python3
"""EgoHandSTModel inference: video (sliding window) or WebDataset windows.

Data defaults (--seq-len, --stride, --batch-size) follow train.py unless overridden.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DEFAULT_SEQ_LEN = 48
_DEFAULT_STRIDE = 16
_DEFAULT_BATCH_SIZE = 8

from dataloader import EpisodeWindowDataLoader
from dataloader.utils import sanitize_key
from egotransformer.model import EgoHandSTConfig, EgoHandSTModel
from training.batch import to_float_tensor

_MANO_KEYS = ("trans", "root_orient", "hand_pose", "betas")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EgoHandSTModel inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--args-json",
        type=str,
        default="",
        help="Training args.json to rebuild EgoHandSTConfig; optional if defaults match checkpoint",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument_group("input (use --video or WebDataset options)")
    p.add_argument("--video", type=str, default="", help="Video file; sliding-window path")
    p.add_argument("--data-path", type=str, default="", help="WebDataset tar dir / file / glob")
    p.add_argument("--shard-glob", type=str, default="*.tar")
    p.add_argument("--episodes-file", type=str, default="", help="One episode name per line (normalized)")
    p.add_argument("--episode-filter", type=str, default="", help="Single episode name; exclusive with episodes-file")

    win = p.add_argument_group("windowing")
    p.add_argument(
        "--seq-len",
        type=int,
        default=_DEFAULT_SEQ_LEN,
        help="Window length T; must be <= max_temporal_length (train.py --seq-len)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=_DEFAULT_STRIDE,
        help="Train-aligned: EpisodeWeb stride. Video: sliding stride; 0 = non-overlap (stride==seq-len). "
        "WebDataset: must be >= 1 (train.py --stride)",
    )

    wds = p.add_argument_group("WebDataset loader")
    p.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, help="train.py --batch-size")
    p.add_argument("--workers", type=int, default=0)

    out = p.add_argument_group("output")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="--video: .npz path. WebDataset: directory with one .npz per episode (per-frame fused)",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="WebDataset only: stop after N batches (0 = full iterator)",
    )
    return p.parse_args()


def _config_from_args_json(path: str) -> EgoHandSTConfig:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return EgoHandSTConfig(
        pretrained_backbone=not bool(d.get("no_pretrained", False)),
        freeze_backbone=not bool(d.get("unfreeze_backbone", False)),
        image_size=384,
        use_mano_cross_decoder=not bool(d.get("no_mano_cross_decoder", False)),
        mano_decoder_depth=int(d.get("mano_decoder_depth", 2)),
        mano_decoder_heads=int(d.get("mano_decoder_heads", 8)),
        use_mano_temporal_refine=not bool(d.get("no_mano_temporal_refine", False)),
        mano_refine_hdim=int(d.get("mano_refine_hdim", 512)),
        mano_refine_layers=int(d.get("mano_refine_layers", 2)),
        mano_refine_heads=int(d.get("mano_refine_heads", 8)),
        use_hand_side_embedding=not bool(d.get("no_hand_side_embedding", False)),
        use_hand_role_mem_bias=not bool(d.get("no_hand_role_mem_bias", False)),
    )


def _video_rgb_np(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames: {path}")
    return np.stack(frames, axis=0).astype(np.uint8)


def _resize_video_tchw(t_chw: torch.Tensor, image_size: int) -> torch.Tensor:
    _, _, h, w = t_chw.shape
    if (h, w) == (image_size, image_size):
        return t_chw
    return F.interpolate(t_chw, size=(image_size, image_size), mode="bilinear", align_corners=False)


def _preprocess_thwc_u8(frames_thwc: np.ndarray, device: torch.device, image_size: int) -> torch.Tensor:
    t = torch.from_numpy(frames_thwc).permute(0, 3, 1, 2).float().to(device)
    if t.numel() and t.max() > 1.5:
        t = t * (1.0 / 255.0)
    return _resize_video_tchw(t, image_size)


def _wds_batch_video_only(batch: dict, device: torch.device, image_size: int) -> torch.Tensor:
    # Same resize rule as training/batch.py wds_batch_to_training_batch (no intrinsics / GT).
    video = to_float_tensor(batch["video"], device)
    if video.dim() == 5 and video.max() > 1.5:
        video = video * (1.0 / 255.0)
    b, t, c, h, w = video.shape
    if (h, w) != (image_size, image_size):
        video = F.interpolate(
            video.flatten(0, 1),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).view(b, t, c, image_size, image_size)
    return video


class _EpisodeFuse:
    __slots__ = ("counts", "exist_sum", "mano_l_sum", "mano_r_sum")

    def __init__(self) -> None:
        self.counts: dict[int, int] = {}
        self.exist_sum: dict[int, np.ndarray] = {}
        self.mano_l_sum: dict[str, dict[int, np.ndarray]] = {k: {} for k in _MANO_KEYS}
        self.mano_r_sum: dict[str, dict[int, np.ndarray]] = {k: {} for k in _MANO_KEYS}

    def add_frame(
        self,
        frame_idx: int,
        prob02: np.ndarray,
        ml: dict[str, np.ndarray],
        mr: dict[str, np.ndarray],
    ) -> None:
        c = self.counts.get(frame_idx, 0) + 1
        self.counts[frame_idx] = c
        p64 = prob02.astype(np.float64, copy=False)
        if frame_idx not in self.exist_sum:
            self.exist_sum[frame_idx] = p64.copy()
            for k in _MANO_KEYS:
                self.mano_l_sum[k][frame_idx] = ml[k].astype(np.float64, copy=True)
                self.mano_r_sum[k][frame_idx] = mr[k].astype(np.float64, copy=True)
        else:
            self.exist_sum[frame_idx] = self.exist_sum[frame_idx] + p64
            for k in _MANO_KEYS:
                self.mano_l_sum[k][frame_idx] = self.mano_l_sum[k][frame_idx] + ml[k].astype(
                    np.float64, copy=False
                )
                self.mano_r_sum[k][frame_idx] = self.mano_r_sum[k][frame_idx] + mr[k].astype(
                    np.float64, copy=False
                )

    def to_npz(self, *, dataset_name: str, episode_name: str) -> dict[str, np.ndarray]:
        order = sorted(self.counts.keys())
        n = len(order)
        if n == 0:
            return {}
        cvec = np.array([self.counts[i] for i in order], dtype=np.float32)
        out: dict[str, np.ndarray] = {
            "dataset_name": np.array(dataset_name),
            "episode_name": np.array(episode_name),
            "frame_index": np.asarray(order, dtype=np.int64),
            "per_frame_window_count": cvec,
            "hand_existence_prob": np.stack(
                [self.exist_sum[i] / self.counts[i] for i in order]
            ).astype(np.float32),
        }
        for k in _MANO_KEYS:
            out[f"mano_left_{k}"] = np.stack(
                [self.mano_l_sum[k][i] / self.counts[i] for i in order]
            ).astype(np.float32)
            out[f"mano_right_{k}"] = np.stack(
                [self.mano_r_sum[k][i] / self.counts[i] for i in order]
            ).astype(np.float32)
        return out


def _collate_str_list(batch_val: object, b: int) -> str:
    if isinstance(batch_val, (list, tuple)):
        return str(batch_val[b])
    return str(batch_val)


def _as_frame_indices_np(batch: dict) -> np.ndarray:
    fi = batch["frame_indices"]
    if torch.is_tensor(fi):
        return fi.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(fi, dtype=np.int64)


@torch.no_grad()
def _infer_video_sliding(
    model: EgoHandSTModel,
    video_t3hw: torch.Tensor,
    *,
    window: int,
    stride: int,
    max_t: int,
) -> dict[str, np.ndarray]:
    if window < 1 or window > max_t:
        raise ValueError(f"window must be in [1, {max_t}], got {window}")
    if stride <= 0:
        stride = window
    t_total = video_t3hw.shape[0]
    device = video_t3hw.device
    count = torch.zeros(t_total, device=device, dtype=torch.float32)
    exist_sum = torch.zeros(t_total, 2, device=device, dtype=torch.float32)
    mano_l: dict[str, torch.Tensor] | None = None
    mano_r: dict[str, torch.Tensor] | None = None

    for start in range(0, t_total, stride):
        end_real = min(start + window, t_total)
        n_real = end_real - start
        chunk = video_t3hw[start:end_real]
        if n_real < window:
            pad = window - n_real
            chunk = torch.cat([chunk, chunk[-1:].expand(pad, *chunk.shape[1:])], dim=0)
        o = model(chunk.unsqueeze(0))
        prob = torch.sigmoid(o["hand_existence_logits"][0, :n_real])
        ml, mr = o["mano_left"], o["mano_right"]
        idx = torch.arange(start, end_real, device=device)
        if mano_l is None:
            mano_l = {
                k: torch.zeros(t_total, *ml[k][0, 0].shape, device=device, dtype=ml[k].dtype)
                for k in _MANO_KEYS
            }
            mano_r = {
                k: torch.zeros(t_total, *mr[k][0, 0].shape, device=device, dtype=mr[k].dtype)
                for k in _MANO_KEYS
            }
        count.index_add_(0, idx, torch.ones(n_real, device=device))
        exist_sum.index_add_(0, idx, prob)
        for k in _MANO_KEYS:
            mano_l[k].index_add_(0, idx, ml[k][0, :n_real])
            mano_r[k].index_add_(0, idx, mr[k][0, :n_real])

    assert mano_l is not None and mano_r is not None
    c = count.clamp(min=1.0).unsqueeze(-1)
    stacked: dict[str, np.ndarray] = {"hand_existence_prob": (exist_sum / c).cpu().numpy()}
    for k in _MANO_KEYS:
        stacked[f"mano_left_{k}"] = (mano_l[k] / c).cpu().numpy()
        stacked[f"mano_right_{k}"] = (mano_r[k] / c).cpu().numpy()
    return stacked


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    use_video = bool(args.video.strip())
    use_wds = bool(args.data_path.strip())
    if use_video == use_wds:
        raise SystemExit("Set exactly one of: --video  OR  --data-path (WebDataset)")

    cfg = _config_from_args_json(args.args_json) if args.args_json.strip() else EgoHandSTConfig()
    max_t = cfg.max_temporal_length
    window = int(args.seq_len)
    if window > max_t:
        raise SystemExit(f"--seq-len {window} > max_temporal_length {max_t}")

    model = EgoHandSTModel(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.eval()

    if use_video:
        frames = _video_rgb_np(args.video)
        vid = _preprocess_thwc_u8(frames, device, cfg.image_size)
        stride = int(args.stride) if int(args.stride) > 0 else window
        stacked = _infer_video_sliding(model, vid, window=window, stride=stride, max_t=max_t)
        out_path = args.out.strip() or str(
            Path(args.video).resolve().parent / f"{Path(args.video).stem}_egohand.npz"
        )
        np.savez_compressed(out_path, **stacked)
        print(f"wrote {out_path} T={frames.shape[0]} window={window} stride={stride}")
        return

    ef = args.episodes_file.strip()
    sf = args.episode_filter.strip()
    if ef and sf:
        raise SystemExit("use either --episodes-file or --episode-filter, not both")

    wds_stride = int(args.stride)
    if wds_stride < 1:
        raise SystemExit("WebDataset inference requires --stride >= 1 (same as train dataloader)")

    loader = EpisodeWindowDataLoader(
        args.data_path,
        window_size=window,
        stride=wds_stride,
        shard_glob=args.shard_glob,
        episode_list_file=ef or None,
        episode_filter=sf or None,
        dist_rank=0,
        dist_world_size=1,
        ddp_read_all_shards=False,
        shuffle=False,
        shuffle_seed=0,
        batch_size=int(args.batch_size),
        num_workers=int(args.workers),
        pin_memory=device.type == "cuda",
    )

    out_dir = Path(args.out.strip() or "infer_wds_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    fused: dict[tuple[str, str], _EpisodeFuse] = {}
    limit = int(args.max_batches)

    for i, batch in enumerate(loader):
        if "frame_indices" not in batch:
            raise SystemExit("WebDataset batch missing frame_indices")
        if "episode_name" not in batch or "dataset_name" not in batch:
            raise SystemExit("WebDataset batch missing episode_name or dataset_name")

        video = _wds_batch_video_only(batch, device, cfg.image_size)
        out = model(video)
        prob = torch.sigmoid(out["hand_existence_logits"]).detach().float().cpu().numpy()
        fi_np = _as_frame_indices_np(batch)
        bsz, t_win = fi_np.shape

        ml_np = {k: out["mano_left"][k].detach().float().cpu().numpy() for k in _MANO_KEYS}
        mr_np = {k: out["mano_right"][k].detach().float().cpu().numpy() for k in _MANO_KEYS}

        for b in range(bsz):
            ds = _collate_str_list(batch["dataset_name"], b)
            ep = _collate_str_list(batch["episode_name"], b)
            key = (ds, ep)
            if key not in fused:
                fused[key] = _EpisodeFuse()
            acc = fused[key]
            for t in range(t_win):
                acc.add_frame(
                    int(fi_np[b, t]),
                    prob[b, t],
                    {k: ml_np[k][b, t] for k in _MANO_KEYS},
                    {k: mr_np[k][b, t] for k in _MANO_KEYS},
                )

        if limit and (i + 1) >= limit:
            break

    n_ep = 0
    for (ds, ep), acc in sorted(fused.items(), key=lambda x: (x[0][0], x[0][1])):
        payload = acc.to_npz(dataset_name=ds, episode_name=ep)
        if not payload:
            continue
        fname = f"{sanitize_key(ds)}__{sanitize_key(ep)}.npz"
        path = out_dir / fname
        np.savez_compressed(path, **payload)
        n_ep += 1
        print(f"wrote {path}  n_frames={payload['frame_index'].shape[0]}")

    if n_ep == 0:
        print("no episode predictions written (empty iterator or fused buffers)")


if __name__ == "__main__":
    main()
