#!/usr/bin/env python3
"""Train EgoHandSTModel (WebDataset tar shards).

Usage:
python train.py \
  --data-path /share_data/zhangtingrui/datasets/taco_v2 \
  --episodes-file splits/train.txt

torchrun --standalone --nproc_per_node=8 train.py \
  --data-path /share_data/zhangtingrui/datasets/taco_v2 \
  --episodes-file splits/taco_v2/train.txt \
  --run-dir runs/my_exp12 \
  --tensorboard-dir runs/my_exp12/tb


# eval_only:
python train.py --eval-only \
  --resume runs/my_exp1/checkpoints/latest.pt \
  --data-path /share_data/zhangtingrui/datasets/taco_v2 \
  --val-episodes-file splits/val.txt \
  --run-dir runs/my_exp1 \
  --tensorboard-dir runs/my_exp1/tb

torchrun --standalone --nproc_per_node=8 train.py --eval-only \
    --resume runs/my_exp2/checkpoints/latest.pt \
    --data-path /share_data/zhangtingrui/datasets/oakink2_v4 \
    --val-episodes-file splits/oakink_full/val.txt \
    --run-dir runs/my_exp2 \
    --tensorboard-dir runs/my_exp2/tb_val

tensorboard --logdir runs/my_exp1/tb

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, cast

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from dataloader import (
    DEFAULT_WINDOW_SHUFFLE_BUFFER_SIZE,
    DEFAULT_WINDOW_SHUFFLE_SEED,
    DEFAULT_WINDOW_SHUFFLE_WINDOWS,
    EpisodeWindowDataLoader,
)
from dataloader.mano_pca import get_cached_mano_pca_decode_layers
from egotransformer.model import EgoHandSTConfig, EgoHandSTModel
from training.checkpoint import module_state_dict
from training.logging_utils import log_line
from training.losses import MANO_PARAM_KEY_CHOICES, mano_joint_weight_vector_21
from training.train_loop import eval_one_epoch, train_one_epoch


def _load_train_config_defaults(path: str, valid_dests: set[str]) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"--config not found: {path}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in --config: {e}") from e

    if not isinstance(payload, dict):
        raise ValueError("--config must be a JSON object")

    flat: dict[str, Any] = {}
    bad_keys: list[str] = []
    for key, value in payload.items():
        if key in valid_dests and not isinstance(value, dict):
            flat[key] = value
            continue

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key in valid_dests:
                    flat[sub_key] = sub_value
                else:
                    bad_keys.append(f"{key}.{sub_key}")
            continue

        bad_keys.append(key)

    if bad_keys:
        sample = ", ".join(sorted(bad_keys)[:20])
        raise ValueError(f"unknown keys in --config: {sample}")

    return flat


def _parse_mano_param_keys(text: str) -> frozenset[str]:
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("--mano-param-keys must list at least one of: trans, root_orient, hand_pose, betas")
    bad = [p for p in parts if p not in MANO_PARAM_KEY_CHOICES]
    if bad:
        raise ValueError(f"unknown --mano-param-keys entries {bad}; allowed: {sorted(MANO_PARAM_KEY_CHOICES)}")
    return frozenset(parts)


def _load_datasets_config(path: str) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"--datasets-config not found: {path}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in --datasets-config: {e}") from e

    items: Any
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("datasets"), list):
        items = payload["datasets"]
    else:
        raise ValueError("--datasets-config must be a JSON list or an object with key 'datasets'")

    normalized: list[dict[str, str]] = []
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise ValueError(f"datasets[{i}] must be a JSON object")
        if raw.get("enabled", True) is False:
            continue

        data_path = str(raw.get("data_path", "")).strip()
        if not data_path:
            raise ValueError(f"datasets[{i}].data_path is required")

        episodes_file = str(raw.get("episodes_file", "")).strip()
        episode_filter = str(raw.get("episode_filter", "")).strip()
        if episodes_file and episode_filter:
            raise ValueError(f"datasets[{i}]: use either episodes_file or episode_filter, not both")

        src: dict[str, str] = {
            "name": str(raw.get("name", f"dataset_{i}")),
            "data_path": data_path,
        }
        shard_glob = str(raw.get("shard_glob", "")).strip()
        if shard_glob:
            src["shard_glob"] = shard_glob
        if episodes_file:
            src["episodes_file"] = episodes_file
        if episode_filter:
            src["episode_filter"] = episode_filter
        normalized.append(src)

    if not normalized:
        raise ValueError("--datasets-config has no enabled datasets")
    return normalized


def _dataset_sources_has_episode_restriction(dataset_sources: list[dict[str, str]]) -> bool:
    for src in dataset_sources:
        if src.get("episodes_file", "").strip() or src.get("episode_filter", "").strip():
            return True
    return False


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train EgoHandSTModel")
    p.add_argument(
        "--config",
        type=str,
        default="configs/train_default.json",
        help="training config JSON (grouped sections); CLI args override config values",
    )

    d = p.add_argument_group("data")
    d.add_argument("--data-path", type=str, default="", help="tar dir, file, or glob (legacy single-dataset mode)")
    d.add_argument(
        "--datasets-config",
        type=str,
        default="configs/multi_datasets.json",
        help="JSON file for multi-dataset train input; list entries require data_path and optional episodes_file/episode_filter",
    )
    d.add_argument("--shard-glob", type=str, default="*.tar")
    d.add_argument(
        "--episodes-file",
        type=str,
        default="",
        help="train episode list, one name per line; empty = all",
    )
    d.add_argument(
        "--episode-filter",
        type=str,
        default="",
        help="single episode (debug); mutually exclusive with --episodes-file",
    )
    d.add_argument("--val-episodes-file", type=str, default="", help="val episode list; empty = skip val")
    d.add_argument("--seq-len", type=int, default=48, help="window length T")
    d.add_argument("--stride", type=int, default=16)
    d.add_argument("--batch-size", type=int, default=8, help="per-GPU batch (global ~ batch * num GPUs)")
    d.add_argument("--workers", type=int, default=0)
    d.add_argument(
        "--no-shuffle-data",
        action="store_true",
        help="disable tar-shard order and within-episode window order shuffling",
    )
    d.add_argument(
        "--no-shuffle-windows",
        action="store_true",
        help="disable window-level shuffle within each epoch (keeps shard shuffle unless --no-shuffle-data)",
    )
    d.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=DEFAULT_WINDOW_SHUFFLE_BUFFER_SIZE,
        help="window shuffle buffer size; <=1 disables window-buffer mixing",
    )
    d.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_WINDOW_SHUFFLE_SEED,
        help="seed for shard/window shuffling; negative means reuse --seed",
    )

    a = p.add_argument_group("augment")
    a.add_argument("--aug-enable", action="store_true", help="enable training-time GPU augmentations")
    a.add_argument("--aug-color-temp-prob", type=float, default=0.0)
    a.add_argument(
        "--aug-color-temp-strength",
        type=float,
        default=0.08,
        help="channel gain strength for color temperature shift (R up / B down)",
    )
    a.add_argument("--aug-contrast-prob", type=float, default=0.0)
    a.add_argument("--aug-contrast-min", type=float, default=0.9)
    a.add_argument("--aug-contrast-max", type=float, default=1.1)
    a.add_argument("--aug-saturation-prob", type=float, default=0.0)
    a.add_argument("--aug-saturation-min", type=float, default=0.9)
    a.add_argument("--aug-saturation-max", type=float, default=1.1)
    a.add_argument("--aug-grayscale-prob", type=float, default=0.0)
    a.add_argument("--aug-scale-prob", type=float, default=0.0)
    a.add_argument("--aug-scale-min", type=float, default=0.9)
    a.add_argument("--aug-scale-max", type=float, default=1.1)
    a.add_argument(
        "--aug-scale-pad-mode",
        type=str,
        default="constant_mean",
        choices=("constant_mean", "constant_zero", "reflect"),
    )
    a.add_argument(
        "--aug-invisible-joint-threshold",
        type=int,
        default=20,
        help="set existence to 0 when invisible MANO joints >= threshold",
    )
    a.add_argument(
        "--aug-no-update-existence",
        action="store_true",
        help="disable visibility-based existence relabel after scaling augment",
    )

    o = p.add_argument_group("optim")
    o.add_argument("--epochs", type=int, default=5)
    o.add_argument("--lr", type=float, default=2e-4)
    o.add_argument("--weight-decay", type=float, default=0.01)
    o.add_argument("--grad-clip", type=float, default=1.0)
    o.add_argument("--grad-accum-steps", type=int, default=4, help="optimizer step every N micro-batches")
    o.add_argument(
        "--warmup-global-steps",
        type=int,
        default=10,
        help="linear LR warmup optimizer steps; during warmup, perspective loss is forced to 0",
    )
    o.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="max optimizer steps per rank; 0 = epoch only",
    )
    o.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    m = p.add_argument_group("mano")
    m.add_argument("--no-decode-hand-pca", action="store_true")
    m.add_argument(
        "--mano-pose-weight",
        type=float,
        default=0.5,
        help="relative weight for hand_pose inside param regression when hand_pose is in --mano-param-keys",
    )
    m.add_argument(
        "--mano-root-orient-weight",
        type=float,
        default=1.0,
        help="relative weight for root_orient inside param regression when root_orient is in --mano-param-keys "
        "(default 1.0 vs 1.0 for trans/betas)",
    )
    m.add_argument(
        "--mano-left-root-fix",
        action="store_true",
        help="",
    )
    m.add_argument(
        "--mano-param-keys",
        type=str,
        default="trans,root_orient,hand_pose,betas",
        help="comma-separated subset of trans,root_orient,hand_pose,betas for vector regression "
        "(default: all four keys)",
    )
    m.add_argument(
        "--mano-param-loss",
        type=str,
        default="l2",
        choices=("l1", "l2", "huber"),
        help="vector loss type for selected param keys",
    )
    m.add_argument(
        "--mano-huber-delta",
        type=float,
        default=1.0,
        help="smooth_l1 beta when kind is huber",
    )
    m.add_argument(
        "--mano-param-loss-weight",
        type=float,
        default=1.0,
        help="scale for vector regression on --mano-param-keys; 0 disables",
    )
    m.add_argument(
        "--mano-joint-loss-weight",
        type=float,
        default=1.0,
        help="scale for J(pred MANO) vs J(GT MANO) MSE (m²); >0 needs MANO decode",
    )
    m.add_argument(
        "--mano-joint-weight-preset",
        type=str,
        default="fingertip",
        choices=("uniform", "fingertip"),
        help="default fingertip: upweight tips by --mano-joint-fingertip-scale; wrist (index 0) by --mano-joint-wrist-scale; "
        "all start from --mano-joint-chain-scale. uniform: only wrist scale applies unless it is 1.0 (then plain mean over 21)",
    )
    m.add_argument(
        "--mano-joint-chain-scale",
        type=float,
        default=0.5,
        help="base weight for non-wrist joints before tip/wrist multipliers (MANO indices 1–20 in fingertip preset); "
        "wrist is chain_scale * wrist_scale; tips are chain_scale * fingertip_scale; must be > 0",
    )
    m.add_argument(
        "--mano-joint-fingertip-scale",
        type=float,
        default=1.0,
        help="fingertip joint weight multiplier vs other joints (must be > 0); only used when preset is fingertip",
    )
    m.add_argument(
        "--mano-joint-wrist-scale",
        type=float,
        default=2.0,
        help="multiply weight on wrist joint (MANO index 0) in 3D joint loss; 1.0 recovers no extra wrist emphasis "
        "when preset is uniform",
    )
    m.add_argument(
        "--mano-joint-smooth-max-weight",
        type=float,
        default=0.0,
        help="optional soft-max over per-joint errors (0=off, default); not used unless >0",
    )
    m.add_argument(
        "--mano-joint-smooth-max-tau",
        type=float,
        default=0.002,
        help="temperature (m^2) for --mano-joint-smooth-max-weight when enabled",
    )
    m.add_argument(
        "--mano-joint-chunk",
        type=int,
        default=512,
        help="sub-batch size for ManoLayer joint forward passes",
    )
    m.add_argument(
        "--mano-persp-loss-weight",
        type=float,
        default=2.5,
        help="scale for full-perspective pixel reprojection using pred_intr(fx,fy,cx,cy); 0 disables",
    )
    m.add_argument(
        "--mano-persp-reg-weight",
        type=float,
        default=0.05,
        help="regularize predicted intrinsics around init priors (low default); 0 disables",
    )
    m.add_argument(
        "--mano-weakcam-loss-weight",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    m.add_argument(
        "--mano-weakcam-reg-weight",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )

    mo = p.add_argument_group("model")
    mo.add_argument("--no-pretrained", action="store_true")
    mo.add_argument("--unfreeze-backbone", action="store_true")
    mo.add_argument(
        "--no-mano-cross-decoder",
        action="store_true",
        help="disable TransformerDecoder from hand token to ST patches; use linear MANO heads on hand token only",
    )
    mo.add_argument("--mano-decoder-depth", type=int, default=2)
    mo.add_argument("--mano-decoder-heads", type=int, default=8)
    mo.add_argument(
        "--no-mano-temporal-refine",
        action="store_true",
        help="disable temporal Transformer on 61-D MANO vector (no sequence refine after heads)",
    )
    mo.add_argument("--mano-refine-hdim", type=int, default=512)
    mo.add_argument("--mano-refine-layers", type=int, default=2)
    mo.add_argument("--mano-refine-heads", type=int, default=8)
    mo.add_argument(
        "--no-hand-side-embedding",
        action="store_true",
        help="disable learnable left/right slot embedding added to hand queries",
    )
    mo.add_argument(
        "--no-hand-role-mem-bias",
        action="store_true",
        help="disable per-hand additive bias on ST tokens (single batched cross-attn)",
    )
    mo.add_argument("--camera-init-fx", type=float, default=384.0)
    mo.add_argument("--camera-init-fy", type=float, default=384.0)
    mo.add_argument("--camera-init-cx", type=float, default=192.0)
    mo.add_argument("--camera-init-cy", type=float, default=192.0)

    io = p.add_argument_group("checkpoint & log")
    io.add_argument("--save", type=str, default="", help="extra final .pt path")
    io.add_argument("--checkpoint-dir", type=str, default="")
    io.add_argument("--save-every-steps", type=int, default=0)
    io.add_argument("--resume", type=str, default="")
    io.add_argument("--tensorboard-dir", type=str, default="")
    io.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging on rank0")
    io.add_argument("--wandb-project", type=str, default="", help="W&B project name; default: egotransformer")
    io.add_argument("--wandb-entity", type=str, default="", help="optional W&B entity/team")
    io.add_argument("--wandb-name", type=str, default="", help="optional W&B run name; default: run-dir basename")
    io.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="W&B mode; disabled behaves like --wandb off",
    )
    io.add_argument(
        "--wandb-dir",
        type=str,
        default="",
        help="directory for W&B metadata/cache; default: run-dir or current working directory",
    )
    io.add_argument("--log-dir", type=str, default="")
    io.add_argument("--run-dir", type=str, default="", help="sets default checkpoints/ and logs/")
    io.add_argument(
        "--render-mano-every",
        type=int,
        default=10,
        help="every N optimizer steps (rank0): save 3-panel MANO PNG (GT | GT intr + Pred | Pred intr + Pred) under <run-dir>/render_mano; "
        "0=off. Default 10 when training; skipped without --run-dir/--render-mano-dir (see log). "
        "Needs MANO decode layers unless --no-decode-hand-pca.",
    )
    io.add_argument(
        "--render-mano-dir",
        type=str,
        default="",
        help="output directory for --render-mano-every; default: <run-dir>/render_mano",
    )

    r = p.add_argument_group("run")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--no-progress", action="store_true")
    r.add_argument(
        "--no-ddp-find-unused",
        action="store_true",
        help="DDP with find_unused_parameters=False (faster; may error if a batch skips MANO heads)",
    )
    r.add_argument(
        "--ddp-read-all-shards",
        action="store_true",
        help="DDP: force each rank to iterate all tar shards (also the default when using "
        "--episodes-file under DDP; avoids ranks stalling on shards with no allowlisted episodes).",
    )
    r.add_argument(
        "--ddp-shard-striping",
        action="store_true",
        help="DDP: shard striping by rank only (disables read-all-shards even with --episodes-file). "
        "May leave fast ranks waiting at the first all_reduce.",
    )
    r.add_argument(
        "--eval-only",
        action="store_true",
        help="validation only; needs --resume and --val-episodes-file",
    )
    return p


def _parse_args() -> argparse.Namespace:
    p = _build_arg_parser()
    pre_args, _ = p.parse_known_args()

    config_path = str(getattr(pre_args, "config", "")).strip()
    if config_path:
        valid_dests = {
            a.dest
            for a in p._actions
            if a.dest not in ("help", argparse.SUPPRESS) and isinstance(a.dest, str)
        }
        try:
            config_defaults = _load_train_config_defaults(config_path, valid_dests)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        p.set_defaults(**config_defaults)

    return p.parse_args()


def _dist_env() -> tuple[bool, int, int, int]:
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1:
        return True, int(os.environ["RANK"]), ws, int(os.environ.get("LOCAL_RANK", "0"))
    return False, 0, 1, 0


def _prepare_ddp_master_addr(world_size: int) -> None:
    if os.environ.get("MASTER_ADDR_USE_HOSTNAME", "").strip().lower() in ("1", "true", "yes"):
        return
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size == local_ws:
        os.environ["MASTER_ADDR"] = "127.0.0.1"


def _ddp_read_all_shards_effective(args: argparse.Namespace, use_dist: bool, world_size: int) -> bool:
    if args.ddp_shard_striping:
        return False
    if args.ddp_read_all_shards:
        return True
    return bool(use_dist and world_size > 1 and args.episodes_file.strip())


def _build_loader(
    data_path: str,
    *,
    dataset_sources: list[dict[str, str]] | None = None,
    window_size: int,
    stride: int,
    shard_glob: str,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    episodes_file: str,
    episode_filter: str | None,
    dist_rank: int = 0,
    dist_world_size: int = 1,
    ddp_read_all_shards: bool = False,
    shuffle: bool = True,
    shuffle_windows: bool = DEFAULT_WINDOW_SHUFFLE_WINDOWS,
    shuffle_buffer_size: int = DEFAULT_WINDOW_SHUFFLE_BUFFER_SIZE,
    shuffle_seed: int = 0,
) -> DataLoader:
    ef = episodes_file.strip()
    sf = (episode_filter or "").strip() or None
    if dataset_sources is None and ef and sf:
        raise ValueError("use either --episodes-file or --episode-filter, not both")
    return EpisodeWindowDataLoader(
        data_path,
        dataset_sources=dataset_sources,
        window_size=window_size,
        stride=stride,
        shard_glob=shard_glob,
        episode_list_file=ef or None,
        episode_filter=sf,
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
        ddp_read_all_shards=ddp_read_all_shards,
        shuffle=shuffle,
        shuffle_windows=shuffle_windows,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_seed=shuffle_seed,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_args_json(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str, sort_keys=True)


def _migrate_legacy_cam_head_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    init_fx: float,
    init_fy: float,
    init_cx: float,
    init_cy: float,
    target_in_dim: int,
) -> dict[str, torch.Tensor]:
    """Migrate legacy camera heads to current intrinsics head shape (4, target_in_dim)."""

    def _migrate_head(prefix: str) -> None:
        w_key = f"{prefix}.weight"
        b_key = f"{prefix}.bias"
        if w_key not in state_dict or b_key not in state_dict:
            return
        w = state_dict[w_key]
        b = state_dict[b_key]
        if w.ndim != 2 or b.ndim != 1:
            return
        out_old, in_old = int(w.shape[0]), int(w.shape[1])
        out_new, in_new = 4, int(target_in_dim)
        if out_old == out_new and in_old == in_new and int(b.shape[0]) == out_new:
            return

        new_w = w.new_zeros((out_new, in_new))
        copy_out = min(out_old, out_new)
        copy_in = min(in_old, in_new)
        new_w[:copy_out, :copy_in] = w[:copy_out, :copy_in]

        if int(b.shape[0]) >= out_new:
            new_b = b[:out_new].clone()
        else:
            new_b = b.new_tensor([float(init_fx), float(init_fy), float(init_cx), float(init_cy)])

        if int(b.shape[0]) == 3:
            # Weak-cam legacy bias has incompatible semantics, reset to intrinsics prior.
            new_b = b.new_tensor([float(init_fx), float(init_fy), float(init_cx), float(init_cy)])

        state_dict[w_key] = new_w
        state_dict[b_key] = new_b

    _migrate_head("cam_head_left")
    _migrate_head("cam_head_right")
    return state_dict


def main() -> None:
    args = _parse_args()
    if args.mano_weakcam_loss_weight is not None:
        args.mano_persp_loss_weight = float(args.mano_weakcam_loss_weight)
    if args.mano_weakcam_reg_weight is not None:
        args.mano_persp_reg_weight = float(args.mano_weakcam_reg_weight)

    dataset_sources: list[dict[str, str]] | None = None
    if args.datasets_config.strip():
        try:
            dataset_sources = _load_datasets_config(args.datasets_config.strip())
        except ValueError as e:
            raise SystemExit(str(e)) from e

    if dataset_sources is None and not args.data_path.strip():
        raise SystemExit("--data-path is required unless --datasets-config is provided")

    if dataset_sources is not None and (args.episodes_file.strip() or args.episode_filter.strip()):
        raise SystemExit("When using --datasets-config, do not pass --episodes-file/--episode-filter (set per dataset in JSON)")

    if args.eval_only:
        if not args.resume.strip():
            raise SystemExit("--eval-only requires --resume")
        if not args.val_episodes_file.strip():
            raise SystemExit("--eval-only requires --val-episodes-file")
    use_dist, rank, world_size, local_rank = _dist_env()
    is_rank0 = rank == 0
    if use_dist:
        _prepare_ddp_master_addr(world_size)
        os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    set_seed(args.seed)

    try:
        mano_param_keys = _parse_mano_param_keys(args.mano_param_keys)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    mano_param_loss = cast(Literal["l1", "l2", "huber"], args.mano_param_loss)
    if args.mano_joint_fingertip_scale <= 0:
        raise SystemExit("--mano-joint-fingertip-scale must be > 0")
    if args.mano_joint_wrist_scale <= 0:
        raise SystemExit("--mano-joint-wrist-scale must be > 0")
    if args.mano_joint_chain_scale <= 0:
        raise SystemExit("--mano-joint-chain-scale must be > 0")
    mano_joint_preset = cast(Literal["uniform", "fingertip"], args.mano_joint_weight_preset)
    mano_joint_w21 = mano_joint_weight_vector_21(
        mano_joint_preset,
        args.mano_joint_fingertip_scale,
        args.mano_joint_wrist_scale,
        args.mano_joint_chain_scale,
        device=device,
        dtype=torch.float32,
    )
    shuffle_seed = args.seed if int(args.shuffle_seed) < 0 else int(args.shuffle_seed)
    shuffle_buffer_size = max(0, int(args.shuffle_buffer_size))
    train_shuffle_windows = (not args.no_shuffle_windows) and shuffle_buffer_size > 1
    train_aug_config: dict[str, object] = {
        "enable": bool(args.aug_enable),
        "color_temp_prob": float(args.aug_color_temp_prob),
        "color_temp_strength": float(args.aug_color_temp_strength),
        "contrast_prob": float(args.aug_contrast_prob),
        "contrast_min": float(args.aug_contrast_min),
        "contrast_max": float(args.aug_contrast_max),
        "saturation_prob": float(args.aug_saturation_prob),
        "saturation_min": float(args.aug_saturation_min),
        "saturation_max": float(args.aug_saturation_max),
        "grayscale_prob": float(args.aug_grayscale_prob),
        "scale_prob": float(args.aug_scale_prob),
        "scale_min": float(args.aug_scale_min),
        "scale_max": float(args.aug_scale_max),
        "scale_pad_mode": str(args.aug_scale_pad_mode),
        "update_existence_from_visibility": not bool(args.aug_no_update_existence),
        "invisible_joint_threshold": int(args.aug_invisible_joint_threshold),
        "visibility_joint_chunk": int(args.mano_joint_chunk),
    }
    eval_aug_config: dict[str, object] = {"enable": False}

    run_root = Path(args.run_dir.strip()).resolve() if args.run_dir.strip() else None
    if run_root is not None:
        run_root.mkdir(parents=True, exist_ok=True)

    tb_dir = args.tensorboard_dir.strip()
    ckpt_dir = args.checkpoint_dir.strip()
    log_dir = args.log_dir.strip()
    if run_root is not None:
        if not ckpt_dir:
            ckpt_dir = str(run_root / "checkpoints")
        if not log_dir:
            log_dir = str(run_root / "logs")

    log_path: Path | None = Path(log_dir) / "train.log" if log_dir else None
    if is_rank0:
        if run_root is not None:
            _save_args_json(run_root / "args.json", args)
        elif ckpt_dir:
            _save_args_json(Path(ckpt_dir) / "args.json", args)
        elif log_dir:
            _save_args_json(Path(log_dir) / "args.json", args)

    cfg = EgoHandSTConfig(
        pretrained_backbone=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
        image_size=384,
        use_mano_cross_decoder=not args.no_mano_cross_decoder,
        mano_decoder_depth=int(args.mano_decoder_depth),
        mano_decoder_heads=int(args.mano_decoder_heads),
        use_mano_temporal_refine=not args.no_mano_temporal_refine,
        mano_refine_hdim=int(args.mano_refine_hdim),
        mano_refine_layers=int(args.mano_refine_layers),
        mano_refine_heads=int(args.mano_refine_heads),
        use_hand_side_embedding=not args.no_hand_side_embedding,
        use_hand_role_mem_bias=not args.no_hand_role_mem_bias,
        camera_init_fx=float(args.camera_init_fx),
        camera_init_fy=float(args.camera_init_fy),
        camera_init_cx=float(args.camera_init_cx),
        camera_init_cy=float(args.camera_init_cy),
    )
    model = EgoHandSTModel(cfg).to(device)
    resume_path = args.resume.strip()
    if resume_path:
        sd = torch.load(resume_path, map_location=device)
        sd = _migrate_legacy_cam_head_state_dict(
            sd,
            init_fx=cfg.camera_init_fx,
            init_fy=cfg.camera_init_fy,
            init_cx=cfg.camera_init_cx,
            init_cy=cfg.camera_init_cy,
            target_in_dim=int(model.cam_head_left.in_features),
        )
        model.load_state_dict(sd, strict=True)

    if use_dist:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=not args.no_ddp_find_unused,
        )

    optim = None
    if not args.eval_only:
        optim = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    loader = None
    if not args.eval_only:
        has_episode_restriction = (
            _dataset_sources_has_episode_restriction(dataset_sources)
            if dataset_sources is not None
            else bool(args.episodes_file.strip())
        )
        ddp_read_all = bool(use_dist and world_size > 1 and has_episode_restriction)
        if args.ddp_shard_striping:
            ddp_read_all = False
        if args.ddp_read_all_shards:
            ddp_read_all = True
        if is_rank0 and use_dist and world_size > 1 and has_episode_restriction and ddp_read_all:
            log_line(
                "DDP: each rank reads all tar shards (non-empty --episodes-file). "
                "Pass --ddp-shard-striping to use per-rank shards only.",
                log_path,
            )
        try:
            loader = _build_loader(
                args.data_path,
                dataset_sources=dataset_sources,
                window_size=args.seq_len,
                stride=args.stride,
                shard_glob=args.shard_glob,
                batch_size=args.batch_size,
                workers=args.workers,
                pin_memory=device.type == "cuda",
                episodes_file=args.episodes_file,
                episode_filter=args.episode_filter,
                dist_rank=rank,
                dist_world_size=world_size,
                ddp_read_all_shards=ddp_read_all,
                shuffle=not args.no_shuffle_data,
                shuffle_windows=(not args.no_shuffle_windows) and (not args.no_shuffle_data),
                shuffle_buffer_size=max(0, int(args.shuffle_buffer_size)),
                shuffle_seed=args.seed if int(args.shuffle_seed) < 0 else int(args.shuffle_seed),
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e

    val_loader: DataLoader | None = None
    if is_rank0 and args.val_episodes_file.strip():
        if not args.data_path.strip():
            raise SystemExit("--val-episodes-file currently requires --data-path")
        val_loader = _build_loader(
            args.data_path,
            window_size=args.seq_len,
            stride=args.stride,
            shard_glob=args.shard_glob,
            batch_size=args.batch_size,
            workers=args.workers,
            pin_memory=device.type == "cuda",
            episodes_file=args.val_episodes_file,
            episode_filter=None,
            dist_rank=0,
            dist_world_size=1,
            shuffle=False,
            shuffle_windows=False,
            shuffle_buffer_size=0,
            shuffle_seed=args.seed if int(args.shuffle_seed) < 0 else int(args.shuffle_seed),
        )

    apply_left = bool(args.mano_left_root_fix)
    mano_pca_layers = None
    if not args.no_decode_hand_pca:
        try:
            mano_pca_layers = get_cached_mano_pca_decode_layers()
        except Exception as e:
            raise SystemExit(f"MANO decode failed: {e}") from e
        ml, mr = mano_pca_layers
        ml.to(device)
        mr.to(device)

    tb_writer = None
    if is_rank0 and tb_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise SystemExit("Missing tensorboard.") from e
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(os.path.abspath(tb_dir))
        log_line(f"TensorBoard: tensorboard --logdir {os.path.abspath(tb_dir)}", log_path)

    wandb_run = None
    wandb_enabled = bool(args.wandb) and str(args.wandb_mode).strip().lower() != "disabled"
    if is_rank0 and wandb_enabled:
        try:
            import wandb
        except ImportError as e:
            raise SystemExit("Missing wandb. Install it with `pip install wandb`.") from e
        wandb_dir = args.wandb_dir.strip()
        if not wandb_dir:
            wandb_dir = str(run_root) if run_root is not None else os.getcwd()
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_project = args.wandb_project.strip() or "egotransformer"
        wandb_name = args.wandb_name.strip()
        if not wandb_name and run_root is not None:
            wandb_name = run_root.name
        init_kwargs: dict[str, Any] = {
            "project": wandb_project,
            "config": dict(vars(args)),
            "dir": os.path.abspath(wandb_dir),
            "mode": str(args.wandb_mode).strip().lower(),
        }
        if args.wandb_entity.strip():
            init_kwargs["entity"] = args.wandb_entity.strip()
        if wandb_name:
            init_kwargs["name"] = wandb_name
        wandb_run = wandb.init(**init_kwargs)
        log_line(
            f"W&B: project={wandb_project!r} name={wandb_name or '<auto>'!r} "
            f"mode={init_kwargs['mode']!r} dir={os.path.abspath(wandb_dir)}",
            log_path,
        )

    if is_rank0 and ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    render_mano_every = max(0, int(args.render_mano_every))
    render_mano_dir: str | None = None
    if not args.eval_only and render_mano_every > 0:
        if mano_pca_layers is None:
            if is_rank0:
                log_line("--render-mano-every ignored (MANO decode off: use default decode or drop --no-decode-hand-pca)", log_path)
            render_mano_every = 0
        else:
            rd = args.render_mano_dir.strip()
            if rd:
                render_mano_dir = os.path.abspath(rd)
            elif run_root is not None:
                render_mano_dir = str(run_root / "render_mano")
            else:
                if is_rank0:
                    log_line(
                        "--render-mano-every ignored (no --run-dir or --render-mano-dir); use 0 explicitly to silence",
                        log_path,
                    )
                render_mano_every = 0
            if render_mano_every > 0 and render_mano_dir is not None and is_rank0:
                os.makedirs(render_mano_dir, exist_ok=True)

    show_progress = not args.no_progress
    max_steps = max(0, int(args.max_steps))
    save_every_steps = max(0, int(args.save_every_steps))
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    warmup_global_steps = max(0, int(args.warmup_global_steps))
    if is_rank0:
        if args.eval_only:
            log_line(
                f"eval-only  dist={use_dist} rank={rank}/{world_size} device={device}  "
                f"resume={resume_path!r}  val={args.val_episodes_file!r}  "
                f"batch={args.batch_size}  seq_len={args.seq_len}",
                log_path,
            )
        else:
            if save_every_steps > 0 and not ckpt_dir:
                log_line("save-every-steps ignored (set --checkpoint-dir or --run-dir)", log_path)
            log_line(
                f"start  dist={use_dist} rank={rank}/{world_size} device={device}  "
                f"epochs={args.epochs}  local_batch={args.batch_size}  "
                f"(global_batch≈{args.batch_size * world_size * grad_accum_steps}; accum={grad_accum_steps})  "
                f"warmup_global_steps={warmup_global_steps} (~{warmup_global_steps * grad_accum_steps} ministeps/rank)  "
                f"max_steps_per_rank={max_steps or 'inf'}  save_every_steps={save_every_steps or 'off'}  "
                f"shuffle_windows={train_shuffle_windows}  shuffle_buffer={shuffle_buffer_size}  "
                f"shuffle_seed={shuffle_seed}",
                log_path,
            )
            if dataset_sources is not None:
                log_line(f"datasets-config={args.datasets_config.strip()!r} count={len(dataset_sources)}", log_path)
                for i, src in enumerate(dataset_sources):
                    log_line(
                        f"dataset[{i}] name={src.get('name', '')!r} data_path={src.get('data_path', '')!r} "
                        f"shard_glob={src.get('shard_glob', args.shard_glob)!r} "
                        f"episodes_file={src.get('episodes_file', '')!r} episode_filter={src.get('episode_filter', '')!r}",
                        log_path,
                    )

    if args.eval_only:
        if is_rank0 and val_loader is None:
            raise SystemExit("eval-only: val_loader is None (set --val-episodes-file)")
        if val_loader is not None:
            vstats = eval_one_epoch(
                val_loader,
                device,
                model,
                image_size=cfg.image_size,
                apply_left_root_fix=apply_left,
                mano_pose_weight=args.mano_pose_weight,
                mano_root_orient_weight=args.mano_root_orient_weight,
                mano_param_loss=mano_param_loss,
                mano_huber_delta=args.mano_huber_delta,
                mano_param_loss_weight=args.mano_param_loss_weight,
                mano_joint_loss_weight=args.mano_joint_loss_weight,
                mano_joint_weight_21=mano_joint_w21,
                mano_joint_smooth_max_weight=args.mano_joint_smooth_max_weight,
                mano_joint_smooth_max_tau=args.mano_joint_smooth_max_tau,
                mano_joint_chunk=args.mano_joint_chunk,
                mano_param_keys=mano_param_keys,
                mano_pca_layers=mano_pca_layers,
                tb_writer=tb_writer,
                epoch=1,
                show_progress=show_progress,
                is_rank0=is_rank0,
                mano_persp_loss_weight=args.mano_persp_loss_weight,
                mano_persp_reg_weight=args.mano_persp_reg_weight,
                camera_init_fx=cfg.camera_init_fx,
                camera_init_fy=cfg.camera_init_fy,
                camera_init_cx=cfg.camera_init_cx,
                camera_init_cy=cfg.camera_init_cy,
                augment_config=eval_aug_config,
            )
            if is_rank0:
                log_line(
                    f"val   loss={vstats['loss']:.4f}  bce={vstats['bce']:.4f}  "
                    f"mano={vstats['mano']:.4f}",
                    log_path,
                )
        if use_dist:
            dist.barrier()
    else:
        global_step = 0
        optim_steps = 0
        ckpt_dir_or_none = ckpt_dir if ckpt_dir else None
        seff = save_every_steps if ckpt_dir_or_none else 0

        for ep in range(1, args.epochs + 1):
            if max_steps > 0 and optim_steps >= max_steps:
                break
            if loader is not None and hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(ep)
            stats, global_step, optim_steps, hit_max = train_one_epoch(
                loader,
                optim,
                device,
                model,
                image_size=cfg.image_size,
                apply_left_root_fix=apply_left,
                mano_pose_weight=args.mano_pose_weight,
                mano_root_orient_weight=args.mano_root_orient_weight,
                mano_param_loss=mano_param_loss,
                mano_huber_delta=args.mano_huber_delta,
                mano_param_loss_weight=args.mano_param_loss_weight,
                mano_joint_loss_weight=args.mano_joint_loss_weight,
                mano_joint_weight_21=mano_joint_w21,
                mano_joint_smooth_max_weight=args.mano_joint_smooth_max_weight,
                mano_joint_smooth_max_tau=args.mano_joint_smooth_max_tau,
                mano_joint_chunk=args.mano_joint_chunk,
                mano_param_keys=mano_param_keys,
                mano_pca_layers=mano_pca_layers,
                grad_clip=args.grad_clip,
                grad_accum_steps=grad_accum_steps,
                base_lr=float(args.lr),
                warmup_global_steps=warmup_global_steps,
                tb_writer=tb_writer,
                wandb_run=wandb_run,
                global_step=global_step,
                optim_steps=optim_steps,
                epoch=ep,
                epochs_total=args.epochs,
                show_progress=show_progress,
                max_steps=max_steps,
                save_every_steps=seff,
                ckpt_dir=ckpt_dir_or_none,
                use_dist=use_dist,
                is_rank0=is_rank0,
                render_mano_every=render_mano_every,
                render_mano_dir=render_mano_dir,
                mano_persp_loss_weight=args.mano_persp_loss_weight,
                mano_persp_reg_weight=args.mano_persp_reg_weight,
                camera_init_fx=cfg.camera_init_fx,
                camera_init_fy=cfg.camera_init_fy,
                camera_init_cx=cfg.camera_init_cx,
                camera_init_cy=cfg.camera_init_cy,
                augment_config=train_aug_config,
            )
            if is_rank0:
                summary = (
                    f"epoch {ep}/{args.epochs}  optim_steps={optim_steps}  loss={stats['loss']:.4f}  "
                    f"bce={stats['bce']:.4f}  mano={stats['mano']:.4f}"
                )
                log_line(summary, log_path)
            if ckpt_dir and is_rank0:
                sd = module_state_dict(model)
                cpath = os.path.join(ckpt_dir, f"epoch_{ep:04d}.pt")
                torch.save(sd, cpath)
                torch.save(sd, os.path.join(ckpt_dir, "latest.pt"))
            if use_dist:
                dist.barrier()

            if val_loader is not None:
                vstats = eval_one_epoch(
                    val_loader,
                    device,
                    model,
                    image_size=cfg.image_size,
                    apply_left_root_fix=apply_left,
                    mano_pose_weight=args.mano_pose_weight,
                    mano_root_orient_weight=args.mano_root_orient_weight,
                    mano_param_loss=mano_param_loss,
                    mano_huber_delta=args.mano_huber_delta,
                    mano_param_loss_weight=args.mano_param_loss_weight,
                    mano_joint_loss_weight=args.mano_joint_loss_weight,
                    mano_joint_weight_21=mano_joint_w21,
                    mano_joint_smooth_max_weight=args.mano_joint_smooth_max_weight,
                    mano_joint_smooth_max_tau=args.mano_joint_smooth_max_tau,
                    mano_joint_chunk=args.mano_joint_chunk,
                    mano_param_keys=mano_param_keys,
                    mano_pca_layers=mano_pca_layers,
                    tb_writer=tb_writer,
                    wandb_run=wandb_run,
                    epoch=ep,
                    log_step=global_step,
                    show_progress=show_progress,
                    is_rank0=is_rank0,
                    mano_persp_loss_weight=args.mano_persp_loss_weight,
                    mano_persp_reg_weight=args.mano_persp_reg_weight,
                    camera_init_fx=cfg.camera_init_fx,
                    camera_init_fy=cfg.camera_init_fy,
                    camera_init_cx=cfg.camera_init_cx,
                    camera_init_cy=cfg.camera_init_cy,
                    augment_config=eval_aug_config,
                )
                if is_rank0:
                    log_line(
                        f"val   ep{ep}  loss={vstats['loss']:.4f}  bce={vstats['bce']:.4f}  "
                        f"mano={vstats['mano']:.4f}",
                        log_path,
                    )
            if use_dist:
                dist.barrier()

            if hit_max and is_rank0:
                log_line(f"stopped: max-steps reached ({max_steps})", log_path)
            if hit_max:
                break

    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    if is_rank0 and args.save:
        save_path = os.path.abspath(args.save)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(module_state_dict(model), save_path)
        log_line(f"saved final: {save_path}", log_path)

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
