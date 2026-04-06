#!/usr/bin/env python3
"""Train EgoHandSTModel (WebDataset tar shards).

Usage:
python train.py \
  --data-path /share_data/zhangtingrui/datasets/taco_v2 \
  --episodes-file splits/train.txt

torchrun --standalone --nproc_per_node=8 train.py \
  --data-path /share_data/zhangtingrui/datasets/oakink2_v4 \
  --episodes-file splits/oakink_full/train.txt \
  --run-dir runs/my_exp1 \
  --tensorboard-dir runs/my_exp1/tb \
  --save-every-steps 100


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
from typing import Literal, cast

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
from training.losses import MANO_PARAM_KEY_CHOICES
from training.train_loop import eval_one_epoch, train_one_epoch


def _parse_mano_param_keys(text: str) -> frozenset[str]:
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("--mano-param-keys must list at least one of: trans, root_orient, hand_pose, betas")
    bad = [p for p in parts if p not in MANO_PARAM_KEY_CHOICES]
    if bad:
        raise ValueError(f"unknown --mano-param-keys entries {bad}; allowed: {sorted(MANO_PARAM_KEY_CHOICES)}")
    return frozenset(parts)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EgoHandSTModel")

    d = p.add_argument_group("data")
    d.add_argument("--data-path", type=str, required=True, help="tar dir, file, or glob")
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
    d.add_argument("--seq-len", type=int, default=32, help="window length T")
    d.add_argument("--stride", type=int, default=1)
    d.add_argument("--batch-size", type=int, default=32, help="per-GPU batch (global ~ batch * num GPUs)")
    d.add_argument("--workers", type=int, default=0)
    d.add_argument(
        "--no-shuffle-windows",
        action="store_true",
        help="disable training window shuffle; by default shuffle is enabled",
    )
    d.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=DEFAULT_WINDOW_SHUFFLE_BUFFER_SIZE,
        help="training only: streaming window-shuffle buffer size when shuffle is enabled; 0/1 keeps sequential order",
    )
    d.add_argument(
        "--shuffle-seed",
        type=int,
        default=-1,
        help="training only: base seed for window shuffle; default uses --seed",
    )

    o = p.add_argument_group("optim")
    o.add_argument("--epochs", type=int, default=5)
    o.add_argument("--lr", type=float, default=3e-4)
    o.add_argument("--weight-decay", type=float, default=0.01)
    o.add_argument("--grad-clip", type=float, default=0.0)
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
        default=1.0,
        help="relative weight for hand_pose inside param regression when hand_pose is in --mano-param-keys",
    )
    m.add_argument(
        "--mano-root-orient-weight",
        type=float,
        default=2.0,
        help="relative weight for root_orient inside param regression when root_orient is in --mano-param-keys "
        "(default 2.0 vs 1.0 for trans/betas)",
    )
    m.add_argument("--mano-no-left-root-fix", action="store_true")
    m.add_argument(
        "--mano-param-keys",
        type=str,
        default="trans,root_orient,hand_pose,betas",
        help="comma-separated subset of trans,root_orient,hand_pose,betas for vector regression "
        "(default: trans,root_orient,hand_pose,betas)",
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
        default=0.5,
        help="scale for vector regression on --mano-param-keys; 0 disables",
    )
    m.add_argument(
        "--mano-joint-loss-weight",
        type=float,
        default=2.0,
        help="scale for 3D joint MSE (m^2); 0 disables; needs PCA decode",
    )
    m.add_argument(
        "--mano-vert-loss-weight",
        type=float,
        default=1.0,
        help="scale for 3D vertex MSE (m^2); 0 disables; needs PCA decode",
    )
    m.add_argument(
        "--mano-bone-loss-weight",
        type=float,
        default=1.0,
        help="scale for bone-length MSE (m^2): 20 parent-child edges in 21-joint MANO order; "
        "0 disables; needs PCA decode",
    )
    m.add_argument(
        "--mano-bone-dir-loss-weight",
        type=float,
        default=1.0,
        help="scale for bone unit-direction MSE (20 edges, dimensionless); 0 disables; needs PCA decode",
    )
    m.add_argument(
        "--mano-hand-pca-loss-weight",
        type=float,
        default=0.0,
        help="scale for hand_pose PCA coefficient MSE (axis-angle -> layer PCA vs GT); default 0 (off); "
        "set e.g. 0.25 to enable",
    )
    m.add_argument(
        "--mano-joint-chunk",
        type=int,
        default=512,
        help="sub-batch size for ManoLayer joint/vertex forward passes",
    )

    mo = p.add_argument_group("model")
    mo.add_argument("--no-pretrained", action="store_true")
    mo.add_argument("--unfreeze-backbone", action="store_true")

    io = p.add_argument_group("checkpoint & log")
    io.add_argument("--save", type=str, default="", help="extra final .pt path")
    io.add_argument("--checkpoint-dir", type=str, default="")
    io.add_argument("--save-every-steps", type=int, default=0)
    io.add_argument("--resume", type=str, default="")
    io.add_argument("--tensorboard-dir", type=str, default="")
    io.add_argument("--log-dir", type=str, default="")
    io.add_argument("--run-dir", type=str, default="", help="sets default checkpoints/ and logs/")
    io.add_argument(
        "--render-mano-every",
        type=int,
        default=10,
        help="every N optimizer steps (rank0): save GT|Pred MANO skeleton PNG under <run-dir>/render_mano; "
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
    shuffle_windows: bool = DEFAULT_WINDOW_SHUFFLE_WINDOWS,
    shuffle_buffer_size: int = DEFAULT_WINDOW_SHUFFLE_BUFFER_SIZE,
    shuffle_seed: int = DEFAULT_WINDOW_SHUFFLE_SEED,
) -> DataLoader:
    ef = episodes_file.strip()
    sf = (episode_filter or "").strip() or None
    if ef and sf:
        raise ValueError("use either --episodes-file or --episode-filter, not both")
    return EpisodeWindowDataLoader(
        data_path,
        window_size=window_size,
        stride=stride,
        shard_glob=shard_glob,
        episode_list_file=ef or None,
        episode_filter=sf,
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
        ddp_read_all_shards=ddp_read_all_shards,
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


def main() -> None:
    args = _parse_args()
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
    shuffle_seed = args.seed if int(args.shuffle_seed) < 0 else int(args.shuffle_seed)
    shuffle_buffer_size = max(0, int(args.shuffle_buffer_size))
    train_shuffle_windows = (not args.no_shuffle_windows) and shuffle_buffer_size > 1

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
        image_size=224,
    )
    model = EgoHandSTModel(cfg).to(device)
    resume_path = args.resume.strip()
    if resume_path:
        sd = torch.load(resume_path, map_location=device)
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
        ddp_read_all = _ddp_read_all_shards_effective(args, use_dist, world_size)
        if is_rank0 and use_dist and world_size > 1 and args.episodes_file.strip() and ddp_read_all:
            log_line(
                "DDP: each rank reads all tar shards (non-empty --episodes-file). "
                "Pass --ddp-shard-striping to use per-rank shards only.",
                log_path,
            )
        try:
            loader = _build_loader(
                args.data_path,
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
                shuffle_windows=train_shuffle_windows,
                shuffle_buffer_size=shuffle_buffer_size,
                shuffle_seed=shuffle_seed,
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e

    val_loader: DataLoader | None = None
    if is_rank0 and args.val_episodes_file.strip():
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
            shuffle_windows=False,
            shuffle_buffer_size=0,
            shuffle_seed=shuffle_seed,
        )

    apply_left = not args.mano_no_left_root_fix
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
                f"(global_batch≈{args.batch_size * world_size})  "
                f"max_steps_per_rank={max_steps or 'inf'}  save_every_steps={save_every_steps or 'off'}  "
                f"shuffle_windows={train_shuffle_windows}  shuffle_buffer={shuffle_buffer_size}  "
                f"shuffle_seed={shuffle_seed}",
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
                mano_vert_loss_weight=args.mano_vert_loss_weight,
                mano_bone_loss_weight=args.mano_bone_loss_weight,
                mano_bone_dir_loss_weight=args.mano_bone_dir_loss_weight,
                mano_joint_chunk=args.mano_joint_chunk,
                mano_param_keys=mano_param_keys,
                mano_hand_pose_pca_loss_weight=args.mano_hand_pca_loss_weight,
                mano_pca_layers=mano_pca_layers,
                tb_writer=tb_writer,
                epoch=1,
                show_progress=show_progress,
                is_rank0=is_rank0,
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
                mano_vert_loss_weight=args.mano_vert_loss_weight,
                mano_bone_loss_weight=args.mano_bone_loss_weight,
                mano_bone_dir_loss_weight=args.mano_bone_dir_loss_weight,
                mano_joint_chunk=args.mano_joint_chunk,
                mano_param_keys=mano_param_keys,
                mano_hand_pose_pca_loss_weight=args.mano_hand_pca_loss_weight,
                mano_pca_layers=mano_pca_layers,
                grad_clip=args.grad_clip,
                tb_writer=tb_writer,
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
                    mano_vert_loss_weight=args.mano_vert_loss_weight,
                    mano_bone_loss_weight=args.mano_bone_loss_weight,
                    mano_bone_dir_loss_weight=args.mano_bone_dir_loss_weight,
                    mano_joint_chunk=args.mano_joint_chunk,
                    mano_param_keys=mano_param_keys,
                    mano_hand_pose_pca_loss_weight=args.mano_hand_pca_loss_weight,
                    mano_pca_layers=mano_pca_layers,
                    tb_writer=tb_writer,
                    epoch=ep,
                    show_progress=show_progress,
                    is_rank0=is_rank0,
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
