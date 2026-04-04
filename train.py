#!/usr/bin/env python3
"""训练入口：EpisodeWindowDataLoader（tar 按 rank 划分 shard）。

Usage:
python train.py \
  --data-path /share_data/zhangtingrui/datasets/taco_v2 \
  --episodes-file splits/train.txt

torchrun --standalone --nproc_per_node=8 train.py \
  --data-path /share_data/zhangtingrui/datasets/taco_v2 \
  --episodes-file splits/train.txt \
  --run-dir runs/my_exp1 \
  --tensorboard-dir runs/my_exp1/tb


"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from dataloader import EpisodeWindowDataLoader, lowdim_wrist_to_mano_cam
from dataloader.mano_pca import get_cached_mano_pca_decode_layers, hand_pca_bt_to_axisang45
from egotransformer.model import EgoHandSTConfig, EgoHandSTModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EgoHandSTModel")

    d = p.add_argument_group("data")
    d.add_argument("--data-path", type=str, required=True, help="tar 目录、单文件或 glob")
    d.add_argument("--shard-glob", type=str, default="*.tar")
    d.add_argument(
        "--episodes-file",
        type=str,
        default="",
        help="训练集 episode 白名单，每行一个 episode_name（与数据内 normalize 后一致）；空=全部",
    )
    d.add_argument(
        "--episode-filter",
        type=str,
        default="",
        help="只训单条 episode（调试）；与 --episodes-file 互斥",
    )
    d.add_argument("--val-episodes-file", type=str, default="", help="验证集列表；空=不跑 val")
    d.add_argument("--seq-len", type=int, default=32, help="窗口长度 T")
    d.add_argument("--stride", type=int, default=1)
    d.add_argument("--batch-size", type=int, default=32, help="每 GPU 的 batch；DDP 时总 batch 约乘 GPU 数")
    d.add_argument("--workers", type=int, default=0)

    o = p.add_argument_group("optim")
    o.add_argument("--epochs", type=int, default=20)
    o.add_argument("--lr", type=float, default=3e-4)
    o.add_argument("--weight-decay", type=float, default=0.01)
    o.add_argument("--grad-clip", type=float, default=0.0)
    o.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="每 rank 优化步上限；任 rank 达到后全局停；0=只按 epoch",
    )
    o.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    m = p.add_argument_group("mano")
    m.add_argument("--no-decode-hand-pca", action="store_true")
    m.add_argument("--mano-pose-weight", type=float, default=0.0)
    m.add_argument("--mano-no-left-root-fix", action="store_true")

    mo = p.add_argument_group("model")
    mo.add_argument("--no-pretrained", action="store_true")
    mo.add_argument("--unfreeze-backbone", action="store_true")

    io = p.add_argument_group("checkpoint & log")
    io.add_argument("--save", type=str, default="", help="结束时额外写一份 .pt")
    io.add_argument("--checkpoint-dir", type=str, default="")
    io.add_argument("--save-every-steps", type=int, default=0)
    io.add_argument("--resume", type=str, default="")
    io.add_argument("--tensorboard-dir", type=str, default="")
    io.add_argument("--log-dir", type=str, default="")
    io.add_argument("--run-dir", type=str, default="", help="默认 checkpoints/ 与 logs/")

    r = p.add_argument_group("run")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def _dist_env() -> tuple[bool, int, int, int]:
    """WORLD_SIZE>1 且已设置 RANK 时启用 DDP（通常由 torchrun 注入）。"""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1:
        return True, int(os.environ["RANK"]), ws, int(os.environ.get("LOCAL_RANK", "0"))
    return False, 0, 1, 0


def _prepare_ddp_master_addr(world_size: int) -> None:
    """单机多卡时 hostname 常只解析到 IPv6，c10d 报 errno 97；默认改连 127.0.0.1。"""
    if os.environ.get("MASTER_ADDR_USE_HOSTNAME", "").strip().lower() in ("1", "true", "yes"):
        return
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size == local_ws:
        os.environ["MASTER_ADDR"] = "127.0.0.1"


def _unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m


def _state_dict(m: nn.Module) -> dict[str, Any]:
    return _unwrap_model(m).state_dict()


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


def _log_line(msg: str, log_file: Path | None) -> None:
    print(msg, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(f"{ts}\t{msg}\n")


def _to_float_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.float().to(device, non_blocking=True)
    else:
        t = torch.from_numpy(x).float().to(device, non_blocking=True)
    return t


def wds_batch_to_training_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    image_size: int,
    image_scale: float = 1.0 / 255.0,
    apply_left_root_fix: bool = True,
    mano_pca_layers: tuple[torch.nn.Module, torch.nn.Module] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Collate 后的 batch → video、existence、相机系 mano 目标（trans/root/beta 来自 lowdim 换算）。"""
    video = _to_float_tensor(batch["video"], device)
    if video.dim() == 5 and video.max() > 1.5:
        video = video * image_scale
    b, t, c, h, w = video.shape
    if (h, w) != (image_size, image_size):
        video = F.interpolate(
            video.flatten(0, 1),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).view(b, t, c, image_size, image_size)

    existence = _to_float_tensor(batch["existence"], device)
    e4 = _to_float_tensor(batch["extrinsic_4x4"], device)

    hand_l_fill = None
    hand_r_fill = None
    if mano_pca_layers is not None:
        ml, mr = mano_pca_layers
        hand_l_fill = hand_pca_bt_to_axisang45(
            _to_float_tensor(batch["left_hand_pose45"], device), ml
        )
        hand_r_fill = hand_pca_bt_to_axisang45(
            _to_float_tensor(batch["right_hand_pose45"], device), mr
        )

    mano_l = lowdim_wrist_to_mano_cam(
        _to_float_tensor(batch["left_translation"], device),
        _to_float_tensor(batch["left_rot6"], device),
        e4,
        _to_float_tensor(batch["left_shape"], device),
        is_left=True,
        apply_left_root_fix=apply_left_root_fix,
        hand_pose_fill=hand_l_fill,
    )
    mano_r = lowdim_wrist_to_mano_cam(
        _to_float_tensor(batch["right_translation"], device),
        _to_float_tensor(batch["right_rot6"], device),
        e4,
        _to_float_tensor(batch["right_shape"], device),
        is_left=False,
        apply_left_root_fix=True,
        hand_pose_fill=hand_r_fill,
    )
    return video, existence, mano_l, mano_r


def bce_existence(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def mano_regression_loss(
    pred: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    mask_bt: torch.Tensor,
    *,
    hand_pose_weight: float = 0.0,
) -> torch.Tensor:
    if mask_bt.sum() < 1e-6:
        return pred["trans"].new_tensor(0.0)
    m = mask_bt.unsqueeze(-1)
    items = (
        ("trans", 1.0),
        ("root_orient", 1.0),
        ("hand_pose", hand_pose_weight),
        ("betas", 1.0),
    )
    acc = 0.0
    w_sum = 0.0
    for k, w in items:
        if w <= 0.0:
            continue
        diff = (pred[k] - tgt[k]).abs() * m
        denom = m.expand_as(diff).sum().clamp_min(1.0)
        acc += w * (diff.sum() / denom)
        w_sum += w
    return acc / w_sum if w_sum > 0.0 else pred["trans"].new_tensor(0.0)


def _maybe_save_step_checkpoint(
    ckpt_dir: str | None, optim_steps: int, model: nn.Module, *, is_rank0: bool
) -> None:
    if not ckpt_dir or not is_rank0:
        return
    sd = _state_dict(model)
    path = os.path.join(ckpt_dir, f"step_{optim_steps:07d}.pt")
    torch.save(sd, path)
    torch.save(sd, os.path.join(ckpt_dir, "latest.pt"))


def train_one_epoch(
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    model: nn.Module,
    *,
    image_size: int,
    apply_left_root_fix: bool,
    mano_pose_weight: float,
    mano_pca_layers: tuple[torch.nn.Module, torch.nn.Module] | None,
    grad_clip: float,
    tb_writer: Any | None,
    global_step: int,
    optim_steps: int,
    epoch: int,
    epochs_total: int,
    show_progress: bool,
    max_steps: int,
    save_every_steps: int,
    ckpt_dir: str | None,
    use_dist: bool,
    is_rank0: bool,
) -> tuple[dict[str, float], int, int, bool]:
    """返回 (epoch 均值统计, tb_global_step, optim_steps, 是否因 max_steps 提前结束训练)."""
    model.train()
    tot = {"loss": 0.0, "bce": 0.0, "mano": 0.0}
    n = 0
    hit_max = False
    pbar = tqdm(
        loader,
        desc=f"train {epoch}/{epochs_total}",
        leave=True,
        dynamic_ncols=True,
        disable=not (show_progress and is_rank0),
    )
    for batch in pbar:
        video, exist_tgt, mano_l_tgt, mano_r_tgt = wds_batch_to_training_batch(
            batch,
            device=device,
            image_size=image_size,
            apply_left_root_fix=apply_left_root_fix,
            mano_pca_layers=mano_pca_layers,
        )

        optim.zero_grad(set_to_none=True)
        out = model(video)
        loss_b = bce_existence(out["hand_existence_logits"], exist_tgt)

        mask_l = exist_tgt[..., 0]
        mask_r = exist_tgt[..., 1]
        loss_m = mano_regression_loss(
            out["mano_left"], mano_l_tgt, mask_l, hand_pose_weight=mano_pose_weight
        ) + mano_regression_loss(
            out["mano_right"], mano_r_tgt, mask_r, hand_pose_weight=mano_pose_weight
        )

        loss = loss_b + loss_m
        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        optim_steps += 1

        tot["loss"] += float(loss.detach())
        tot["bce"] += float(loss_b.detach())
        tot["mano"] += float(loss_m.detach())
        n += 1
        pbar.set_postfix(
            step=str(optim_steps),
            loss=f"{float(loss.detach()):.4f}",
            bce=f"{float(loss_b.detach()):.4f}",
            mano=f"{float(loss_m.detach()):.4f}",
        )
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", float(loss.detach()), global_step)
            tb_writer.add_scalar("train/bce", float(loss_b.detach()), global_step)
            tb_writer.add_scalar("train/mano_l1", float(loss_m.detach()), global_step)
            tb_writer.add_scalar("train/optim_step", float(optim_steps), global_step)
            global_step += 1

        if save_every_steps > 0 and optim_steps % save_every_steps == 0:
            _maybe_save_step_checkpoint(ckpt_dir, optim_steps, model, is_rank0=is_rank0)

        if max_steps > 0 and optim_steps >= max_steps:
            hit_max = True
            break

    t = torch.tensor(
        [tot["loss"], tot["bce"], tot["mano"], float(n)],
        device=device,
        dtype=torch.float64,
    )
    if use_dist:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    nn_ = max(int(t[3].item()), 1)
    avg = {"loss": t[0].item() / nn_, "bce": t[1].item() / nn_, "mano": t[2].item() / nn_}
    if tb_writer is not None:
        tb_writer.add_scalar("epoch/loss", avg["loss"], epoch)
        tb_writer.add_scalar("epoch/bce", avg["bce"], epoch)
        tb_writer.add_scalar("epoch/mano_l1", avg["mano"], epoch)
    hm = torch.tensor([int(hit_max)], device=device, dtype=torch.int32)
    if use_dist:
        dist.all_reduce(hm, op=dist.ReduceOp.MAX)
    return avg, global_step, optim_steps, bool(hm.item())


@torch.no_grad()
def eval_one_epoch(
    loader: DataLoader,
    device: torch.device,
    model: nn.Module,
    *,
    image_size: int,
    apply_left_root_fix: bool,
    mano_pose_weight: float,
    mano_pca_layers: tuple[torch.nn.Module, torch.nn.Module] | None,
    tb_writer: Any | None,
    epoch: int,
    show_progress: bool,
    is_rank0: bool,
) -> dict[str, float]:
    model.eval()
    tot = {"loss": 0.0, "bce": 0.0, "mano": 0.0}
    n = 0
    pbar = tqdm(
        loader,
        desc=f"val ep{epoch}",
        leave=True,
        dynamic_ncols=True,
        disable=not (show_progress and is_rank0),
    )
    for batch in pbar:
        video, exist_tgt, mano_l_tgt, mano_r_tgt = wds_batch_to_training_batch(
            batch,
            device=device,
            image_size=image_size,
            apply_left_root_fix=apply_left_root_fix,
            mano_pca_layers=mano_pca_layers,
        )
        out = model(video)
        loss_b = bce_existence(out["hand_existence_logits"], exist_tgt)
        mask_l = exist_tgt[..., 0]
        mask_r = exist_tgt[..., 1]
        loss_m = mano_regression_loss(
            out["mano_left"], mano_l_tgt, mask_l, hand_pose_weight=mano_pose_weight
        ) + mano_regression_loss(
            out["mano_right"], mano_r_tgt, mask_r, hand_pose_weight=mano_pose_weight
        )
        loss = loss_b + loss_m
        tot["loss"] += float(loss)
        tot["bce"] += float(loss_b)
        tot["mano"] += float(loss_m)
        n += 1
        pbar.set_postfix(
            loss=f"{float(loss):.4f}",
            bce=f"{float(loss_b):.4f}",
            mano=f"{float(loss_m):.4f}",
        )
    nn_ = max(n, 1)
    avg = {k: tot[k] / nn_ for k in tot}
    if tb_writer is not None:
        tb_writer.add_scalar("val/loss", avg["loss"], epoch)
        tb_writer.add_scalar("val/bce", avg["bce"], epoch)
        tb_writer.add_scalar("val/mano_l1", avg["mano"], epoch)
    return avg


def main() -> None:
    args = _parse_args()
    use_dist, rank, world_size, local_rank = _dist_env()
    is_rank0 = rank == 0
    if use_dist:
        _prepare_ddp_master_addr(world_size)
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    set_seed(args.seed)

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
            find_unused_parameters=True,
        )

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
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
        )

    apply_left = not args.mano_no_left_root_fix
    mano_pca_layers = None
    if not args.no_decode_hand_pca:
        try:
            mano_pca_layers = get_cached_mano_pca_decode_layers()
        except Exception as e:
            raise SystemExit(f"MANO decode failed: {e}") from e

    tb_writer = None
    if is_rank0 and tb_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise SystemExit("Missing tensorboard.") from e
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(os.path.abspath(tb_dir))
        _log_line(f"TensorBoard: tensorboard --logdir {os.path.abspath(tb_dir)}", log_path)

    if is_rank0 and ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    show_progress = not args.no_progress
    max_steps = max(0, int(args.max_steps))
    save_every_steps = max(0, int(args.save_every_steps))
    if is_rank0:
        if save_every_steps > 0 and not ckpt_dir:
            _log_line("save-every-steps ignored (set --checkpoint-dir or --run-dir)", log_path)
        _log_line(
            f"start  dist={use_dist} rank={rank}/{world_size} device={device}  "
            f"epochs={args.epochs}  local_batch={args.batch_size}  "
            f"(global_batch≈{args.batch_size * world_size})  "
            f"max_steps_per_rank={max_steps or 'inf'}  save_every_steps={save_every_steps or 'off'}",
            log_path,
        )

    global_step = 0
    optim_steps = 0
    ckpt_dir_or_none = ckpt_dir if ckpt_dir else None
    seff = save_every_steps if ckpt_dir_or_none else 0

    for ep in range(1, args.epochs + 1):
        if max_steps > 0 and optim_steps >= max_steps:
            break
        stats, global_step, optim_steps, hit_max = train_one_epoch(
            loader,
            optim,
            device,
            model,
            image_size=cfg.image_size,
            apply_left_root_fix=apply_left,
            mano_pose_weight=args.mano_pose_weight,
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
        )
        if is_rank0:
            summary = (
                f"epoch {ep}/{args.epochs}  optim_steps={optim_steps}  loss={stats['loss']:.4f}  "
                f"bce={stats['bce']:.4f}  mano_l1={stats['mano']:.4f}"
            )
            _log_line(summary, log_path)
        if ckpt_dir and is_rank0:
            sd = _state_dict(model)
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
                mano_pca_layers=mano_pca_layers,
                tb_writer=tb_writer,
                epoch=ep,
                show_progress=show_progress,
                is_rank0=is_rank0,
            )
            if is_rank0:
                _log_line(
                    f"val   ep{ep}  loss={vstats['loss']:.4f}  bce={vstats['bce']:.4f}  "
                    f"mano_l1={vstats['mano']:.4f}",
                    log_path,
                )
        if use_dist:
            dist.barrier()

        if hit_max and is_rank0:
            _log_line(f"stopped: max-steps reached ({max_steps})", log_path)
        if hit_max:
            break

    if tb_writer is not None:
        tb_writer.close()

    if is_rank0 and args.save:
        save_path = os.path.abspath(args.save)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(_state_dict(model), save_path)
        _log_line(f"saved final: {save_path}", log_path)

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
