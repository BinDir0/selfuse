#!/usr/bin/env python3
"""训练入口：使用 dataloader EpisodeWindowDataLoader（tar shards）。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import EpisodeWindowDataLoader, lowdim_wrist_to_mano_cam
from egotransformer.model import EgoHandSTConfig, EgoHandSTModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EgoHandSTModel")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=8, help="窗口长度 T（EpisodeWindow window_size）")
    p.add_argument("--stride", type=int, default=1, help="Episode 窗口 stride")
    p.add_argument("--data-path", type=str, required=True, help="tar 目录、单个 .tar 或 glob")
    p.add_argument("--shard-glob", type=str, default="*.tar")
    p.add_argument("--episode-filter", type=str, default="", help="只保留该 episode_name；空则不过滤")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--save", type=str, default="", help="保存 state_dict 路径")
    p.add_argument("--no-pretrained", action="store_true", help="骨干随机初始化")
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mano-pose-weight",
        type=float,
        default=0.0,
        help="hand_pose 监督权重；lowdim 无轴角 GT 时保持 0，拟合 finger 后再调大",
    )
    p.add_argument(
        "--mano-no-left-root-fix",
        action="store_true",
        help="不做左手 diag(-1,1,1) root 修正（若与可视化/manopth 不一致可关掉试）",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    mano_l = lowdim_wrist_to_mano_cam(
        _to_float_tensor(batch["left_translation"], device),
        _to_float_tensor(batch["left_rot6"], device),
        e4,
        _to_float_tensor(batch["left_shape"], device),
        is_left=True,
        apply_left_root_fix=apply_left_root_fix,
        hand_pose_fill=None,
    )
    mano_r = lowdim_wrist_to_mano_cam(
        _to_float_tensor(batch["right_translation"], device),
        _to_float_tensor(batch["right_rot6"], device),
        e4,
        _to_float_tensor(batch["right_shape"], device),
        is_left=False,
        apply_left_root_fix=True,
        hand_pose_fill=None,
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


def train_one_epoch(
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    model: EgoHandSTModel,
    *,
    image_size: int,
    apply_left_root_fix: bool,
    mano_pose_weight: float,
) -> dict[str, float]:
    model.train()
    tot = {"loss": 0.0, "bce": 0.0, "mano": 0.0}
    n = 0
    for batch in loader:
        video, exist_tgt, mano_l_tgt, mano_r_tgt = wds_batch_to_training_batch(
            batch,
            device=device,
            image_size=image_size,
            apply_left_root_fix=apply_left_root_fix,
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
        optim.step()

        tot["loss"] += float(loss.detach())
        tot["bce"] += float(loss_b.detach())
        tot["mano"] += float(loss_m.detach())
        n += 1
    for k in tot:
        tot[k] /= max(n, 1)
    return tot


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    cfg = EgoHandSTConfig(
        pretrained_backbone=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
        image_size=224,
    )
    model = EgoHandSTModel(cfg).to(device)

    filt = args.episode_filter.strip() or None
    loader = EpisodeWindowDataLoader(
        args.data_path,
        window_size=args.seq_len,
        stride=args.stride,
        shard_glob=args.shard_glob,
        episode_filter=filt,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    apply_left = not args.mano_no_left_root_fix
    for ep in range(1, args.epochs + 1):
        stats = train_one_epoch(
            loader,
            optim,
            device,
            model,
            image_size=cfg.image_size,
            apply_left_root_fix=apply_left,
            mano_pose_weight=args.mano_pose_weight,
        )
        print(
            f"epoch {ep}/{args.epochs}  loss={stats['loss']:.4f}  bce={stats['bce']:.4f}  mano_l1={stats['mano']:.4f}"
        )

    if args.save:
        save_path = os.path.abspath(args.save)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
