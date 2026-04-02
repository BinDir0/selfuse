#!/usr/bin/env python3
"""训练入口。batch 与 datasets.collate_hand_batch 一致：video (B,T,3,H,W)，existence (B,T,2)，MANO 各键 (B,T,·)。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from egotransformer.datasets import DummyVideoHandDataset, collate_hand_batch
from egotransformer.model import EgoHandSTConfig, EgoHandSTModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EgoHandSTModel")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=8, help="每段视频采样帧数 T")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--save", type=str, default="", help="保存 state_dict 路径")
    p.add_argument("--no-pretrained", action="store_true", help="骨干随机初始化")
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bce_existence(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def mano_regression_loss(
    pred: dict[str, torch.Tensor],
    tgt: dict[str, torch.Tensor],
    mask_bt: torch.Tensor,
) -> torch.Tensor:
    """mask_bt: (B,T)，0/1，无监督位置权重为 0。"""
    if mask_bt.sum() < 1e-6:
        return pred["trans"].new_tensor(0.0)
    m = mask_bt.unsqueeze(-1)
    keys = ("trans", "root_orient", "hand_pose", "betas")
    acc = 0.0
    for k in keys:
        diff = (pred[k] - tgt[k]).abs() * m
        denom = m.expand_as(diff).sum().clamp_min(1.0)
        acc += diff.sum() / denom
    return acc / len(keys)


def train_one_epoch(
    model: EgoHandSTModel,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    tot = {"loss": 0.0, "bce": 0.0, "mano": 0.0}
    n = 0
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        exist_tgt = batch["existence"].to(device, non_blocking=True)
        mano_l_tgt = {k: v.to(device, non_blocking=True) for k, v in batch["mano_left"].items()}
        mano_r_tgt = {k: v.to(device, non_blocking=True) for k, v in batch["mano_right"].items()}

        optim.zero_grad(set_to_none=True)
        out = model(video)
        loss_b = bce_existence(out["hand_existence_logits"], exist_tgt)

        mask_l = exist_tgt[..., 0]
        mask_r = exist_tgt[..., 1]
        loss_m = mano_regression_loss(out["mano_left"], mano_l_tgt, mask_l) + mano_regression_loss(
            out["mano_right"], mano_r_tgt, mask_r
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

    mano_dims = {
        "trans": cfg.mano_trans_dim,
        "root": cfg.mano_root_orient_dim,
        "pose": cfg.mano_hand_pose_dim,
        "betas": cfg.mano_betas_dim,
    }

    ds = DummyVideoHandDataset(
        num_samples=256,
        seq_len=args.seq_len,
        image_size=cfg.image_size,
        mano_dims=mano_dims,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_hand_batch,
        drop_last=True,
    )

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for ep in range(1, args.epochs + 1):
        stats = train_one_epoch(model, loader, optim, device)
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
