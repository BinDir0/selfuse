from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def log_line(msg: str, log_file: Path | None) -> None:
    print(msg, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(f"{ts}\t{msg}\n")


def tb_add_scalars(writer: Any | None, step: int, scalars: Mapping[str, float]) -> None:
    if writer is None:
        return
    for tag, val in scalars.items():
        writer.add_scalar(tag, val, step)


def train_step_tb_dict(
    *,
    loss: float,
    bce: float,
    mano: float,
    mano_param: float,
    mano_joint: float,
    lp_l: float,
    lp_r: float,
    jl_w: float,
    jr_w: float,
    optim_steps: int,
) -> dict[str, float]:
    return {
        "train/loss": loss,
        "train/bce": bce,
        "train/mano": mano,
        "train/mano_param": mano_param,
        "train/mano_joint": mano_joint,
        "train/mano_param_left": lp_l,
        "train/mano_param_right": lp_r,
        "train/mano_joint_left": jl_w,
        "train/mano_joint_right": jr_w,
        "train/mano_left": lp_l + jl_w,
        "train/mano_right": lp_r + jr_w,
        "train/optim_step": float(optim_steps),
    }


def val_avg_to_tb_dict(avg: Mapping[str, float]) -> dict[str, float]:
    return {
        "val/loss": float(avg["loss"]),
        "val/bce": float(avg["bce"]),
        "val/mano": float(avg["mano"]),
        "val/mano_param": float(avg["mano_param"]),
        "val/mano_joint": float(avg["mano_joint"]),
        "val/mano_param_left": float(avg["mano_param_left"]),
        "val/mano_param_right": float(avg["mano_param_right"]),
        "val/mano_joint_left": float(avg["mano_joint_left"]),
        "val/mano_joint_right": float(avg["mano_joint_right"]),
        "val/mano_left": float(avg["mano_param_left"] + avg["mano_joint_left"]),
        "val/mano_right": float(avg["mano_param_right"] + avg["mano_joint_right"]),
    }
