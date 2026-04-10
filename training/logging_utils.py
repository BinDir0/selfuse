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
    mano_param_raw: float,
    mano_param_weighted: float,
    mano_joint: float,
    lp_l: float,
    lp_r: float,
    jl_w: float,
    jr_w: float,
    optim_steps: int,
    mano_param_per_key: Mapping[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {
        "train/loss": loss,
        "train/bce": bce,
        "train/mano": mano,
        "train/mano_param_raw": mano_param_raw,
        "train/mano_param_weighted": mano_param_weighted,
        "train/mano_joint": mano_joint,
        "train/mano_param_left": lp_l,
        "train/mano_param_right": lp_r,
        "train/mano_joint_left": jl_w,
        "train/mano_joint_right": jr_w,
        "train/mano_left": lp_l + jl_w,
        "train/mano_right": lp_r + jr_w,
        "train/optim_step": float(optim_steps),
    }
    for k, v in mano_param_per_key.items():
        out[f"train/mano_param_key/{k}"] = v
    return out


def val_avg_to_tb_dict(avg: Mapping[str, float]) -> dict[str, float]:
    mp_l = float(avg["mano_param_left"])
    mp_r = float(avg["mano_param_right"])
    jl_w = float(avg["mano_joint_left"])
    jr_w = float(avg["mano_joint_right"])
    out: dict[str, float] = {
        "val/loss": float(avg["loss"]),
        "val/bce": float(avg["bce"]),
        "val/mano": float(avg["mano"]),
        "val/mano_param_raw": float(avg.get("mano_param_raw", avg["mano_param"])),
        "val/mano_param_weighted": float(avg.get("mano_param_weighted", avg["mano_param"])),
        "val/mano_joint": float(avg["mano_joint"]),
        "val/mano_param_left": mp_l,
        "val/mano_param_right": mp_r,
        "val/mano_joint_left": jl_w,
        "val/mano_joint_right": jr_w,
        "val/mano_left": mp_l + jl_w,
        "val/mano_right": mp_r + jr_w,
    }
    if "mano_persp" in avg:
        out["val/mano_persp"] = float(avg["mano_persp"])
    elif "mano_weakcam" in avg:
        out["val/mano_persp"] = float(avg["mano_weakcam"])
    if "mano_persp_reg" in avg:
        out["val/mano_persp_reg"] = float(avg["mano_persp_reg"])
    elif "mano_weakcam_reg" in avg:
        out["val/mano_persp_reg"] = float(avg["mano_weakcam_reg"])
    for key in ("trans", "root_orient", "hand_pose", "betas"):
        pk = f"mano_pk_{key}"
        if pk in avg:
            out[f"val/mano_param_key/{key}"] = float(avg[pk])
    return out
