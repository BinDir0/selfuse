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
    mano_vert: float,
    mano_bone: float,
    mano_bone_dir: float,
    mano_hand_pca: float,
    lp_l: float,
    lp_r: float,
    jl_w: float,
    jr_w: float,
    vl_w: float,
    vr_w: float,
    bl_w: float,
    br_w: float,
    bdl_w: float,
    bdr_w: float,
    hpc_l: float,
    hpc_r: float,
    optim_steps: int,
) -> dict[str, float]:
    return {
        "train/loss": loss,
        "train/bce": bce,
        "train/mano": mano,
        "train/mano_param_raw": mano_param_raw,
        "train/mano_param_weighted": mano_param_weighted,
        "train/mano_joint": mano_joint,
        "train/mano_vert": mano_vert,
        "train/mano_bone": mano_bone,
        "train/mano_bone_dir": mano_bone_dir,
        "train/mano_hand_pca": mano_hand_pca,
        "train/mano_param_left": lp_l,
        "train/mano_param_right": lp_r,
        "train/mano_joint_left": jl_w,
        "train/mano_joint_right": jr_w,
        "train/mano_vert_left": vl_w,
        "train/mano_vert_right": vr_w,
        "train/mano_bone_left": bl_w,
        "train/mano_bone_right": br_w,
        "train/mano_bone_dir_left": bdl_w,
        "train/mano_bone_dir_right": bdr_w,
        "train/mano_hand_pca_left": hpc_l,
        "train/mano_hand_pca_right": hpc_r,
        "train/mano_left": lp_l + jl_w + vl_w + bl_w + bdl_w + hpc_l,
        "train/mano_right": lp_r + jr_w + vr_w + br_w + bdr_w + hpc_r,
        "train/optim_step": float(optim_steps),
    }


def val_avg_to_tb_dict(avg: Mapping[str, float]) -> dict[str, float]:
    mp_l = float(avg["mano_param_left"])
    mp_r = float(avg["mano_param_right"])
    jl_w = float(avg["mano_joint_left"])
    jr_w = float(avg["mano_joint_right"])
    vl_w = float(avg.get("mano_vert_left", 0.0))
    vr_w = float(avg.get("mano_vert_right", 0.0))
    bl_l = float(avg.get("mano_bone_left", 0.0))
    bl_r = float(avg.get("mano_bone_right", 0.0))
    bdl_l = float(avg.get("mano_bone_dir_left", 0.0))
    bdl_r = float(avg.get("mano_bone_dir_right", 0.0))
    hpc_l = float(avg.get("mano_hand_pca_left", 0.0))
    hpc_r = float(avg.get("mano_hand_pca_right", 0.0))
    return {
        "val/loss": float(avg["loss"]),
        "val/bce": float(avg["bce"]),
        "val/mano": float(avg["mano"]),
        "val/mano_param_raw": float(avg.get("mano_param_raw", avg["mano_param"])),
        "val/mano_param_weighted": float(avg.get("mano_param_weighted", avg["mano_param"])),
        "val/mano_joint": float(avg["mano_joint"]),
        "val/mano_vert": float(avg.get("mano_vert", 0.0)),
        "val/mano_bone": float(avg.get("mano_bone", 0.0)),
        "val/mano_bone_dir": float(avg.get("mano_bone_dir", 0.0)),
        "val/mano_hand_pca": float(avg.get("mano_hand_pca", 0.0)),
        "val/mano_param_left": mp_l,
        "val/mano_param_right": mp_r,
        "val/mano_joint_left": jl_w,
        "val/mano_joint_right": jr_w,
        "val/mano_vert_left": vl_w,
        "val/mano_vert_right": vr_w,
        "val/mano_bone_left": bl_l,
        "val/mano_bone_right": bl_r,
        "val/mano_bone_dir_left": bdl_l,
        "val/mano_bone_dir_right": bdl_r,
        "val/mano_hand_pca_left": hpc_l,
        "val/mano_hand_pca_right": hpc_r,
        "val/mano_left": mp_l + jl_w + vl_w + bl_l + bdl_l + hpc_l,
        "val/mano_right": mp_r + jr_w + vr_w + bl_r + bdl_r + hpc_r,
    }
