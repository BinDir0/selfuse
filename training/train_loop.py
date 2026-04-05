from __future__ import annotations

import os
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.batch import wds_batch_to_training_batch
from training.checkpoint import maybe_save_step_checkpoint
from training.logging_utils import tb_add_scalars, train_step_tb_dict, val_avg_to_tb_dict
from training.mano_train_render import maybe_save_train_mano_compare_png
from training.losses import (
    bce_existence,
    mano_hand_pose_pca_mse,
    mano_masked_bone_direction_mse,
    mano_masked_bone_length_mse_m2,
    mano_masked_joint_mse_m2,
    mano_masked_vert_mse_m2,
    mano_regression_loss,
)


def train_one_epoch(
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    model: nn.Module,
    *,
    image_size: int,
    apply_left_root_fix: bool,
    mano_pose_weight: float,
    mano_root_orient_weight: float,
    mano_param_loss: Literal["l1", "l2", "huber"],
    mano_huber_delta: float,
    mano_param_loss_weight: float,
    mano_joint_loss_weight: float,
    mano_vert_loss_weight: float,
    mano_bone_loss_weight: float,
    mano_bone_dir_loss_weight: float,
    mano_joint_chunk: int,
    mano_param_keys: frozenset[str],
    mano_hand_pose_pca_loss_weight: float,
    mano_pca_layers: tuple[nn.Module, nn.Module] | None,
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
    render_mano_every: int,
    render_mano_dir: str | None,
) -> tuple[dict[str, float], int, int, bool]:
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
        lp_l = mano_regression_loss(
            out["mano_left"],
            mano_l_tgt,
            mask_l,
            param_keys=mano_param_keys,
            hand_pose_weight=mano_pose_weight,
            root_orient_weight=mano_root_orient_weight,
            param_loss=mano_param_loss,
            huber_delta=mano_huber_delta,
        )
        lp_r = mano_regression_loss(
            out["mano_right"],
            mano_r_tgt,
            mask_r,
            param_keys=mano_param_keys,
            hand_pose_weight=mano_pose_weight,
            root_orient_weight=mano_root_orient_weight,
            param_loss=mano_param_loss,
            huber_delta=mano_huber_delta,
        )
        loss_param = lp_l + lp_r
        loss_param_w = mano_param_loss_weight * loss_param

        zj = out["mano_left"]["trans"].new_tensor(0.0)
        lj_l, lj_r = zj, zj
        vv_l, vv_r = zj, zj
        bl_l, bl_r = zj, zj
        bd_l, bd_r = zj, zj
        hpc_l, hpc_r = zj, zj
        if mano_pca_layers is not None:
            ml, mr = mano_pca_layers
            if mano_joint_loss_weight > 0.0:
                lj_l = mano_masked_joint_mse_m2(
                    out["mano_left"], mano_l_tgt, mask_l, ml, chunk=mano_joint_chunk
                )
                lj_r = mano_masked_joint_mse_m2(
                    out["mano_right"], mano_r_tgt, mask_r, mr, chunk=mano_joint_chunk
                )
            if mano_vert_loss_weight > 0.0:
                vv_l = mano_masked_vert_mse_m2(
                    out["mano_left"], mano_l_tgt, mask_l, ml, chunk=mano_joint_chunk
                )
                vv_r = mano_masked_vert_mse_m2(
                    out["mano_right"], mano_r_tgt, mask_r, mr, chunk=mano_joint_chunk
                )
            if mano_bone_loss_weight > 0.0:
                bl_l = mano_masked_bone_length_mse_m2(
                    out["mano_left"], mano_l_tgt, mask_l, ml, chunk=mano_joint_chunk
                )
                bl_r = mano_masked_bone_length_mse_m2(
                    out["mano_right"], mano_r_tgt, mask_r, mr, chunk=mano_joint_chunk
                )
            if mano_bone_dir_loss_weight > 0.0:
                bd_l = mano_masked_bone_direction_mse(
                    out["mano_left"], mano_l_tgt, mask_l, ml, chunk=mano_joint_chunk
                )
                bd_r = mano_masked_bone_direction_mse(
                    out["mano_right"], mano_r_tgt, mask_r, mr, chunk=mano_joint_chunk
                )
            if mano_hand_pose_pca_loss_weight > 0.0:
                hpc_l = mano_hand_pose_pca_mse(out["mano_left"], mano_l_tgt, mask_l, ml)
                hpc_r = mano_hand_pose_pca_mse(out["mano_right"], mano_r_tgt, mask_r, mr)
        loss_joint = mano_joint_loss_weight * (lj_l + lj_r)
        jl_w = mano_joint_loss_weight * lj_l
        jr_w = mano_joint_loss_weight * lj_r
        loss_vert = mano_vert_loss_weight * (vv_l + vv_r)
        vl_w = mano_vert_loss_weight * vv_l
        vr_w = mano_vert_loss_weight * vv_r
        loss_bone = mano_bone_loss_weight * (bl_l + bl_r)
        blw_l = mano_bone_loss_weight * bl_l
        blw_r = mano_bone_loss_weight * bl_r
        loss_bone_dir = mano_bone_dir_loss_weight * (bd_l + bd_r)
        bddl_w = mano_bone_dir_loss_weight * bd_l
        bddr_w = mano_bone_dir_loss_weight * bd_r
        loss_hand_pca = mano_hand_pose_pca_loss_weight * (hpc_l + hpc_r)
        hpcl_w = mano_hand_pose_pca_loss_weight * hpc_l
        hpcr_w = mano_hand_pose_pca_loss_weight * hpc_r
        loss_m = loss_param_w + loss_joint + loss_vert + loss_bone + loss_bone_dir + loss_hand_pca

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
            tb_add_scalars(
                tb_writer,
                global_step,
                train_step_tb_dict(
                    loss=float(loss.detach()),
                    bce=float(loss_b.detach()),
                    mano=float(loss_m.detach()),
                    mano_param_raw=float(loss_param.detach()),
                    mano_param_weighted=float(loss_param_w.detach()),
                    mano_joint=float(loss_joint.detach()),
                    mano_vert=float(loss_vert.detach()),
                    mano_bone=float(loss_bone.detach()),
                    mano_bone_dir=float(loss_bone_dir.detach()),
                    mano_hand_pca=float(loss_hand_pca.detach()),
                    lp_l=float(lp_l.detach()),
                    lp_r=float(lp_r.detach()),
                    jl_w=float(jl_w.detach()),
                    jr_w=float(jr_w.detach()),
                    vl_w=float(vl_w.detach()),
                    vr_w=float(vr_w.detach()),
                    bl_w=float(blw_l.detach()),
                    br_w=float(blw_r.detach()),
                    bdl_w=float(bddl_w.detach()),
                    bdr_w=float(bddr_w.detach()),
                    hpc_l=float(hpcl_w.detach()),
                    hpc_r=float(hpcr_w.detach()),
                    optim_steps=optim_steps,
                ),
            )
            global_step += 1

        if save_every_steps > 0 and optim_steps % save_every_steps == 0:
            maybe_save_step_checkpoint(ckpt_dir, optim_steps, model, is_rank0=is_rank0)

        if (
            render_mano_every > 0
            and render_mano_dir
            and mano_pca_layers is not None
            and is_rank0
            and optim_steps % render_mano_every == 0
        ):
            _outp = os.path.join(render_mano_dir, f"e{epoch:04d}_step{optim_steps:08d}.png")
            maybe_save_train_mano_compare_png(
                out_path=_outp,
                video_btchw=video,
                batch=batch,
                mano_l_gt=mano_l_tgt,
                mano_r_gt=mano_r_tgt,
                mano_l_pr=out["mano_left"],
                mano_r_pr=out["mano_right"],
                exist_bt2=exist_tgt,
                mano_pca_layers=mano_pca_layers,
                joint_chunk=mano_joint_chunk,
                device=device,
            )

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
        tb_add_scalars(
            tb_writer,
            epoch,
            {
                "epoch/loss": avg["loss"],
                "epoch/bce": avg["bce"],
                "epoch/mano": avg["mano"],
            },
        )
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
    mano_root_orient_weight: float,
    mano_param_loss: Literal["l1", "l2", "huber"],
    mano_huber_delta: float,
    mano_param_loss_weight: float,
    mano_joint_loss_weight: float,
    mano_vert_loss_weight: float,
    mano_bone_loss_weight: float,
    mano_bone_dir_loss_weight: float,
    mano_joint_chunk: int,
    mano_param_keys: frozenset[str],
    mano_hand_pose_pca_loss_weight: float,
    mano_pca_layers: tuple[nn.Module, nn.Module] | None,
    tb_writer: Any | None,
    epoch: int,
    show_progress: bool,
    is_rank0: bool,
) -> dict[str, float]:
    model.eval()
    tot = {
        "loss": 0.0,
        "bce": 0.0,
        "mano": 0.0,
        "mano_param": 0.0,
        "mano_param_raw": 0.0,
        "mano_param_weighted": 0.0,
        "mano_joint": 0.0,
        "mano_vert": 0.0,
        "mano_bone": 0.0,
        "mano_bone_dir": 0.0,
        "mano_hand_pca": 0.0,
        "mano_param_left": 0.0,
        "mano_param_right": 0.0,
        "mano_joint_left": 0.0,
        "mano_joint_right": 0.0,
        "mano_vert_left": 0.0,
        "mano_vert_right": 0.0,
        "mano_bone_left": 0.0,
        "mano_bone_right": 0.0,
        "mano_bone_dir_left": 0.0,
        "mano_bone_dir_right": 0.0,
        "mano_hand_pca_left": 0.0,
        "mano_hand_pca_right": 0.0,
    }
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
        lp_l = mano_regression_loss(
            out["mano_left"],
            mano_l_tgt,
            mask_l,
            param_keys=mano_param_keys,
            hand_pose_weight=mano_pose_weight,
            root_orient_weight=mano_root_orient_weight,
            param_loss=mano_param_loss,
            huber_delta=mano_huber_delta,
        )
        lp_r = mano_regression_loss(
            out["mano_right"],
            mano_r_tgt,
            mask_r,
            param_keys=mano_param_keys,
            hand_pose_weight=mano_pose_weight,
            root_orient_weight=mano_root_orient_weight,
            param_loss=mano_param_loss,
            huber_delta=mano_huber_delta,
        )
        loss_param = lp_l + lp_r
        loss_param_w = mano_param_loss_weight * loss_param

        zj = out["mano_left"]["trans"].new_tensor(0.0)
        lj_l, lj_r = zj, zj
        vv_l, vv_r = zj, zj
        bl_l, bl_r = zj, zj
        bd_l, bd_r = zj, zj
        hpc_l, hpc_r = zj, zj
        if mano_pca_layers is not None:
            ml, mr = mano_pca_layers
            if mano_joint_loss_weight > 0.0:
                lj_l = mano_masked_joint_mse_m2(
                    out["mano_left"], mano_l_tgt, mask_l, ml, chunk=mano_joint_chunk
                )
                lj_r = mano_masked_joint_mse_m2(
                    out["mano_right"], mano_r_tgt, mask_r, mr, chunk=mano_joint_chunk
                )
            if mano_vert_loss_weight > 0.0:
                vv_l = mano_masked_vert_mse_m2(
                    out["mano_left"], mano_l_tgt, mask_l, ml, chunk=mano_joint_chunk
                )
                vv_r = mano_masked_vert_mse_m2(
                    out["mano_right"], mano_r_tgt, mask_r, mr, chunk=mano_joint_chunk
                )
            if mano_bone_loss_weight > 0.0:
                bl_l = mano_masked_bone_length_mse_m2(
                    out["mano_left"], mano_l_tgt, mask_l, ml, chunk=mano_joint_chunk
                )
                bl_r = mano_masked_bone_length_mse_m2(
                    out["mano_right"], mano_r_tgt, mask_r, mr, chunk=mano_joint_chunk
                )
            if mano_bone_dir_loss_weight > 0.0:
                bd_l = mano_masked_bone_direction_mse(
                    out["mano_left"], mano_l_tgt, mask_l, ml, chunk=mano_joint_chunk
                )
                bd_r = mano_masked_bone_direction_mse(
                    out["mano_right"], mano_r_tgt, mask_r, mr, chunk=mano_joint_chunk
                )
            if mano_hand_pose_pca_loss_weight > 0.0:
                hpc_l = mano_hand_pose_pca_mse(out["mano_left"], mano_l_tgt, mask_l, ml)
                hpc_r = mano_hand_pose_pca_mse(out["mano_right"], mano_r_tgt, mask_r, mr)
        loss_joint = mano_joint_loss_weight * (lj_l + lj_r)
        jl_w = mano_joint_loss_weight * lj_l
        jr_w = mano_joint_loss_weight * lj_r
        loss_vert = mano_vert_loss_weight * (vv_l + vv_r)
        vl_w = mano_vert_loss_weight * vv_l
        vr_w = mano_vert_loss_weight * vv_r
        loss_bone = mano_bone_loss_weight * (bl_l + bl_r)
        blw_l = mano_bone_loss_weight * bl_l
        blw_r = mano_bone_loss_weight * bl_r
        loss_bone_dir = mano_bone_dir_loss_weight * (bd_l + bd_r)
        bddl_w = mano_bone_dir_loss_weight * bd_l
        bddr_w = mano_bone_dir_loss_weight * bd_r
        loss_hand_pca = mano_hand_pose_pca_loss_weight * (hpc_l + hpc_r)
        hpcl_w = mano_hand_pose_pca_loss_weight * hpc_l
        hpcr_w = mano_hand_pose_pca_loss_weight * hpc_r
        loss_m = loss_param_w + loss_joint + loss_vert + loss_bone + loss_bone_dir + loss_hand_pca
        loss = loss_b + loss_m
        tot["loss"] += float(loss)
        tot["bce"] += float(loss_b)
        tot["mano"] += float(loss_m)
        tot["mano_param"] += float(loss_param)
        tot["mano_param_raw"] += float(loss_param)
        tot["mano_param_weighted"] += float(loss_param_w)
        tot["mano_joint"] += float(loss_joint)
        tot["mano_vert"] += float(loss_vert)
        tot["mano_bone"] += float(loss_bone)
        tot["mano_bone_dir"] += float(loss_bone_dir)
        tot["mano_hand_pca"] += float(loss_hand_pca)
        tot["mano_param_left"] += float(lp_l)
        tot["mano_param_right"] += float(lp_r)
        tot["mano_joint_left"] += float(jl_w)
        tot["mano_joint_right"] += float(jr_w)
        tot["mano_vert_left"] += float(vl_w)
        tot["mano_vert_right"] += float(vr_w)
        tot["mano_bone_left"] += float(blw_l)
        tot["mano_bone_right"] += float(blw_r)
        tot["mano_bone_dir_left"] += float(bddl_w)
        tot["mano_bone_dir_right"] += float(bddr_w)
        tot["mano_hand_pca_left"] += float(hpcl_w)
        tot["mano_hand_pca_right"] += float(hpcr_w)
        n += 1
        pbar.set_postfix(
            loss=f"{float(loss):.4f}",
            bce=f"{float(loss_b):.4f}",
            mano=f"{float(loss_m):.4f}",
        )
    nn_ = max(n, 1)
    avg = {k: tot[k] / nn_ for k in tot}
    if tb_writer is not None:
        tb_add_scalars(tb_writer, epoch, val_avg_to_tb_dict(avg))
    return avg
