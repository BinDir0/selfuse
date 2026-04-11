from __future__ import annotations

from collections import Counter
import os
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.batch import wds_batch_to_training_batch
from training.checkpoint import maybe_save_step_checkpoint
from training.logging_utils import (
    tb_add_scalars,
    train_step_tb_dict,
    val_avg_to_tb_dict,
    wandb_add_scalars,
)
from training.mano_train_render import maybe_save_train_mano_compare_png
from training.losses import (
    bce_existence,
    intrinsic_reg_loss,
    mano_masked_fullperspective_reproj_pixel_m2,
    mano_masked_joint_mse_m2,
    mano_regression_loss,
    mano_regression_per_key_raw,
)


def _batch_dataset_counter(batch: dict[str, Any]) -> Counter[str]:
    names: list[str] = []

    # Preferred: window-level dataset names (length B after default_collate).
    ds = batch.get("dataset_name", None)
    if isinstance(ds, (list, tuple)):
        names.extend(str(x) for x in ds)
    elif isinstance(ds, str):
        names.append(ds)

    # Fallback: frame-level meta shapes from different collate forms.
    if not names:
        meta = batch.get("meta", None)
        if isinstance(meta, dict):
            meta_ds = meta.get("dataset_name", None)
            if isinstance(meta_ds, (list, tuple)):
                # Could be (B,T) nested lists or flat list.
                for item in meta_ds:
                    if isinstance(item, (list, tuple)):
                        names.extend(str(x) for x in item)
                    else:
                        names.append(str(item))
        elif isinstance(meta, (list, tuple)):
            for seq in meta:
                if isinstance(seq, dict):
                    dn = seq.get("dataset_name", None)
                    if dn is not None:
                        names.append(str(dn))
                    continue
                if isinstance(seq, (list, tuple)):
                    for frame in seq:
                        if isinstance(frame, dict) and "dataset_name" in frame:
                            names.append(str(frame["dataset_name"]))

    return Counter(names)


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
    mano_joint_weight_21: torch.Tensor | None,
    mano_joint_smooth_max_weight: float,
    mano_joint_smooth_max_tau: float,
    mano_joint_chunk: int,
    mano_param_keys: frozenset[str],
    mano_pca_layers: tuple[nn.Module, nn.Module] | None,
    grad_clip: float,
    grad_accum_steps: int,
    base_lr: float,
    warmup_global_steps: int,
    tb_writer: Any | None,
    wandb_run: Any | None,
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
    mano_persp_loss_weight: float,
    mano_persp_reg_weight: float,
    camera_init_fx: float,
    camera_init_fy: float,
    camera_init_cx: float,
    camera_init_cy: float,
    augment_config: dict[str, Any] | None,
) -> tuple[dict[str, float], int, int, bool]:
    model.train()
    _ds = loader.dataset
    if hasattr(_ds, "set_epoch"):
        _ds.set_epoch(epoch)
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

    DEBUG_DATASET_DIST_EVERY = 100  # 每N步打印一次
    step_in_epoch = 0
    accum_steps = max(1, int(grad_accum_steps))
    warm_steps = max(0, int(warmup_global_steps))
    accum_counter = 0
    global_step = int(optim_steps)

    def _set_lr_for_step(step: int) -> float:
        if warm_steps <= 0:
            lr_now = float(base_lr)
        else:
            s = max(0, int(step))
            if s >= warm_steps:
                lr_now = float(base_lr)
            else:
                lr_now = float(base_lr) * (float(s) / float(warm_steps))
        for pg in optim.param_groups:
            pg["lr"] = lr_now
        return lr_now

    _set_lr_for_step(global_step)
    optim.zero_grad(set_to_none=True)
    for batch in pbar:
        # Debug: 打印当前batch的dataset_name分布
        if step_in_epoch % DEBUG_DATASET_DIST_EVERY == 0 and is_rank0:
            dist = _batch_dataset_counter(batch)
            print(f"[DEBUG] Step {step_in_epoch}: batch dataset_name distribution: {dict(dist)}")

        video, exist_tgt, mano_l_tgt, mano_r_tgt, intr_bt = wds_batch_to_training_batch(
            batch,
            device=device,
            image_size=image_size,
            apply_left_root_fix=apply_left_root_fix,
            mano_pca_layers=mano_pca_layers,
            augment_config=augment_config,
        )
        step_in_epoch += 1

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
        ppl, ppr = zj, zj
        preg_l, preg_r = zj, zj
        if mano_pca_layers is not None:
            ml, mr = mano_pca_layers
            if mano_joint_loss_weight > 0.0:
                lj_l = mano_masked_joint_mse_m2(
                    out["mano_left"],
                    mano_l_tgt,
                    mask_l,
                    ml,
                    chunk=mano_joint_chunk,
                    joint_weight_21=mano_joint_weight_21,
                    smooth_max_weight=mano_joint_smooth_max_weight,
                    smooth_max_tau=mano_joint_smooth_max_tau,
                )
                lj_r = mano_masked_joint_mse_m2(
                    out["mano_right"],
                    mano_r_tgt,
                    mask_r,
                    mr,
                    chunk=mano_joint_chunk,
                    joint_weight_21=mano_joint_weight_21,
                    smooth_max_weight=mano_joint_smooth_max_weight,
                    smooth_max_tau=mano_joint_smooth_max_tau,
                )
            if mano_persp_loss_weight > 0.0:
                ppl = mano_masked_fullperspective_reproj_pixel_m2(
                    out["mano_left"],
                    mano_l_tgt,
                    out["pred_cam"][:, :, 0, :],
                    intr_bt,
                    mask_l,
                    ml,
                    chunk=mano_joint_chunk,
                    joint_weight_21=mano_joint_weight_21,
                )
                ppr = mano_masked_fullperspective_reproj_pixel_m2(
                    out["mano_right"],
                    mano_r_tgt,
                    out["pred_cam"][:, :, 1, :],
                    intr_bt,
                    mask_r,
                    mr,
                    chunk=mano_joint_chunk,
                    joint_weight_21=mano_joint_weight_21,
                )
            if mano_persp_reg_weight > 0.0:
                preg_l = intrinsic_reg_loss(
                    out["pred_cam"][:, :, 0, :],
                    mask_l,
                    ref_fx=camera_init_fx,
                    ref_fy=camera_init_fy,
                    ref_cx=camera_init_cx,
                    ref_cy=camera_init_cy,
                )
                preg_r = intrinsic_reg_loss(
                    out["pred_cam"][:, :, 1, :],
                    mask_r,
                    ref_fx=camera_init_fx,
                    ref_fy=camera_init_fy,
                    ref_cx=camera_init_cx,
                    ref_cy=camera_init_cy,
                )

        in_warmup = global_step < warm_steps
        persp_weight_eff = 0.0 if in_warmup else float(mano_persp_loss_weight)

        loss_joint = mano_joint_loss_weight * (lj_l + lj_r)
        jl_w = mano_joint_loss_weight * lj_l
        jr_w = mano_joint_loss_weight * lj_r
        loss_persp = persp_weight_eff * (ppl + ppr)
        loss_persp_reg = mano_persp_reg_weight * (preg_l + preg_r)

        loss_m = loss_param_w + loss_joint + loss_persp + loss_persp_reg

        loss = loss_b + loss_m
        (loss / float(accum_steps)).backward()
        accum_counter += 1

        did_step = accum_counter >= accum_steps
        if did_step:
            _set_lr_for_step(global_step)
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)
            accum_counter = 0
            optim_steps += 1
            global_step += 1

        tot["loss"] += float(loss.detach())
        tot["bce"] += float(loss_b.detach())
        tot["mano"] += float(loss_m.detach())
        n += 1
        pbar.set_postfix(
            step=str(optim_steps),
            loss=f"{float(loss.detach()):.4f}",
            bce=f"{float(loss_b.detach()):.4f}",
            mano=f"{float(loss_m.detach()):.4f}",
            param=f"{float(loss_param_w.detach()):.4f}",
            joint=f"{float(loss_joint.detach()):.4f}",
            persp=f"{float(loss_persp.detach()):.4f}",
            preg=f"{float(loss_persp_reg.detach()):.4f}",
            persp_raw=f"{float((ppl + ppr).detach()):.4f}",
            preg_raw=f"{float((preg_l + preg_r).detach()):.4f}",
            warmup=int(in_warmup),
            lr=f"{float(optim.param_groups[0]['lr']):.2e}",
        )
        train_scalars: dict[str, float] | None = None
        cam_scalars: dict[str, float] | None = None
        if tb_writer is not None or wandb_run is not None:
            pk_l = mano_regression_per_key_raw(
                out["mano_left"],
                mano_l_tgt,
                mask_l,
                param_keys=mano_param_keys,
                param_loss=mano_param_loss,
                huber_delta=mano_huber_delta,
            )
            pk_r = mano_regression_per_key_raw(
                out["mano_right"],
                mano_r_tgt,
                mask_r,
                param_keys=mano_param_keys,
                param_loss=mano_param_loss,
                huber_delta=mano_huber_delta,
            )
            param_per_key = {
                k: 0.5 * (float(pk_l[k].detach()) + float(pk_r[k].detach()))
                for k in pk_l.keys()
            }
            train_scalars = train_step_tb_dict(
                loss=float(loss.detach()),
                bce=float(loss_b.detach()),
                mano=float(loss_m.detach()),
                mano_param_raw=float(loss_param.detach()),
                mano_param_weighted=float(loss_param_w.detach()),
                mano_joint=float(loss_joint.detach()),
                lp_l=float(lp_l.detach()),
                lp_r=float(lp_r.detach()),
                jl_w=float(jl_w.detach()),
                jr_w=float(jr_w.detach()),
                optim_steps=optim_steps,
                mano_param_per_key=param_per_key,
            )
            cam_scalars = {
                "train/mano_persp_weighted": float(loss_persp.detach()),
                "train/mano_persp_reg_weighted": float(loss_persp_reg.detach()),
                "train/pred_cam_left_fx": float(out["pred_cam"][:, :, 0, 0].detach().mean()),
                "train/pred_cam_left_fy": float(out["pred_cam"][:, :, 0, 1].detach().mean()),
                "train/pred_cam_left_cx": float(out["pred_cam"][:, :, 0, 2].detach().mean()),
                "train/pred_cam_left_cy": float(out["pred_cam"][:, :, 0, 3].detach().mean()),
                "train/pred_cam_right_fx": float(out["pred_cam"][:, :, 1, 0].detach().mean()),
                "train/pred_cam_right_fy": float(out["pred_cam"][:, :, 1, 1].detach().mean()),
                "train/pred_cam_right_cx": float(out["pred_cam"][:, :, 1, 2].detach().mean()),
                "train/pred_cam_right_cy": float(out["pred_cam"][:, :, 1, 3].detach().mean()),
            }
        if train_scalars is not None:
            tb_add_scalars(tb_writer, global_step, train_scalars)
        if cam_scalars is not None:
            tb_add_scalars(tb_writer, global_step, cam_scalars)
        if did_step and train_scalars is not None:
            wandb_add_scalars(wandb_run, global_step, train_scalars)
        if did_step and cam_scalars is not None:
            wandb_add_scalars(wandb_run, global_step, cam_scalars)
        if did_step and save_every_steps > 0 and optim_steps % save_every_steps == 0:
            maybe_save_step_checkpoint(ckpt_dir, optim_steps, model, is_rank0=is_rank0)

        if (
            did_step
            and render_mano_every > 0
            and render_mano_dir
            and mano_pca_layers is not None
            and is_rank0
            and optim_steps % render_mano_every == 0
        ):
            _outp = os.path.join(render_mano_dir, f"e{epoch:04d}_step{optim_steps:08d}.png")
            maybe_save_train_mano_compare_png(
                out_path=_outp,
                video_btchw=video,
                intr_bt=intr_bt,
                mano_l_gt=mano_l_tgt,
                mano_r_gt=mano_r_tgt,
                mano_l_pr=out["mano_left"],
                mano_r_pr=out["mano_right"],
                pred_intr_bt=out["pred_cam"],
                exist_bt2=exist_tgt,
                mano_pca_layers=mano_pca_layers,
                joint_chunk=mano_joint_chunk,
                device=device,
            )

        if did_step and max_steps > 0 and optim_steps >= max_steps:
            hit_max = True
            break

    if not hit_max and accum_counter > 0:
        _set_lr_for_step(global_step)
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)
        optim_steps += 1
        global_step += 1
        if save_every_steps > 0 and optim_steps % save_every_steps == 0:
            maybe_save_step_checkpoint(ckpt_dir, optim_steps, model, is_rank0=is_rank0)
        if max_steps > 0 and optim_steps >= max_steps:
            hit_max = True

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
    wandb_add_scalars(
        wandb_run,
        global_step,
        {
            "epoch/loss": avg["loss"],
            "epoch/bce": avg["bce"],
            "epoch/mano": avg["mano"],
            "epoch/index": float(epoch),
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
    mano_joint_weight_21: torch.Tensor | None,
    mano_joint_smooth_max_weight: float,
    mano_joint_smooth_max_tau: float,
    mano_joint_chunk: int,
    mano_param_keys: frozenset[str],
    mano_pca_layers: tuple[nn.Module, nn.Module] | None,
    tb_writer: Any | None,
    wandb_run: Any | None,
    epoch: int,
    log_step: int,
    show_progress: bool,
    is_rank0: bool,
    mano_persp_loss_weight: float,
    mano_persp_reg_weight: float,
    camera_init_fx: float,
    camera_init_fy: float,
    camera_init_cx: float,
    camera_init_cy: float,
    augment_config: dict[str, Any] | None,
) -> dict[str, float]:
    model.eval()
    _ds = loader.dataset
    if hasattr(_ds, "set_epoch"):
        _ds.set_epoch(epoch)
    tot: dict[str, float] = {
        "loss": 0.0,
        "bce": 0.0,
        "mano": 0.0,
        "mano_param": 0.0,
        "mano_param_raw": 0.0,
        "mano_param_weighted": 0.0,
        "mano_joint": 0.0,
        "mano_param_left": 0.0,
        "mano_param_right": 0.0,
        "mano_joint_left": 0.0,
        "mano_joint_right": 0.0,
        "mano_persp": 0.0,
        "mano_persp_reg": 0.0,
    }
    for k in sorted(mano_param_keys & frozenset({"trans", "root_orient", "hand_pose", "betas"})):
        tot[f"mano_pk_{k}"] = 0.0
    n = 0
    pbar = tqdm(
        loader,
        desc=f"val ep{epoch}",
        leave=True,
        dynamic_ncols=True,
        disable=not (show_progress and is_rank0),
    )
    for batch in pbar:
        video, exist_tgt, mano_l_tgt, mano_r_tgt, intr_bt = wds_batch_to_training_batch(
            batch,
            device=device,
            image_size=image_size,
            apply_left_root_fix=apply_left_root_fix,
            mano_pca_layers=mano_pca_layers,
            augment_config=augment_config,
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
        ppl, ppr = zj, zj
        preg_l, preg_r = zj, zj
        if mano_pca_layers is not None:
            ml, mr = mano_pca_layers
            if mano_joint_loss_weight > 0.0:
                lj_l = mano_masked_joint_mse_m2(
                    out["mano_left"],
                    mano_l_tgt,
                    mask_l,
                    ml,
                    chunk=mano_joint_chunk,
                    joint_weight_21=mano_joint_weight_21,
                    smooth_max_weight=mano_joint_smooth_max_weight,
                    smooth_max_tau=mano_joint_smooth_max_tau,
                )
                lj_r = mano_masked_joint_mse_m2(
                    out["mano_right"],
                    mano_r_tgt,
                    mask_r,
                    mr,
                    chunk=mano_joint_chunk,
                    joint_weight_21=mano_joint_weight_21,
                    smooth_max_weight=mano_joint_smooth_max_weight,
                    smooth_max_tau=mano_joint_smooth_max_tau,
                )
            if mano_persp_loss_weight > 0.0:
                ppl = mano_masked_fullperspective_reproj_pixel_m2(
                    out["mano_left"],
                    mano_l_tgt,
                    out["pred_cam"][:, :, 0, :],
                    intr_bt,
                    mask_l,
                    ml,
                    chunk=mano_joint_chunk,
                    joint_weight_21=mano_joint_weight_21,
                )
                ppr = mano_masked_fullperspective_reproj_pixel_m2(
                    out["mano_right"],
                    mano_r_tgt,
                    out["pred_cam"][:, :, 1, :],
                    intr_bt,
                    mask_r,
                    mr,
                    chunk=mano_joint_chunk,
                    joint_weight_21=mano_joint_weight_21,
                )
            if mano_persp_reg_weight > 0.0:
                preg_l = intrinsic_reg_loss(
                    out["pred_cam"][:, :, 0, :],
                    mask_l,
                    ref_fx=camera_init_fx,
                    ref_fy=camera_init_fy,
                    ref_cx=camera_init_cx,
                    ref_cy=camera_init_cy,
                )
                preg_r = intrinsic_reg_loss(
                    out["pred_cam"][:, :, 1, :],
                    mask_r,
                    ref_fx=camera_init_fx,
                    ref_fy=camera_init_fy,
                    ref_cx=camera_init_cx,
                    ref_cy=camera_init_cy,
                )

        loss_joint = mano_joint_loss_weight * (lj_l + lj_r)
        jl_w = mano_joint_loss_weight * lj_l
        jr_w = mano_joint_loss_weight * lj_r
        loss_persp = mano_persp_loss_weight * (ppl + ppr)
        loss_persp_reg = mano_persp_reg_weight * (preg_l + preg_r)

        loss_m = loss_param_w + loss_joint + loss_persp + loss_persp_reg
        loss = loss_b + loss_m

        pk_l = mano_regression_per_key_raw(
            out["mano_left"],
            mano_l_tgt,
            mask_l,
            param_keys=mano_param_keys,
            param_loss=mano_param_loss,
            huber_delta=mano_huber_delta,
        )
        pk_r = mano_regression_per_key_raw(
            out["mano_right"],
            mano_r_tgt,
            mask_r,
            param_keys=mano_param_keys,
            param_loss=mano_param_loss,
            huber_delta=mano_huber_delta,
        )
        for k in pk_l.keys():
            tot[f"mano_pk_{k}"] += 0.5 * (float(pk_l[k]) + float(pk_r[k]))

        tot["loss"] += float(loss)
        tot["bce"] += float(loss_b)
        tot["mano"] += float(loss_m)
        tot["mano_param"] += float(loss_param)
        tot["mano_param_raw"] += float(loss_param)
        tot["mano_param_weighted"] += float(loss_param_w)
        tot["mano_joint"] += float(loss_joint)
        tot["mano_param_left"] += float(lp_l)
        tot["mano_param_right"] += float(lp_r)
        tot["mano_joint_left"] += float(jl_w)
        tot["mano_joint_right"] += float(jr_w)
        tot["mano_persp"] += float(loss_persp)
        tot["mano_persp_reg"] += float(loss_persp_reg)
        n += 1
        pbar.set_postfix(
            loss=f"{float(loss):.4f}",
            bce=f"{float(loss_b):.4f}",
            mano=f"{float(loss_m):.4f}",
            param=f"{float(loss_param_w):.4f}",
            joint=f"{float(loss_joint):.4f}",
            persp=f"{float(loss_persp):.4f}",
            preg=f"{float(loss_persp_reg):.4f}",
            persp_raw=f"{float(ppl + ppr):.4f}",
            preg_raw=f"{float(preg_l + preg_r):.4f}",
        )
    nn_ = max(n, 1)
    avg = {k: tot[k] / nn_ for k in tot}
    val_scalars = val_avg_to_tb_dict(avg)
    if tb_writer is not None:
        tb_add_scalars(tb_writer, epoch, val_scalars)
    wandb_add_scalars(
        wandb_run,
        log_step,
        {
            **val_scalars,
            "val/epoch": float(epoch),
        },
    )
    return avg
