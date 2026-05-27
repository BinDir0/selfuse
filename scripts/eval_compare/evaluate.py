"""Stage 3a: score one prediction folder against a sequence's ground truth.

Computes the four requested metrics (faithful to HaWoR paper definitions):
  ATE, ATE-S            camera trajectory (m)
  RPE_trans / RPE_rot   camera trajectory local consistency (m / deg)
  PA-MPJPE              per-frame Procrustes hand error (mm)
  WA-MPJPE              world-aligned hand error over 100-frame segments (mm)
  (W-MPJPE also reported for the PA <= WA <= W sanity ordering.)

Frame alignment: predictions and GT both index from 0 at the same fps, so we truncate
to the shorter length and intersect the per-hand validity masks.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from scripts.eval_compare import metrics as M
from scripts.eval_compare.gt_adapters.base import GTSequence
from scripts.eval_compare.load_pred import load_prediction


def _truncate(*arrs):
    n = min(a.shape[0] for a in arrs)
    return [a[:n] for a in arrs], n


def evaluate(gt_path: str, seq_folder: str, wa_segment: int = 100, rpe_delta: int = 1,
             use_cuda: bool = True) -> dict:
    gt = GTSequence.load_npz(gt_path)
    pred = load_prediction(seq_folder, use_cuda=use_cuda)

    out: dict = {"seq_id": gt.seq_id, "dataset": gt.dataset}

    # ----- camera trajectory ----- #
    (gt_pos, pred_pos), n = _truncate(gt.cam_pos_world, pred.cam_pos_world)
    cam_valid = np.isfinite(gt_pos).all(1) & np.isfinite(pred_pos).all(1)
    out["ATE"] = M.ate(pred_pos, gt_pos, valid=cam_valid, with_scale=True)
    out["ATE_S"] = M.ate_s(pred_pos, gt_pos, valid=cam_valid)
    gt_R, pred_R = gt.cam_R_c2w[:n], pred.cam_R_c2w[:n]
    rpe = M.rpe(pred_pos, gt_pos, pred_R, gt_R, delta=rpe_delta, valid=cam_valid)
    out["RPE_trans"] = rpe["rpe_trans"]
    out["RPE_rot_deg"] = rpe["rpe_rot_deg"]
    out["n_cam_frames"] = int(cam_valid.sum())

    # ----- hands (per hand, then mean over present hands) ----- #
    pa, wa, w = [], [], []
    for h in (0, 1):
        (gj, pj, gv, pv), nn = _truncate(
            gt.joints_world[h], pred.joints_world[h], gt.valid[h], pred.valid[h]
        )
        valid = gv.astype(bool) & pv.astype(bool)
        if valid.sum() < 3:
            out[f"PA_MPJPE_{'LR'[h]}"] = float("nan")
            out[f"WA_MPJPE_{'LR'[h]}"] = float("nan")
            continue
        pa_h = M.pa_mpjpe(pj, gj, valid=valid)
        wa_h = M.wa_mpjpe(pj, gj, valid=valid, seg=wa_segment)
        w_h = M.w_mpjpe(pj, gj, valid=valid, seg=wa_segment)
        out[f"PA_MPJPE_{'LR'[h]}"] = pa_h
        out[f"WA_MPJPE_{'LR'[h]}"] = wa_h
        out[f"n_hand_{'LR'[h]}"] = int(valid.sum())
        pa.append(pa_h); wa.append(wa_h); w.append(w_h)

    out["PA_MPJPE"] = float(np.nanmean(pa)) if pa else float("nan")
    out["WA_MPJPE"] = float(np.nanmean(wa)) if wa else float("nan")
    out["W_MPJPE"] = float(np.nanmean(w)) if w else float("nan")
    out["pred_scale"] = pred.scale
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="path to gt.npz")
    ap.add_argument("--pred", required=True, help="prediction seq_folder")
    ap.add_argument("--wa_segment", type=int, default=100)
    ap.add_argument("--rpe_delta", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    res = evaluate(args.gt, args.pred, args.wa_segment, args.rpe_delta, use_cuda=not args.cpu)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
