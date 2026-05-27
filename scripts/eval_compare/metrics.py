"""Metric math for the fork-vs-upstream HaWoR comparison.

All functions are pure NumPy (no torch / repo deps) so they can be unit-tested in
isolation. Definitions follow the HaWoR paper (arXiv:2501.02973):

  * PA-MPJPE  : per-frame similarity (scale+R+t) Procrustes alignment, then MPJPE.
  * W-MPJPE   : per 100-frame segment, align the FIRST frame only (rigid), then MPJPE
                (kept for completeness / sanity ordering; the paper reports it too).
  * WA-MPJPE  : per 100-frame segment, single similarity transform over the WHOLE
                segment, then MPJPE.
  * ATE       : camera-trajectory RMSE after a single Sim(3) (scale+R+t) alignment.
  * ATE-S     : camera-trajectory RMSE using HaWoR's OWN metric scale (rigid SE(3)
                alignment only, no GT-scale fitting) -> sensitive to absolute scale.
  * RPE       : relative pose error over a fixed frame delta (translation + rotation).

Joint arrays are MPJPE inputs in metres; MPJPE results are returned in millimetres.
Camera trajectories are (T,3) positions (+ optional (T,3,3) rotations for RPE).

Cross-check reference (run on production on one sequence):
  thirdparty/DROID-SLAM/thirdparty/tartanair_tools/evaluation/evaluate_ate_scale.py
  thirdparty/DROID-SLAM/thirdparty/tartanair_tools/evaluation/evaluate_rpe.py
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Alignment primitives
# --------------------------------------------------------------------------- #


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Solves for (s, R, t) minimising ``|| dst - (s R src + t) ||`` (Umeyama 1991).

    Args:
        src: (N, D) source points.
        dst: (N, D) target points.
        with_scale: if False, fix s = 1 (rigid SE(D)).

    Returns:
        (s, R, t): float scale, (D, D) rotation, (D,) translation.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.ndim == 2, (src.shape, dst.shape)
    n, dim = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s_correct = np.eye(dim)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_correct[-1, -1] = -1.0
    rot = u @ s_correct @ vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        scale = float(np.trace(np.diag(d) @ s_correct) / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0

    trans = mu_dst - scale * rot @ mu_src
    return scale, rot, trans


def apply_sim3(points: np.ndarray, s: float, rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply ``s R p + t`` to (..., D) points."""
    return s * (points @ rot.T) + t


# --------------------------------------------------------------------------- #
# Camera-trajectory metrics
# --------------------------------------------------------------------------- #


def _valid_pair(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    if valid is None:
        valid = np.ones(pred.shape[0], dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    finite = np.isfinite(pred).all(axis=tuple(range(1, pred.ndim))) & np.isfinite(gt).all(
        axis=tuple(range(1, gt.ndim))
    )
    valid = valid & finite
    return pred, gt, valid


def ate(pred_xyz: np.ndarray, gt_xyz: np.ndarray, valid=None, with_scale: bool = True) -> float:
    """Absolute Trajectory Error (m) after a single Sim(3) alignment.

    with_scale=True  -> ATE  (scale+R+t fitted to GT, monocular-SLAM standard).
    with_scale=False -> rigid SE(3) alignment (use for ATE-S after the prediction
                        has already been put in metric units; see ``ate_s``).
    """
    pred, gt, valid = _valid_pair(pred_xyz, gt_xyz, valid)
    p, g = pred[valid], gt[valid]
    if len(p) < 3:
        return float("nan")
    s, rot, t = umeyama(p, g, with_scale=with_scale)
    aligned = apply_sim3(p, s, rot, t)
    return float(np.sqrt(((aligned - g) ** 2).sum(axis=1).mean()))


def ate_s(pred_xyz: np.ndarray, gt_xyz: np.ndarray, valid=None) -> float:
    """ATE-S: trajectory error using the prediction's OWN metric scale.

    ``pred_xyz`` must already be in metric units (HaWoR multiplies by its SLAM
    ``scale``; see load_pred). Only a rigid SE(3) alignment is fitted, so any
    error in the predicted absolute scale is penalised.
    """
    return ate(pred_xyz, gt_xyz, valid=valid, with_scale=False)


def _so3_geodesic_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    r = r_a.T @ r_b
    cos = (np.trace(r) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return np.degrees(np.arccos(cos))


def rpe(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    pred_rot: np.ndarray | None = None,
    gt_rot: np.ndarray | None = None,
    delta: int = 1,
    valid=None,
):
    """Relative Pose Error over a fixed frame delta.

    Compares the relative motion (frame i -> i+delta) of prediction and GT.
    Returns dict with translational RMSE (m) and, if rotations are given,
    rotational RMSE (deg). Scale-sensitive (no global scale fit), matching the
    TUM/TartanAir RPE convention.
    """
    pred, gt, valid = _valid_pair(pred_xyz, gt_xyz, valid)
    n = len(pred)
    trans_errs, rot_errs = [], []
    for i in range(n - delta):
        j = i + delta
        if not (valid[i] and valid[j]):
            continue
        d_pred = pred[j] - pred[i]
        d_gt = gt[j] - gt[i]
        trans_errs.append(np.linalg.norm(d_pred - d_gt))
        if pred_rot is not None and gt_rot is not None:
            rel_pred = pred_rot[i].T @ pred_rot[j]
            rel_gt = gt_rot[i].T @ gt_rot[j]
            rot_errs.append(_so3_geodesic_deg(rel_pred, rel_gt))
    out = {
        "rpe_trans": float(np.sqrt(np.mean(np.square(trans_errs)))) if trans_errs else float("nan"),
        "rpe_rot_deg": float(np.sqrt(np.mean(np.square(rot_errs)))) if rot_errs else float("nan"),
        "n_pairs": len(trans_errs),
    }
    return out


# --------------------------------------------------------------------------- #
# Hand-joint metrics
# --------------------------------------------------------------------------- #


def _mpjpe_mm(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-joint position error in mm for (F, J, 3) metre arrays."""
    return float(np.linalg.norm(a - b, axis=-1).mean() * 1000.0)


def pa_mpjpe(pred_joints: np.ndarray, gt_joints: np.ndarray, valid=None) -> float:
    """Procrustes-Aligned MPJPE (mm): per-frame similarity alignment then MPJPE.

    pred_joints, gt_joints: (T, J, 3) in metres. Alignment is per frame with
    scale+R+t (full Procrustes), so it measures local articulation accuracy and
    is independent of the reference frame.
    """
    pred = np.asarray(pred_joints, dtype=np.float64)
    gt = np.asarray(gt_joints, dtype=np.float64)
    assert pred.shape == gt.shape and pred.ndim == 3, (pred.shape, gt.shape)
    if valid is None:
        valid = np.ones(pred.shape[0], dtype=bool)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(pred).all((1, 2)) & np.isfinite(gt).all((1, 2))

    errs = []
    for f in np.where(valid)[0]:
        s, rot, t = umeyama(pred[f], gt[f], with_scale=True)
        aligned = apply_sim3(pred[f], s, rot, t)
        errs.append(np.linalg.norm(aligned - gt[f], axis=-1).mean())
    return float(np.mean(errs) * 1000.0) if errs else float("nan")


def _segment_aligned_mpjpe(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, seg: int, mode: str
) -> float:
    """Shared driver for W-MPJPE / WA-MPJPE over non-overlapping segments."""
    n = len(pred)
    seg_errs = []
    for start in range(0, n, seg):
        end = min(start + seg, n)
        idx = np.where(valid[start:end])[0] + start
        if len(idx) < 3:
            continue
        p = pred[idx].reshape(-1, 3)
        g = gt[idx].reshape(-1, 3)
        if mode == "wa":  # align entire segment (all frames, all joints)
            s, rot, t = umeyama(p, g, with_scale=True)
        elif mode == "w":  # align using the first valid frame only (rigid)
            f0 = idx[0]
            s, rot, t = umeyama(pred[f0], gt[f0], with_scale=False)
        else:
            raise ValueError(mode)
        aligned = apply_sim3(pred[idx], s, rot, t)
        seg_errs.append(np.linalg.norm(aligned - gt[idx], axis=-1).mean())
    return float(np.mean(seg_errs) * 1000.0) if seg_errs else float("nan")


def wa_mpjpe(pred_joints: np.ndarray, gt_joints: np.ndarray, valid=None, seg: int = 100) -> float:
    """World-Aligned MPJPE (mm): single similarity transform per 100-frame segment.

    pred_joints, gt_joints: (T, J, 3) world-frame metres.
    """
    pred = np.asarray(pred_joints, dtype=np.float64)
    gt = np.asarray(gt_joints, dtype=np.float64)
    if valid is None:
        valid = np.ones(pred.shape[0], dtype=bool)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(pred).all((1, 2)) & np.isfinite(gt).all((1, 2))
    return _segment_aligned_mpjpe(pred, gt, valid, seg, mode="wa")


def w_mpjpe(pred_joints: np.ndarray, gt_joints: np.ndarray, valid=None, seg: int = 100) -> float:
    """World MPJPE (mm): align first frame of each 100-frame segment (rigid), then MPJPE.

    Reported for completeness and for the sanity invariant PA <= WA <= W.
    """
    pred = np.asarray(pred_joints, dtype=np.float64)
    gt = np.asarray(gt_joints, dtype=np.float64)
    if valid is None:
        valid = np.ones(pred.shape[0], dtype=bool)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(pred).all((1, 2)) & np.isfinite(gt).all((1, 2))
    return _segment_aligned_mpjpe(pred, gt, valid, seg, mode="w")


if __name__ == "__main__":  # tiny self-test (runnable anywhere, no GPU/data)
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 3))
    # identical -> ~0 error
    assert ate(pts, pts) < 1e-9
    assert pa_mpjpe(pts[:, None, :].repeat(21, 1), pts[:, None, :].repeat(21, 1)) < 1e-6
    # known sim3 -> ATE ~0 after alignment
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    moved = apply_sim3(pts, 2.0, rot, np.array([1.0, 2.0, 3.0]))
    assert ate(moved, pts) < 1e-6
    print("metrics.py self-test OK")
