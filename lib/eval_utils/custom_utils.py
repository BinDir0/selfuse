import copy
import numpy as np
import torch

from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_quaternion, rotation_matrix_to_angle_axis
from scipy.interpolate import interp1d


def cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, handedness):
    init_rot_mat = copy.deepcopy(data_out["init_root_orient"])
    init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w_sla, init_rot_mat)
    init_rot = rotation_matrix_to_angle_axis(init_rot_mat)
    init_rot_quat = angle_axis_to_quaternion(init_rot)
    # data_out["init_root_orient"] = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    # data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
    data_out_init_root_orient = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    data_out_init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])

    init_trans = data_out["init_trans"] # (B, T, 3)
    if handedness == "right":
        outputs = run_mano(data_out["init_trans"], data_out_init_root_orient, data_out_init_hand_pose, betas=data_out["init_betas"])
    elif handedness == "left":
        outputs = run_mano_left(data_out["init_trans"], data_out_init_root_orient, data_out_init_hand_pose, betas=data_out["init_betas"])
    root_loc = outputs["joints"][..., 0, :].cpu()  # (B, T, 3)
    offset = init_trans - root_loc  # It is a constant, no matter what the rotation is.
    init_trans = (
        torch.einsum("tij,btj->bti", R_c2w_sla, root_loc)
        + t_c2w_sla[None, :]
        + offset
    )

    data_world = {
        "init_root_orient": init_rot, # (B, T, 3)
        "init_hand_pose": data_out_init_hand_pose, # (B, T, 15, 3)
        "init_trans": init_trans,  # (B, T, 3)
        "init_betas": data_out["init_betas"]  # (B, T, 10)
    }

    return data_world

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def load_slam_cam(fpath):
    print(f"Loading cameras from {fpath}...")
    pred_cam = dict(np.load(fpath, allow_pickle=True))
    pred_traj = pred_cam["traj"]
    # Use float32 everywhere: traj may be float32 in npz while np.float64(scale) promotes
    # (tensor * scale) to float64, but quaternion_to_matrix stays float32 → einsum dtype error.
    scale = float(pred_cam["scale"])
    t_c2w_sla = torch.as_tensor(pred_traj[:, :3], dtype=torch.float32) * scale
    pred_camq = torch.as_tensor(pred_traj[:, 3:], dtype=torch.float32)
    R_c2w_sla = quaternion_to_matrix(pred_camq[:, [3, 0, 1, 2]])
    R_w2c_sla = R_c2w_sla.transpose(-1, -2)
    t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)
    return R_w2c_sla, t_w2c_sla, R_c2w_sla, t_c2w_sla


def validate_dense_slam_export(fpath):
    """
    Validate that hawor_slam_w_scale export is already dense per video frame.
    Old sparse DPVO exports must be repaired before infiller runs.
    """
    pred_cam = dict(np.load(fpath, allow_pickle=False))
    pred_traj = pred_cam["traj"]
    tstamp = np.asarray(pred_cam.get("tstamp", np.arange(len(pred_traj)))).astype(
        np.int64
    ).reshape(-1)

    dense_by_video_frame = pred_traj.shape[0] != tstamp.shape[0]
    dense_by_contiguous_tstamp = pred_traj.shape[0] == tstamp.shape[0] and np.array_equal(
        tstamp,
        np.arange(pred_traj.shape[0], dtype=np.int64),
    )
    if not dense_by_video_frame and not dense_by_contiguous_tstamp:
        raise RuntimeError(
            f"DPVO infiller requires dense per-frame SLAM cameras; interpolation is disabled for {fpath}. "
            "Repair old exports first, e.g. with "
            "deprecated/unrefactored_tools/scripts/repair_dpvo_dense_slam_exports.py."
        )


def interpolate_slam_cameras_at_video_frames(fpath, video_frame_indices):
    """
    按视频帧号获取 c2w 的 R、t。
    仅支持逐帧 dense 轨迹；如果仍需关键帧插值，则直接报错，由上层将该视频记为 failed。
    """
    pred_cam = dict(np.load(fpath, allow_pickle=True))
    pred_traj = pred_cam["traj"]
    scale = float(pred_cam["scale"])
    tstamp = np.asarray(pred_cam.get("tstamp", np.arange(len(pred_traj)))).astype(
        np.int64
    ).reshape(-1)
    t_c2w = (pred_traj[:, :3] * scale).astype(np.float64)
    pred_camq = torch.tensor(pred_traj[:, 3:])
    R_c2w = quaternion_to_matrix(pred_camq[:, [3, 0, 1, 2]]).numpy()

    vf = np.asarray(video_frame_indices, dtype=np.int64).reshape(-1)

    validate_dense_slam_export(fpath)

    if pred_traj.shape[0] <= 0:
        raise ValueError("empty SLAM trajectory")
    vf_clipped = np.clip(vf, 0, pred_traj.shape[0] - 1)
    return (
        torch.tensor(R_c2w[vf_clipped], dtype=torch.float32),
        torch.tensor(t_c2w[vf_clipped], dtype=torch.float32),
    )


def validate_motion_velocity(bboxes, max_relative_velocity=3.0):
    """
    Validate motion velocity to detect physically implausible movements.
    Uses relative velocity (movement relative to bbox size) instead of absolute pixels.

    Args:
        bboxes: (T, 5) array of [x1, y1, x2, y2, conf]
        max_relative_velocity: Maximum movement as multiple of bbox diagonal per frame

    Returns:
        valid_mask: Boolean array indicating valid frames
    """
    T = bboxes.shape[0]
    if T < 2:
        return np.ones(T, dtype=bool)

    # Calculate bbox centers and sizes
    centers = np.stack([
        (bboxes[:, 0] + bboxes[:, 2]) / 2,
        (bboxes[:, 1] + bboxes[:, 3]) / 2
    ], axis=1)

    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    diagonals = np.sqrt(widths**2 + heights**2)

    # Calculate frame-to-frame displacements
    displacements = np.linalg.norm(centers[1:] - centers[:-1], axis=1)

    # Calculate relative velocities (displacement / average diagonal)
    avg_diagonals = (diagonals[1:] + diagonals[:-1]) / 2
    relative_velocities = np.zeros(T - 1)
    valid_diag_mask = avg_diagonals > 0
    relative_velocities[valid_diag_mask] = displacements[valid_diag_mask] / avg_diagonals[valid_diag_mask]

    # Mark frames with excessive relative velocity as invalid
    valid = np.ones(T, dtype=bool)
    valid[1:] = relative_velocities < max_relative_velocity

    return valid


def interpolate_bboxes(bboxes, max_size_change_ratio=2.5):
    """
    Interpolate missing bboxes with size consistency validation.

    Args:
        bboxes: (T, 5) array of [x1, y1, x2, y2, conf]
        max_size_change_ratio: Maximum allowed size change between adjacent frames
    """
    T = bboxes.shape[0]

    # First pass: filter out bboxes with abnormal size changes
    non_zero_mask = np.any(bboxes != 0, axis=1)
    non_zero_indices = np.where(non_zero_mask)[0]

    if len(non_zero_indices) > 1:
        # Calculate bbox areas
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        areas = widths * heights

        # Check size changes between consecutive valid detections
        for i in range(len(non_zero_indices) - 1):
            curr_idx = non_zero_indices[i]
            next_idx = non_zero_indices[i + 1]

            curr_area = areas[curr_idx]
            next_area = areas[next_idx]

            if curr_area > 0 and next_area > 0:
                size_ratio = max(curr_area, next_area) / min(curr_area, next_area)

                # If size change is too large, mark the detection with lower confidence as invalid
                if size_ratio > max_size_change_ratio:
                    # Keep the one with higher confidence, or the earlier one if confidence is same
                    if bboxes[curr_idx, 4] < bboxes[next_idx, 4]:
                        bboxes[curr_idx] = 0
                        non_zero_mask[curr_idx] = False
                    else:
                        bboxes[next_idx] = 0
                        non_zero_mask[next_idx] = False

        # Update non_zero_indices after filtering
        non_zero_indices = np.where(non_zero_mask)[0]

    zero_indices = np.where(~non_zero_mask)[0]

    if len(zero_indices) == 0 or len(non_zero_indices) == 0:
        return bboxes

    interpolated_bboxes = bboxes.copy()
    for i in range(5):
        interp_func = interp1d(non_zero_indices, bboxes[non_zero_indices, i], kind='linear', fill_value="extrapolate")
        interpolated_bboxes[zero_indices, i] = interp_func(zero_indices)

    return interpolated_bboxes
