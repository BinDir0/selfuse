#!/usr/bin/env python3
import os
import sys
import argparse
import random
import numpy as np
import zarr
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

AXIS_BASE = np.array([[0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
AXIS_TRANSFORM = np.linalg.inv(AXIS_BASE).astype(np.float32)

def rot6_to_rotmat(r6: np.ndarray) -> np.ndarray:
    a1 = r6[:3]
    a2 = r6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    return R.astype(np.float32)


def rotmat_to_axisangle(R: np.ndarray) -> np.ndarray:
    R = R.astype(np.float32)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return (axis * theta).astype(np.float32)


def rotation_diff_axis_angle(Ra: np.ndarray, Rb: np.ndarray) -> tuple:
    """Return (angle_deg, axis_vec) for Rdiff = Ra @ Rb^T with robust axis for angles near 180."""
    Rdiff = Ra @ Rb.T
    tr = np.trace(Rdiff)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(c)))
    # Try standard axis from skew-symmetric part
    axis = np.array([
        Rdiff[2, 1] - Rdiff[1, 2],
        Rdiff[0, 2] - Rdiff[2, 0],
        Rdiff[1, 0] - Rdiff[0, 1],
    ], dtype=np.float64)
    n = np.linalg.norm(axis)
    if n > 1e-8:
        axis = (axis / n).astype(np.float32)
    else:
        # Fallback: eigenvector with eigenvalue ~1
        vals, vecs = np.linalg.eig(Rdiff)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        axis = np.real(vecs[:, idx])
        axis = (axis / (np.linalg.norm(axis) + 1e-8)).astype(np.float32)
    return ang, axis


def plot_mano_on_axes(ax, verts: np.ndarray, faces: np.ndarray, color: str):
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor(color)
    mesh.set_edgecolor('k')
    mesh.set_linewidth(0.2)
    ax.add_collection3d(mesh)


def try_import_manolayer(candidates: list):
    last_err = None
    for p in candidates:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
        try:
            from manopth.manolayer import ManoLayer  # noqa: F401
            return 'manopth.manolayer', ManoLayer
        except Exception as e:
            last_err = e
        try:
            from manopth.manopth.manolayer import ManoLayer  # noqa: F401
            return 'manopth.manopth.manolayer', ManoLayer
        except Exception as e:
            last_err = e
    raise last_err if last_err else ImportError('Failed to import manopth ManoLayer')


def load_intrinsics(path: str) -> tuple:
    try:
        with open(path, 'r') as f:
            txt = f.read()
        nums = [float(x) for x in txt.replace(',', ' ').split() if x.strip()]
        if len(nums) >= 4:
            fx, fy, cx, cy = nums[0], nums[1], nums[2], nums[3]
            return fx, fy, cx, cy
    except Exception:
        pass
    return None


def project_points(pts_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    zs = pts_cam[:, 2] + 1e-8
    us = fx * (pts_cam[:, 0] / zs) + cx
    vs = fy * (pts_cam[:, 1] / zs) + cy
    return np.stack([us, vs], axis=1)

def rot_axis_angle(R):
    tr = np.trace(R)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(c))
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=np.float64)
    n = np.linalg.norm(axis) + 1e-12
    axis = (axis / n).astype(np.float64)
    return axis, angle  # angle in radians

def main():
    parser = argparse.ArgumentParser(description='Visualize a random TACO Zarr frame (image + instruction + MANO hands)')
    parser.add_argument('--zarr', type=str, required=True, help='Path to TACO zarr store')
    parser.add_argument('--mano_root', type=str, default='/home/guantianrui/manopth/mano/models', help='MANO model directory')
    parser.add_argument('--manopth_path', type=str, default='/home/guantianrui/manopth', help='Path to local manopth repo or its parent (added to sys.path)')
    parser.add_argument('--index', type=int, default=-1, help='Specific frame index to visualize (default random)')
    parser.add_argument('--save', type=str, default='', help='If set, save visualization to this path instead of showing')
    parser.add_argument('--camera_view', action='store_true', help='Project MANO meshes to image using camera intrinsics and overlay')
    parser.add_argument('--coord_frame', type=str, default='world', choices=['world', 'camera'], help='3D rendering frame when not overlaying')
    parser.add_argument('--intrinsic_path', type=str, default='', help='Path to intrinsics text (fx fy cx cy)')
    parser.add_argument('--fx', type=float, default=0, help='fx if no intrinsic file provided')
    parser.add_argument('--fy', type=float, default=0, help='fy if no intrinsic file provided')
    parser.add_argument('--cx', type=float, default=0, help='cx if no intrinsic file provided')
    parser.add_argument('--cy', type=float, default=0, help='cy if no intrinsic file provided')
    args = parser.parse_args()

    repo = args.manopth_path.strip()
    parent = os.path.dirname(os.path.normpath(repo)) if repo else ''
    subpkg = os.path.join(repo, 'manopth') if repo else ''
    candidates = [repo, parent, subpkg, '/home/guantianrui']
    modname, ManoLayer = try_import_manolayer(candidates)

    import torch

    def build_mano_layer(mano_root: str, side: str) -> ManoLayer:
        return ManoLayer(mano_root=mano_root, use_pca=True, ncomps=15, flat_hand_mean=True, side=side, center_idx=0)

    def mano_vertices_from_params(mano_layer, pose15: np.ndarray, global_r_aa: np.ndarray, trans: np.ndarray) -> np.ndarray:
        theta = np.concatenate([global_r_aa.reshape(1, 3), pose15.reshape(1, 15)], axis=1)
        theta_t = torch.from_numpy(theta).float()
        beta_t = torch.zeros((1, 10), dtype=torch.float32)
        verts, joints = mano_layer(theta_t, beta_t)
        verts_np = verts.detach().cpu().numpy()[0] / 1000.0
        verts_np = verts_np + trans.reshape(1, 3)
        # print(global_r_aa)
        # joints_np = joints.detach().cpu().numpy()[0] / 1000.0
        # joints_np = joints_np + trans.reshape(1, 3)
        # print(joints_np)
        return verts_np.astype(np.float32)

    def mano_joints_from_params(mano_layer, pose15: np.ndarray, global_r_aa: np.ndarray, trans: np.ndarray) -> np.ndarray:
        theta = np.concatenate([global_r_aa.reshape(1, 3), pose15.reshape(1, 15)], axis=1)
        theta_t = torch.from_numpy(theta).float()
        beta_t = torch.zeros((1, 10), dtype=torch.float32)
        verts, joints = mano_layer(theta_t, beta_t)
        # verts_np = verts.detach().cpu().numpy()[0] / 1000.0
        # verts_np = verts_np + trans.reshape(1, 3)
        # print(global_r_aa)
        joints_np = joints.detach().cpu().numpy()[0] / 1000.0
        joints_np = joints_np + trans.reshape(1, 3)
        # print(joints_np)
        return joints_np.astype(np.float32)

    store = zarr.DirectoryStore(args.zarr)
    root = zarr.group(store=store, overwrite=False)
    data = root['data']

    img_arr = data['image']
    instr_arr = data['instruction']
    state_hand = data['state']['hand']
    state_wrist = data['state']['wrist']

    # extrinsic is optional but preferred for camera_view; assumed world->camera
    extrinsic_ds = data.get('extrinsic', None)

    S = state_hand.shape[0]
    if S == 0:
        print('Empty dataset')
        return
    idx = args.index if args.index >= 0 else random.randrange(S)
    # idx = 13041
    img = img_arr[idx]
    H, W = img.shape[0], img.shape[1]
    instr = str(instr_arr[idx])
    hand30 = state_hand[idx]
    wrist18 = state_wrist[idx]

    L_pose15 = hand30[:15]
    R_pose15 = hand30[15:30]
    # L_pose15 = np.zeros(15)
    # R_pose15 = np.zeros(15)
    print(L_pose15, R_pose15)

    Lt3 = wrist18[0:3]
    Rt3 = wrist18[3:6]
    Lrot6 = wrist18[6:12]
    Rrot6 = wrist18[12:18]

    L_R = rot6_to_rotmat(Lrot6)
    R_R = rot6_to_rotmat(Rrot6)
    
    L_aa = rotmat_to_axisangle(L_R)
    R_aa = rotmat_to_axisangle(R_R)
    # diff = L_R @ R_R.T
    # axis, angle = rot_axis_angle(diff)  # Rdiff = R_right @ R_left.T
    # print('angle_deg=', np.degrees(angle), 'axis=', axis)
    # print(np.linalg.det(L_R @ R_R.T))
    print(L_aa, R_aa)

    # Diagnose relative rotation between hands in the same frame (world frame)
    # try:
    #     ang_deg, axis_vec = rotation_diff_axis_angle(R_R, L_R)
    #     sims = [abs(float(np.dot(axis_vec, np.array([1, 0, 0], dtype=np.float32)))),
    #             abs(float(np.dot(axis_vec, np.array([0, 1, 0], dtype=np.float32)))),
    #             abs(float(np.dot(axis_vec, np.array([0, 0, 1], dtype=np.float32))))]
    #     axis_names = ['x', 'y', 'z']
    #     top_axis = axis_names[int(np.argmax(sims))]
    #     print(f"[Diag] Rdiff angle(deg)≈{ang_deg:.2f}, axis≈{axis_vec.tolist()}, closest_axis={top_axis} (|dot|={max(sims):.3f})")
    # except Exception as e:
    #     print(f"[Diag] Failed to compute Rdiff axis/angle: {e}")

    mano_left = build_mano_layer(args.mano_root, side='left')
    mano_right = build_mano_layer(args.mano_root, side='right')

    # World-space vertices
    L_verts_world = mano_vertices_from_params(mano_left, L_pose15, L_aa, Lt3)
    R_verts_world = mano_vertices_from_params(mano_right, R_pose15, R_aa, Rt3)
    L_joints_world = mano_joints_from_params(mano_left, L_pose15, L_aa, Lt3)
    R_joints_world = mano_joints_from_params(mano_right, R_pose15, R_aa, Rt3)
    # print(Lt3, Rt3, extrinsic_ds[idx])

    faces = mano_left.th_faces.numpy().astype(np.int32)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.set_title(f'Idx {idx}\n{instr}')
    ax1.axis('off')

    if args.camera_view:
        # Prepare intrinsics
        K = None
        if args.intrinsic_path:
            K = load_intrinsics(args.intrinsic_path)
        if (not K) and args.fx > 0 and args.fy > 0:
            K = (args.fx, args.fy, args.cx if args.cx > 0 else W / 2.0, args.cy if args.cy > 0 else H / 2.0)
        if not K:
            f = max(W, H)
            K = (f, f, W / 2.0, H / 2.0)
        fx, fy, cx, cy = K
        # World->Camera using extrinsic
        if extrinsic_ds is None:
            print('Warning: data/extrinsic not found; assume identity world->camera')
            M_wc = np.eye(4, dtype=np.float32)
        else:
            M_wc = extrinsic_ds[idx].reshape(4, 4).astype(np.float32)
        R_wc = M_wc[:3, :3]
        t_wc = M_wc[:3, 3]
        # Transform to camera coords
        # print(L_joints_world[8] - L_joints_world[0])
        # print(Lt3 - Rt3)
        # print((Lt3 - Rt3) @ (L_joints_world[8] - L_joints_world[0]))
        for verts_world, color in [(R_verts_world, 'yellow'), (R_joints_world, 'red')]:
            verts_cam = (R_wc @ verts_world.T).T + t_wc.reshape(1, 3)
            # print(t_wc.reshape(1, 3))
            # print((R_wc @ Lt3.T).T + t_wc.reshape(1, 3), (R_wc @ Rt3.T).T + t_wc.reshape(1, 3))
            # 将两个手腕的点投影到图像上，并可视化出来
            # uv_L = project_points((R_wc @ Lt3.T).T + t_wc.reshape(1, 3), fx, fy, cx, cy)
            # uv_R = project_points((R_wc @ Rt3.T).T + t_wc.reshape(1, 3), fx, fy, cx, cy)
            # ax1.scatter(uv_L[:, 0], uv_L[:, 1], s=5.0, c='blue', alpha=0.8)
            # ax1.scatter(uv_R[:, 0], uv_R[:, 1], s=5.0, c='red', alpha=0.8)
            # print(np.mean(verts_cam[:,2]))
            # if color == 'blue' or color == 'red':
            #     print(verts_cam[8] - verts_cam[0])
            #     print((R_wc @ Lt3.T).T + t_wc.reshape(1, 3) - ((R_wc @ Rt3.T).T + t_wc.reshape(1, 3)))
            #     print(((R_wc @ Lt3.T).T + t_wc.reshape(1, 3) - ((R_wc @ Rt3.T).T + t_wc.reshape(1, 3))) @ (verts_cam[8] - verts_cam[0]))
            uv = project_points(verts_cam, fx, fy, cx, cy)
            # print(f"verts_cam x: {np.min(verts_cam[:, 0])}, {np.max(verts_cam[:, 0])}")
            # print(f"verts_cam y: {np.min(verts_cam[:, 1])}, {np.max(verts_cam[:, 1])}")
            # print(f"verts_cam z: {np.min(verts_cam[:, 2])}, {np.max(verts_cam[:, 2])}")
            # print(f"uv x: {np.min(uv[:, 0])}, {np.max(uv[:, 0])}")
            # print(f"uv y: {np.min(uv[:, 1])}, {np.max(uv[:, 1])}")
            mask = verts_cam[:, 2] > 1e-6
            uv = uv[mask]
            ax1.scatter(uv[:, 0], uv[:, 1], s=1.0, c=color, alpha=0.8)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'Camera-view overlay shown on the left image', ha='center', va='center')
    else:
        # 3D rendering without overlay: support world or camera frame
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        title = 'Reconstructed MANO (world frame)'

        plot_mano_on_axes(ax2, L_verts_world, faces, color='#8ecae6')
        plot_mano_on_axes(ax2, R_verts_world, faces, color='#ffb703')
        all_pts = np.concatenate([L_verts_world, R_verts_world], axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        center = (mins + maxs) / 2.0
        extent = (maxs - mins).max() * 0.6 + 1e-3
        ax2.set_xlim(center[0] - extent, center[0] + extent)
        ax2.set_ylim(center[1] - extent, center[1] + extent)
        ax2.set_zlim(center[2] - extent, center[2] + extent)
        ax2.set_box_aspect([1, 1, 1])
        ax2.set_title(title)

    plt.tight_layout()

    backend = matplotlib.get_backend().lower()
    has_display = bool(os.environ.get('DISPLAY')) or sys.platform.startswith(('win', 'darwin'))
    out_path = args.save
    if out_path or (not has_display) or ('agg' in backend):
        if not out_path:
            out_path = f'viz_{idx}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization to {out_path}')
    else:
        plt.show()


if __name__ == '__main__':
    main() 