"""Render MANO hands on frame. Code extracted from zarr-web-viewer."""
import inspect
import os
import sys
import collections

import numpy as np
import cv2
import torch

MANO_AVAILABLE = False
MANO_LAYER = None
ManoLayer = None
_MANO_LAYERS_CACHE = None
_KNOWN_MANOPTH_PATHS = (
    "/home/guantianrui/manopth",
    "/home/guantianrui",
    "/share_data/lijiayi/EgoYOLO/manopth"
)
MANO_ROOT = os.environ.get("MANO_ROOT", "/share_data/lijiayi/EgoYOLO/manopth/mano/models")

LEFT_MANO_SLICE = slice(0, 45)
RIGHT_MANO_SLICE = slice(45, 90)
LEFT_TRANSLATION_SLICE = slice(0, 3)
RIGHT_TRANSLATION_SLICE = slice(3, 6)
LEFT_ROTATION_SLICE = slice(6, 12)
RIGHT_ROTATION_SLICE = slice(12, 18)
LEFT_SHAPE_SLICE = slice(0, 10)
RIGHT_SHAPE_SLICE = slice(10, 20)
LEFT_FINGERTIPS_SLICE = slice(0, 15)
RIGHT_FINGERTIPS_SLICE = slice(15, 30)

def inject_monkey_patch():
    """Inject inspect.getargspec/formatargspec for legacy deps on Python 3.11+."""
    if hasattr(inspect, "getargspec") and hasattr(inspect, "formatargspec"):
        return

    arg_spec_type = collections.namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    if not hasattr(inspect, "getargspec"):
        def getargspec(func):
            full_spec = inspect.getfullargspec(func)
            return arg_spec_type(
                args=full_spec.args,
                varargs=full_spec.varargs,
                keywords=full_spec.varkw,
                defaults=full_spec.defaults,
            )

        setattr(inspect, "getargspec", getargspec)

    if not hasattr(inspect, "formatargspec"):
        def formatargspec(
            args,
            varargs=None,
            varkw=None,
            defaults=None,
            kwonlyargs=(),
            kwonlydefaults=None,
            annotations=None,
            formatarg=str,
            formatvarargs=lambda name: "*" + name,
            formatvarkw=lambda name: "**" + name,
            formatvalue=lambda value: "=" + repr(value),
            formatreturns=lambda text: " -> " + text,
            formatannotation=None,
        ):
            del kwonlyargs, kwonlydefaults, annotations, formatreturns, formatannotation
            defaults = defaults or ()
            specs = []
            first_default = len(args) - len(defaults) if defaults else -1

            for idx, arg in enumerate(args):
                spec = formatarg(arg)
                if first_default >= 0 and idx >= first_default:
                    spec += formatvalue(defaults[idx - first_default])
                specs.append(spec)

            if varargs is not None:
                specs.append(formatvarargs(varargs))
            if varkw is not None:
                specs.append(formatvarkw(varkw))

            return "(" + ", ".join(specs) + ")"

        setattr(inspect, "formatargspec", formatargspec)

    if not hasattr(np, "int"):
        np.int = int
    
    if not hasattr(np, "float"):
        np.float = float
    
    if not hasattr(np, "bool"):
        np.bool = bool

    if not hasattr(np, "object"):
        np.object = object
    
    if not hasattr(np, "str"):
        np.str = str
    
    if not hasattr(np, "complex"):
        np.complex = complex

    if not hasattr(np, "unicode"):
        np.unicode = str

    if not hasattr(np, "long"):
        np.long = int

inject_monkey_patch()


def _append_existing_paths_to_sys_path(paths):
    for raw_path in paths:
        if not raw_path:
            continue
        path = str(raw_path)
        if path not in sys.path and os.path.exists(path):
            sys.path.insert(0, path)



try:
    _append_existing_paths_to_sys_path(_KNOWN_MANOPTH_PATHS)
    from manopth.manolayer import ManoLayer  # type: ignore

    MANO_AVAILABLE = True
except ImportError:
    print("mano_render: manopth not importable; MANO mesh rendering disabled", file=sys.stderr)
    MANO_AVAILABLE = False



def _instantiate_mano_layers(layer_cls, **layer_kwargs):
    left_layer = layer_cls(side="left", **layer_kwargs)
    right_layer = layer_cls(side="right", **layer_kwargs)
    return {
        "left": left_layer,
        "right": right_layer,
    }


def get_cached_mano_layers():
    """Lazily initialize and cache left/right MANO layers."""
    global _MANO_LAYERS_CACHE

    if _MANO_LAYERS_CACHE is not None:
        return _MANO_LAYERS_CACHE
    if not MANO_AVAILABLE or torch is None:
        print("Error: MANO is not available, cannot initialize layers")
        return None

    layer_cls = ManoLayer
    if layer_cls is None:
        print("Error: ManoLayer import is missing")
        return None

    mano_root = MANO_ROOT
    if not os.path.exists(mano_root):
        print(f"Error: MANO root does not exist: {mano_root}")
        return None

    try:
        _MANO_LAYERS_CACHE = {
            "left": layer_cls(
                mano_root=mano_root,
                use_pca=True,
                ncomps=45,
                flat_hand_mean=True,
                side="left",
                center_idx=0,
            ),
            "right": layer_cls(
                mano_root=mano_root,
                use_pca=True,
                ncomps=45,
                flat_hand_mean=True,
                side="right",
                center_idx=0,
            ),
        }
        print("MANO layers initialized")
        return _MANO_LAYERS_CACHE
    except Exception as e:
        print(f"Error occurred while initializing MANO layers: {e}")
        return None


def split_mano_params(mano_row):
    mano_row = np.asarray(mano_row, dtype=np.float32).reshape(-1)
    if mano_row.size < RIGHT_MANO_SLICE.stop:
        raise ValueError(f"Invalid mano row size: expected >=90 values, got {mano_row.size}")
    return {
        "left": mano_row[LEFT_MANO_SLICE],
        "right": mano_row[RIGHT_MANO_SLICE],
    }


def split_wrist_params(wrist_row):
    wrist_row = np.asarray(wrist_row, dtype=np.float32).reshape(-1)
    if wrist_row.size < RIGHT_ROTATION_SLICE.stop:
        raise ValueError(f"Invalid wrist row size: expected >=18 values, got {wrist_row.size}")
    return {
        "left_translation": wrist_row[LEFT_TRANSLATION_SLICE],
        "right_translation": wrist_row[RIGHT_TRANSLATION_SLICE],
        "left_rotation": wrist_row[LEFT_ROTATION_SLICE],
        "right_rotation": wrist_row[RIGHT_ROTATION_SLICE],
    }


def split_shape_params(shape_row):
    shape_row = np.asarray(shape_row, dtype=np.float32).reshape(-1)
    if shape_row.size < RIGHT_SHAPE_SLICE.stop:
        raise ValueError(f"Invalid shape row size: expected >=20 values, got {shape_row.size}")
    return {
        "left": shape_row[LEFT_SHAPE_SLICE],
        "right": shape_row[RIGHT_SHAPE_SLICE],
    }


def split_fingertips_params(fingertips_row):
    fingertips_row = np.asarray(fingertips_row, dtype=np.float32).reshape(-1)
    if fingertips_row.size < RIGHT_FINGERTIPS_SLICE.stop:
        raise ValueError(f"Invalid fingertips row size: expected >=30 values, got {fingertips_row.size}")
    return {
        "left": fingertips_row[LEFT_FINGERTIPS_SLICE],
        "right": fingertips_row[RIGHT_FINGERTIPS_SLICE],
    }

def render_hand_on_frame(
    frame,
    mano_params=None,
    wrist_params=None,
    extrinsic=None,
    intrinsic=None,
    presence=None,
    shape_params=None,
    mano_layers=None,
    fingertips=None,
    *,
    auto_reframe: bool = True,
):
    """Draw MANO hands on an RGB frame; returns RGB uint8 (see visualize_episode_video-style pipeline).

    presence: 0 none, 1 left, 2 right, 3 both (when not None). fingertips: optional 5*3 world coords per hand.
    If auto_reframe is True and projected mesh points exceed the frame (10px margin), the image is shrunk
    onto a dark gray canvas so geometry stays visible; set False to always use the full frame (e.g. training GT|Pred).
    """
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        frame = (frame * 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    H, W = frame.shape[:2]

    L_present = True
    R_present = True
    if presence is not None:
        L_present = presence in [1, 3]
        R_present = presence in [2, 3]

    fx, fy, cx, cy = 384.0, 384.0, 192.0, 192.0
    if intrinsic is not None:
        parsed = parse_intrinsics(intrinsic)
        if parsed:
            fx, fy, cx, cy = parsed

    if extrinsic is not None:
        if extrinsic.shape == (16,):
            extr_mat = extrinsic.reshape(4, 4)
        else:
            extr_mat = extrinsic
        R_wc = extr_mat[:3, :3]
        t_wc = extr_mat[:3, 3]
    else:
        R_wc = np.eye(3, dtype=np.float32)
        t_wc = np.zeros(3, dtype=np.float32)

    all_projected_points = []
    cached_mano_data = {}

    if MANO_AVAILABLE and mano_layers is not None and mano_params is not None and wrist_params is not None:
        if L_present and 'left' in mano_params and 'left_translation' in wrist_params:
            try:
                L_pose = mano_params['left']
                Lt3 = wrist_params['left_translation']
                L_rot6 = wrist_params['left_rotation']
                L_rotmat = rot6_to_rotmat(L_rot6)
                L_aa = rotmat_to_axisangle(L_rotmat)
                L_shape = shape_params.get('left') if shape_params else None
                
                L_verts_world, L_joints_world = generate_mano_mesh(
                    mano_layers['left'], L_pose, L_aa, Lt3, L_shape
                )
                
                if L_verts_world is not None:
                    cached_mano_data['left'] = {
                        'verts_world': L_verts_world,
                        'joints_world': L_joints_world
                    }
                    
                    verts_cam = (R_wc @ L_verts_world.T).T + t_wc
                    uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                    valid_mask = verts_cam[:, 2] > 1e-6
                    all_projected_points.extend(uv_verts[valid_mask].tolist())
            except Exception:
                pass

        if R_present and 'right' in mano_params and 'right_translation' in wrist_params:
            try:
                R_pose = mano_params['right']
                Rt3 = wrist_params['right_translation']
                R_rot6 = wrist_params['right_rotation']
                R_rotmat = rot6_to_rotmat(R_rot6)
                R_aa = rotmat_to_axisangle(R_rotmat)
                R_shape = shape_params.get('right') if shape_params else None
                
                R_verts_world, R_joints_world = generate_mano_mesh(
                    mano_layers['right'], R_pose, R_aa, Rt3, R_shape
                )
                
                if R_verts_world is not None:
                    cached_mano_data['right'] = {
                        'verts_world': R_verts_world,
                        'joints_world': R_joints_world
                    }
                    
                    verts_cam = (R_wc @ R_verts_world.T).T + t_wc
                    uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                    valid_mask = verts_cam[:, 2] > 1e-6
                    all_projected_points.extend(uv_verts[valid_mask].tolist())
            except Exception:
                pass

    scale_factor = 1.0
    offset_x = 0
    offset_y = 0
    new_W = W
    new_H = H
    img_overlay = frame.copy()

    if auto_reframe and len(all_projected_points) > 0:
        pts = np.array(all_projected_points)
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)

        margin = 10
        out_of_bounds = (min_x < margin or min_y < margin or 
                        max_x > W - margin or max_y > H - margin)
        
        if out_of_bounds:
            required_w = max(max_x - min_x + 2 * margin, W)
            required_h = max(max_y - min_y + 2 * margin, H)

            if np.isfinite(required_w) and np.isfinite(required_h) and required_w > 0 and required_h > 0:
                scale_w = W / required_w
                scale_h = H / required_h
                scale_factor = min(scale_w, scale_h, 0.85)

                if np.isfinite(scale_factor) and scale_factor > 0:
                    scaled_w = max(1, int(round(W * scale_factor)))
                    scaled_h = max(1, int(round(H * scale_factor)))
                    new_W = W
                    new_H = H
                    offset_x = max(0, (new_W - scaled_w) // 2)
                    offset_y = max(0, (new_H - scaled_h) // 2)

                    scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    img_overlay = np.full((new_H, new_W, 3), 40, dtype=np.uint8)
                    img_overlay[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w] = scaled_frame

                    fx = fx * scale_factor
                    fy = fy * scale_factor
                    cx = cx * scale_factor + offset_x
                    cy = cy * scale_factor + offset_y

                    H, W = new_H, new_W

    if MANO_AVAILABLE and mano_layers is not None and mano_params is not None and wrist_params is not None:
        joint_tree = [
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            [(0, 5), (5, 6), (6, 7), (7, 8)],
            [(0, 9), (9, 10), (10, 11), (11, 12)],
            [(0, 13), (13, 14), (14, 15), (15, 16)],
            [(0, 17), (17, 18), (18, 19), (19, 20)],
        ]

        overlay_alpha = img_overlay.copy()

        if L_present and 'left' in mano_params and 'left_translation' in wrist_params:
            if 'left' in cached_mano_data:
                L_verts_world = cached_mano_data['left']['verts_world']
                L_joints_world = cached_mano_data['left']['joints_world']
            else:
                L_pose = mano_params['left']
                Lt3 = wrist_params['left_translation']
                L_rot6 = wrist_params['left_rotation']
                L_rotmat = rot6_to_rotmat(L_rot6)
                L_aa = rotmat_to_axisangle(L_rotmat)
                L_shape = shape_params.get('left') if shape_params else None
                L_verts_world, L_joints_world = generate_mano_mesh(
                    mano_layers['left'], L_pose, L_aa, Lt3, L_shape
                )
            
            if L_verts_world is not None and L_joints_world is not None:
                verts_cam = (R_wc @ L_verts_world.T).T + t_wc
                joints_cam = (R_wc @ L_joints_world.T).T + t_wc
                uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                uv_joints = project_points(joints_cam, fx, fy, cx, cy).astype(np.int32)
                mask_verts = (verts_cam[:, 2] > 1e-6) & \
                            (uv_verts[:, 0] >= 0) & (uv_verts[:, 0] < W) & \
                            (uv_verts[:, 1] >= 0) & (uv_verts[:, 1] < H)
                for pt in uv_verts[mask_verts].astype(np.int32):
                    cv2.circle(overlay_alpha, tuple(pt), 1, (255, 255, 0), -1)
                cv2.addWeighted(overlay_alpha, 0.6, img_overlay, 0.4, 0, img_overlay)
                mask_joints = (joints_cam[:, 2] > 1e-6) & \
                             (uv_joints[:, 0] >= 0) & (uv_joints[:, 0] < W) & \
                             (uv_joints[:, 1] >= 0) & (uv_joints[:, 1] < H)
                for finger_chain in joint_tree:
                    for (j1, j2) in finger_chain:
                        if mask_joints[j1] and mask_joints[j2]:
                            cv2.line(img_overlay, tuple(uv_joints[j1]), tuple(uv_joints[j2]), 
                                   (255, 100, 0), 2)
                for i in range(21):
                    if mask_joints[i]:
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 2, (255, 0, 0), -1)
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 3, (255, 255, 255), 1)

        if R_present and 'right' in mano_params and 'right_translation' in wrist_params:
            if 'right' in cached_mano_data:
                R_verts_world = cached_mano_data['right']['verts_world']
                R_joints_world = cached_mano_data['right']['joints_world']
            else:
                R_pose = mano_params['right']
                Rt3 = wrist_params['right_translation']
                R_rot6 = wrist_params['right_rotation']
                R_rotmat = rot6_to_rotmat(R_rot6)
                R_aa = rotmat_to_axisangle(R_rotmat)
                R_shape = shape_params.get('right') if shape_params else None
                R_verts_world, R_joints_world = generate_mano_mesh(
                    mano_layers['right'], R_pose, R_aa, Rt3, R_shape
                )
            
            if R_verts_world is not None and R_joints_world is not None:
                verts_cam = (R_wc @ R_verts_world.T).T + t_wc
                joints_cam = (R_wc @ R_joints_world.T).T + t_wc
                uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                uv_joints = project_points(joints_cam, fx, fy, cx, cy).astype(np.int32)
                mask_verts = (verts_cam[:, 2] > 1e-6) & \
                            (uv_verts[:, 0] >= 0) & (uv_verts[:, 0] < W) & \
                            (uv_verts[:, 1] >= 0) & (uv_verts[:, 1] < H)
                overlay_alpha2 = img_overlay.copy()
                for pt in uv_verts[mask_verts].astype(np.int32):
                    cv2.circle(overlay_alpha2, tuple(pt), 1, (0, 255, 255), -1)
                cv2.addWeighted(overlay_alpha2, 0.6, img_overlay, 0.4, 0, img_overlay)
                mask_joints = (joints_cam[:, 2] > 1e-6) & \
                             (uv_joints[:, 0] >= 0) & (uv_joints[:, 0] < W) & \
                             (uv_joints[:, 1] >= 0) & (uv_joints[:, 1] < H)
                for finger_chain in joint_tree:
                    for (j1, j2) in finger_chain:
                        if mask_joints[j1] and mask_joints[j2]:
                            cv2.line(img_overlay, tuple(uv_joints[j1]), tuple(uv_joints[j2]), 
                                   (0, 100, 255), 2)
                for i in range(21):
                    if mask_joints[i]:
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 2, (0, 0, 255), -1)
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 3, (255, 255, 255), 1)

    if fingertips is not None:
        if L_present and 'left' in fingertips:
            left_tips = np.array(fingertips['left']).reshape(5, 3)
            tips_cam = (R_wc @ left_tips.T).T + t_wc
            uv_tips = project_points(tips_cam, fx, fy, cx, cy).astype(np.int32)
            mask_tips = (tips_cam[:, 2] > 1e-6) & \
                       (uv_tips[:, 0] >= 0) & (uv_tips[:, 0] < W) & \
                       (uv_tips[:, 1] >= 0) & (uv_tips[:, 1] < H)

            for i in range(5):
                if mask_tips[i]:
                    pt = tuple(uv_tips[i])
                    cv2.circle(img_overlay, pt, 3, (0, 255, 0), 1)
                    cv2.circle(img_overlay, pt, 2, (255, 255, 0), -1)
                    cv2.putText(img_overlay, f"L{i}", (pt[0]+5, pt[1]-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        if R_present and 'right' in fingertips:
            right_tips = np.array(fingertips['right']).reshape(5, 3)
            tips_cam = (R_wc @ right_tips.T).T + t_wc
            uv_tips = project_points(tips_cam, fx, fy, cx, cy).astype(np.int32)
            mask_tips = (tips_cam[:, 2] > 1e-6) & \
                       (uv_tips[:, 0] >= 0) & (uv_tips[:, 0] < W) & \
                       (uv_tips[:, 1] >= 0) & (uv_tips[:, 1] < H)

            for i in range(5):
                if mask_tips[i]:
                    pt = tuple(uv_tips[i])
                    cv2.circle(img_overlay, pt, 3, (255, 0, 255), 1)
                    cv2.circle(img_overlay, pt, 2, (0, 255, 255), -1)
                    cv2.putText(img_overlay, f"R{i}", (pt[0]+5, pt[1]-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
    
    else:
        if wrist_params is not None and intrinsic is not None:
            if L_present and 'left_translation' in wrist_params:
                wrist_pos = wrist_params['left_translation']
                if wrist_pos[2] > 0:
                    wrist_cam = R_wc @ wrist_pos + t_wc
                    if wrist_cam[2] > 0:
                        x = int(fx * wrist_cam[0] / wrist_cam[2] + cx)
                        y = int(fy * wrist_cam[1] / wrist_cam[2] + cy)
                        if 0 <= x < W and 0 <= y < H:
                            cv2.circle(img_overlay, (x, y), 8, (255, 255, 0), -1)
                            cv2.putText(img_overlay, "L", (x+10, y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            if R_present and 'right_translation' in wrist_params:
                wrist_pos = wrist_params['right_translation']
                if wrist_pos[2] > 0:
                    wrist_cam = R_wc @ wrist_pos + t_wc
                    if wrist_cam[2] > 0:
                        x = int(fx * wrist_cam[0] / wrist_cam[2] + cx)
                        y = int(fy * wrist_cam[1] / wrist_cam[2] + cy)
                        if 0 <= x < W and 0 <= y < H:
                            cv2.circle(img_overlay, (x, y), 8, (0, 255, 255), -1)
                            cv2.putText(img_overlay, "R", (x+10, y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
    return img_overlay




def rot6_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """Zhou 6D rotation (6,) -> (3,3) rotation matrix."""
    a1 = r6[:3]
    a2 = r6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    return R.astype(np.float32)


def rotmat_to_axisangle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) -> axis-angle (3,)."""
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


def project_points(pts_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Project camera-frame 3D points to pixel coordinates."""
    zs = pts_cam[:, 2] + 1e-8
    us = fx * (pts_cam[:, 0] / zs) + cx
    vs = fy * (pts_cam[:, 1] / zs) + cy
    return np.stack([us, vs], axis=1)


def parse_intrinsics(intrinsics):
    """Return (fx, fy, cx, cy) from 3x3, 9-elem, or 4-elem intrinsics."""
    if intrinsics is None:
        return None
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        return intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    if intrinsics.shape == (9,):
        intrinsics = intrinsics.reshape(3, 3)
        return intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    if intrinsics.shape == (4,):
        return intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    print(f"Warning: Unrecognized intrinsics shape {intrinsics.shape}, using default values")
    return None


def generate_mano_mesh(mano_layer, pose_params, global_r_aa, trans, shape_params=None):
    """Run ManoLayer; return verts (778,3) and joints (21,3) in meters + translation."""
    if not MANO_AVAILABLE or mano_layer is None:
        print(mano_layer)
        return None, None

    if pose_params.shape[-1] == 15:
        theta = np.concatenate([global_r_aa.reshape(1, 3), pose_params.reshape(1, 15)], axis=1)
    else:
        theta = np.concatenate([global_r_aa.reshape(1, 3), pose_params.reshape(1, 45)], axis=1)
    theta_t = torch.from_numpy(theta).float()

    if shape_params is not None:
        beta_t = torch.from_numpy(shape_params.reshape(1, 10)).float()
    else:
        beta_t = torch.zeros((1, 10), dtype=torch.float32)

    with torch.no_grad():
        verts, joints = mano_layer(theta_t, beta_t)
        verts_np = verts.detach().cpu().numpy()[0] / 1000.0 + trans.reshape(1, 3)
        joints_np = joints.detach().cpu().numpy()[0] / 1000.0 + trans.reshape(1, 3)
    
    return verts_np.astype(np.float32), joints_np.astype(np.float32)
