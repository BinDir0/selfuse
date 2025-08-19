#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import zarr
import cv2

INSTRUCTION_PLACEHOLDER = "finish the task"
INSTRUCTION_MAX_CHARS = 512


def axis_angle_to_rotmat(axis_angle: np.ndarray) -> np.ndarray:
    aa = np.asarray(axis_angle, dtype=np.float32)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = aa / angle
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    R = np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C    ],
    ], dtype=np.float32)
    return R


def rotmat_to_rot6(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def collect_mano_pairs(root_dir: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Scan for *_left_mano.npy and *_right_mano.npy and pair by basename.

    Returns list of (base_key, left_path, right_path) sorted by base_key.
    """
    left_map: Dict[str, str] = {}
    right_map: Dict[str, str] = {}
    for cur_root, dirs, files in os.walk(root_dir):
        for f in files:
            if not f.endswith("_mano.npy"):
                continue
            path = os.path.join(cur_root, f)
            name = f
            if name.endswith("_left_mano.npy"):
                base = name[: -len("_left_mano.npy")]
                left_map[base] = path
            elif name.endswith("_right_mano.npy"):
                base = name[: -len("_right_mano.npy")]
                right_map[base] = path
    keys = sorted(set(left_map.keys()) | set(right_map.keys()))
    pairs: List[Tuple[str, Optional[str], Optional[str]]] = []
    for k in keys:
        pairs.append((k, left_map.get(k, None), right_map.get(k, None)))
    return pairs


def load_mano_npy(npy_path: str) -> Dict:
    data = np.load(npy_path, allow_pickle=True).item()
    return data


def forward_fill_rows(a: np.ndarray) -> np.ndarray:
    if a is None or a.size == 0:
        return a
    out = a.copy()
    last = out[0]
    for i in range(out.shape[0]):
        if i > 0 and np.allclose(out[i], 0):
            out[i] = last
        else:
            last = out[i]
    return out


def load_intrinsics(path: str) -> tuple:
    """Load camera intrinsics from a text file containing 3x3 intrinsic matrix.
    Returns tuple of (fx, fy, cx, cy) or None if loading fails.
    """
    try:
        with open(path, 'r') as f:
            txt = f.read()
        nums = [float(x) for x in txt.replace(',', ' ').split() if x.strip()]
        if len(nums) >= 9:
            # 3x3 matrix format: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            fx, _, cx = nums[0], nums[1], nums[2]
            _, fy, cy = nums[3], nums[4], nums[5]
            return fx, fy, cx, cy
    except Exception:
        pass
    return None


def ensure_T(arr: Optional[np.ndarray], T: int, dim: int) -> np.ndarray:
    if arr is None or arr.size == 0:
        return np.zeros((T, dim), dtype=np.float32)
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    # time pad/truncate
    if a.shape[0] < T:
        pad = np.zeros((T - a.shape[0], a.shape[1]), dtype=np.float32)
        a = np.concatenate([a, pad], axis=0)
    elif a.shape[0] > T:
        a = a[:T]
    # dim pad/truncate
    if a.shape[1] != dim:
        out = np.zeros((T, dim), dtype=np.float32)
        use = min(dim, a.shape[1])
        out[:, :use] = a[:, :use]
        a = out
    return a


def resolve_mano_root(data_root: str) -> str:
    """Return the directory that contains *_mano.npy files.
    Accept either data_root itself or data_root/Mano_Poses.
    """
    if any(fname.endswith("_mano.npy") for fname in os.listdir(data_root) if os.path.isfile(os.path.join(data_root, fname))):
        return data_root
    candidate = os.path.join(data_root, "Mano_Poses")
    if os.path.isdir(candidate):
        return candidate
    return data_root


def make_instruction_from_episode_dir(dir_path: Optional[str]) -> str:
    """Derive an English instruction from episode folder name formatted as <action,tool,object>.

    Fallbacks: try comma, underscore, hyphen, or space separated triples. If parsing fails,
    return the folder name as-is (truncated to max length).
    """
    if not dir_path:
        return INSTRUCTION_PLACEHOLDER
    name = os.path.basename(dir_path)
    # Strip common wrappers
    cleaned = name.replace("<", "").replace(">", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "")
    candidates = []
    for sep in [",", "_", "-", " "]:
        parts = [p.strip() for p in cleaned.split(sep) if p.strip()]
        if len(parts) == 3:
            candidates = parts
            break
    if len(candidates) == 3:
        action, tool, obj = candidates[0], candidates[1], candidates[2]
        sent = f"{action} the {obj} with the {tool}."
        return sent[:INSTRUCTION_MAX_CHARS]
    # Fallback
    return cleaned[:INSTRUCTION_MAX_CHARS]


def find_triple_dir_for_path(file_path: Optional[str], max_up: int = 8) -> Optional[str]:
    """Given a file path inside an episode tree, walk up ancestors to find a folder
    whose name looks like a triple: (a, b, c) or a,b,c. Return that folder path.
    """
    if not file_path:
        return None
    cur = os.path.dirname(file_path)
    for _ in range(max_up):
        name = os.path.basename(cur)
        # print(f"name: {name}")
        if not name:
            break
        # Accept names with exactly two commas, optionally wrapped by parentheses or brackets
        base = name.strip()
        # Quick check: two commas
        if base.count(',') == 2:
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def make_instruction_from_filename(file_path: Optional[str]) -> Optional[str]:
    """Try to parse instruction triple from *_mano.npy filename, e.g.
    (brush, brush, bowl)_20230919_036_left_mano.npy -> "Please brush the bowl with the brush."
    """
    if not file_path:
        return None
    base = os.path.basename(file_path)
    stem, _ = os.path.splitext(base)
    # Take token before first underscore
    if '_' in stem:
        triple_token = stem.split('_', 1)[0]
    else:
        triple_token = stem
    cleaned = triple_token.replace("<", "").replace(">", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "")
    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
    if len(parts) == 3:
        action, tool, obj = parts[0], parts[1], parts[2]
        sent = f"{action} the {obj} with the {tool}."
        return sent[:INSTRUCTION_MAX_CHARS]
    return None


def derive_instruction(left_path: Optional[str], right_path: Optional[str]) -> str:
    """Derive instruction by (1) filename triple, (2) ancestor folder triple, else placeholder."""
    txt = make_instruction_from_filename(left_path)
    if txt:
        return txt
    txt = make_instruction_from_filename(right_path)
    if txt:
        return txt
    ep_dir = find_triple_dir_for_path(left_path)
    if ep_dir is None:
        ep_dir = find_triple_dir_for_path(right_path)
    return make_instruction_from_episode_dir(ep_dir)


def parse_triple_and_session_from_filename(file_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """From *_mano.npy filename parse (triple_dir_name, session_id).
    Example: (brush, brush, bowl)_20230919_036_left_mano.npy -> ("(brush, brush, bowl)", "20230919_036")
    """
    if not file_path:
        return None, None
    base = os.path.basename(file_path)
    stem, _ = os.path.splitext(base)
    tokens = stem.split('_')
    if len(tokens) < 3:
        return None, None
    triple_token = tokens[0]
    # Re-add parentheses if absent but we expect Egocentric_Camera_Parameters uses parentheses
    triple_dir = triple_token if triple_token.startswith('(') else f"({triple_token})"
    session = f"{tokens[1]}_{tokens[2]}"
    return triple_dir, session


def load_episode_extrinsics(data_root: str, any_side_path: Optional[str], T: int) -> np.ndarray:
    """Load egocentric 4x4 extrinsic matrices for an episode; return [T,16] float32 aligned via pad/truncate.
    If missing or malformed, return zeros.
    """
    triple_dir, session = parse_triple_and_session_from_filename(any_side_path)
    if triple_dir is None or session is None:
        return np.zeros((T, 16), dtype=np.float32)
    extr_path = os.path.join(data_root, 'Egocentric_Camera_Parameters', triple_dir, session, 'egocentric_frame_extrinsic.npy')
    try:
        extr = np.load(extr_path, allow_pickle=True)
        # Accept shapes [Tc,4,4] or [Tc,16]
        if extr.ndim == 3 and extr.shape[1:] == (4, 4):
            extr_flat = extr.reshape(extr.shape[0], 16).astype(np.float32)
        elif extr.ndim == 2 and extr.shape[1] == 16:
            extr_flat = extr.astype(np.float32)
        else:
            return np.zeros((T, 16), dtype=np.float32)
        # Align to T
        if extr_flat.shape[0] < T:
            pad = np.zeros((T - extr_flat.shape[0], 16), dtype=np.float32)
            extr_flat = np.concatenate([extr_flat, pad], axis=0)
        elif extr_flat.shape[0] > T:
            extr_flat = extr_flat[:T]
        # Forward-fill zeros rows
        extr_flat = forward_fill_rows(extr_flat)
        return extr_flat
    except Exception as e:
        # print(f"[WARN] load extrinsic failed for {extr_path}: {e}")
        return np.zeros((T, 16), dtype=np.float32)


def load_episode_intrinsics(data_root: str, any_side_path: Optional[str], T: int, img_size: int = 384) -> np.ndarray:
    """Load camera intrinsics for an episode and scale for resized images.
    Returns [T, 4] float32 array with (fx, fy, cx, cy) per frame aligned via pad/truncate.
    If missing or malformed, return default intrinsics.
    """
    triple_dir, session = parse_triple_and_session_from_filename(any_side_path)
    if triple_dir is None or session is None:
        # Default intrinsics for 384x384 images
        f = img_size
        return np.full((T, 4), [f, f, img_size / 2.0, img_size / 2.0], dtype=np.float32)
    
    intr_path = os.path.join(data_root, 'Egocentric_Camera_Parameters', triple_dir, session, 'egocentric_intrinsic.txt')
    try:
        K = load_intrinsics(intr_path)
        if K is None:
            # Default intrinsics for 384x384 images
            f = img_size
            fx, fy, cx, cy = f, f, img_size / 2.0, img_size / 2.0
        else:
            fx, fy, cx, cy = K
            # Scale intrinsics based on image resizing
            # Assume original TACO images are 1920x1080 (common for egocentric cameras)
            # When resizing to 384x384, we need to scale intrinsics accordingly
            original_width = 1920.0
            original_height = 1080.0
            
            # Scale factors for x and y directions
            scale_x = img_size / original_width
            scale_y = img_size / original_height
            
            # Apply scaling to intrinsic parameters
            fx_scaled = fx * scale_x
            fy_scaled = fy * scale_y
            cx_scaled = cx * scale_x
            cy_scaled = cy * scale_y
            
            fx, fy, cx, cy = fx_scaled, fy_scaled, cx_scaled, cy_scaled
        
        # Create array with same intrinsics for all T frames
        intrinsics_per_frame = np.full((T, 4), [fx, fy, cx, cy], dtype=np.float32)
        return intrinsics_per_frame
        
    except Exception as e:
        print(f"[WARN] load intrinsic failed for {intr_path}: {e}")
        # Default intrinsics for 384x384 images
        f = img_size
        return np.full((T, 4), [f, f, img_size / 2.0, img_size / 2.0], dtype=np.float32)


def load_episode_images(data_root: str, any_side_path: Optional[str], T: int, img_size: int = 384) -> np.ndarray:
    """Load RGB frames for an episode and align to length T.
    Source: Egocentric_RGB_Videos/(action, tool, object)/{session}/color.mp4
    Returns [T, img_size, img_size, 3] uint8. Missing frames are forward-filled.
    """
    triple_dir, session = parse_triple_and_session_from_filename(any_side_path)
    if triple_dir is None or session is None:
        return np.zeros((T, img_size, img_size, 3), dtype=np.uint8)
    vid_path = os.path.join(data_root, 'Egocentric_RGB_Videos', triple_dir, session, 'color.mp4')
    if not os.path.isfile(vid_path):
        return np.zeros((T, img_size, img_size, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(vid_path)
    frames: List[np.ndarray] = []
    if not cap.isOpened():
        return np.zeros((T, img_size, img_size, 3), dtype=np.uint8)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                continue
            # BGR to RGB
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # unexpected channels
                frame = np.repeat(frame[:, :, :1], 3, axis=2)
            # Resize to square img_size
            img = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
            img = np.clip(img, 0, 255).astype(np.uint8)
            frames.append(img)
    finally:
        cap.release()
    if len(frames) == 0:
        return np.zeros((T, img_size, img_size, 3), dtype=np.uint8)
    imgs = np.stack(frames, axis=0)  # [Tv, H, W, 3]
    # Align to T with pad/truncate and forward-fill
    Tv = imgs.shape[0]
    if Tv < T:
        pad = np.zeros((T - Tv, img_size, img_size, 3), dtype=np.uint8)
        imgs = np.concatenate([imgs, pad], axis=0)
    elif Tv > T:
        imgs = imgs[:T]
    # Forward-fill zeros (if any)
    out = imgs.copy()
    for i in range(1, T):
        if not out[i].any():
            out[i] = out[i - 1]
    return out


def build_taco_zarr(data_root: str, output_zarr: str):
    os.makedirs(os.path.dirname(output_zarr), exist_ok=True)
    store = zarr.DirectoryStore(output_zarr)
    root = zarr.group(store=store, overwrite=True)

    # Root groups: data and meta
    data_grp = root.create_group("data")
    meta_grp = root.create_group("meta")

    # Under data/: state and action groups
    state_grp = data_grp.create_group("state")
    action_grp = data_grp.create_group("action")

    # Concatenated long arrays with appendable first dim
    state_hand_ds = state_grp.create_dataset(
        "hand", shape=(0, 30), chunks=(65536, 30), dtype=np.float32, overwrite=True, maxshape=(None, 30)
    )
    state_wrist_ds = state_grp.create_dataset(
        "wrist", shape=(0, 18), chunks=(65536, 18), dtype=np.float32, overwrite=True, maxshape=(None, 18)
    )
    action_hand_ds = action_grp.create_dataset(
        "hand", shape=(0, 30), chunks=(65536, 30), dtype=np.float32, overwrite=True, maxshape=(None, 30)
    )
    action_wrist_ds = action_grp.create_dataset(
        "wrist", shape=(0, 18), chunks=(65536, 18), dtype=np.float32, overwrite=True, maxshape=(None, 18)
    )
    instruction_ds = data_grp.create_dataset(
        "instruction", shape=(0,), chunks=(65536,), dtype=f"U{INSTRUCTION_MAX_CHARS}", overwrite=True, maxshape=(None,)
    )
    camera_extrinsic_ds = data_grp.create_dataset(
        "extrinsic", shape=(0, 16), chunks=(65536, 16), dtype=np.float32, overwrite=True, maxshape=(None,)
    )
    camera_intrinsic_ds = data_grp.create_dataset(
        "intrinsic", shape=(0, 4), chunks=(65536, 4), dtype=np.float32, overwrite=True, maxshape=(None, 4)
    )
    image_ds = data_grp.create_dataset(
        "image", shape=(0, 384, 384, 3), chunks=(64, 384, 384, 3), dtype=np.uint8, overwrite=True, maxshape=(None, 384, 384, 3)
    )

    episode_ends: List[int] = []
    presence: List[int] = []  # 0 none, 1 left, 2 right, 3 both
    total_frames = 0
    total_episodes = 0

    mano_root = resolve_mano_root(data_root)
    pairs = collect_mano_pairs(mano_root)
    print(f"Found {len(pairs)} paired (or single-sided) episodes under: {mano_root}")

    for base_key, left_path, right_path in pairs:
        left = load_mano_npy(left_path) if left_path else None
        right = load_mano_npy(right_path) if right_path else None

        # Extract arrays
        l_pose15 = left.get("pose_coeff") if left is not None else None  # [Tl,15]
        r_pose15 = right.get("pose_coeff") if right is not None else None  # [Tr,15]
        l_aa3 = left.get("global_rot") if left is not None else None  # [Tl,3]
        r_aa3 = right.get("global_rot") if right is not None else None  # [Tr,3]
        l_t3 = left.get("trans") if left is not None else None  # [Tl,3]
        r_t3 = right.get("trans") if right is not None else None  # [Tr,3]

        Tl = l_pose15.shape[0] if isinstance(l_pose15, np.ndarray) and l_pose15.ndim == 2 else (l_t3.shape[0] if isinstance(l_t3, np.ndarray) and l_t3.ndim == 2 else 0)
        Tr = r_pose15.shape[0] if isinstance(r_pose15, np.ndarray) and r_pose15.ndim == 2 else (r_t3.shape[0] if isinstance(r_t3, np.ndarray) and r_t3.ndim == 2 else 0)
        T = max(Tl, Tr)
        if T == 0:
            print(f"  Skip {base_key}: empty left/right arrays")
            continue
        # Presence code: 0 none, 1 left only, 2 right only, 3 both
        presence_code = (1 if Tl > 0 else 0) + (2 if Tr > 0 else 0)

        l_pose15 = ensure_T(l_pose15, T, 15)
        r_pose15 = ensure_T(r_pose15, T, 15)
        l_aa3 = ensure_T(l_aa3, T, 3)
        r_aa3 = ensure_T(r_aa3, T, 3)
        l_t3 = ensure_T(l_t3, T, 3)
        r_t3 = ensure_T(r_t3, T, 3)

        # Forward-fill
        l_pose15 = forward_fill_rows(l_pose15)
        r_pose15 = forward_fill_rows(r_pose15)
        l_aa3 = forward_fill_rows(l_aa3)
        r_aa3 = forward_fill_rows(r_aa3)
        l_t3 = forward_fill_rows(l_t3)
        r_t3 = forward_fill_rows(r_t3)

        # Compute rot6 from axis-angle per frame
        l_rot6 = np.zeros((T, 6), dtype=np.float32)
        r_rot6 = np.zeros((T, 6), dtype=np.float32)
        for i in range(T):
            lR = axis_angle_to_rotmat(l_aa3[i])
            rR = axis_angle_to_rotmat(r_aa3[i])
            l_rot6[i] = rotmat_to_rot6(lR)
            r_rot6[i] = rotmat_to_rot6(rR)

        # Camera extrinsic per frame [T,16]
        any_side_path = left_path if left_path else right_path
        cam_extr_flat = load_episode_extrinsics(data_root, any_side_path, T)
        # Load camera intrinsics per frame [T,4] 
        cam_intr_flat = load_episode_intrinsics(data_root, any_side_path, T, img_size=384)
        # Load images aligned to T
        images = load_episode_images(data_root, any_side_path, T, img_size=384)

        # Build state/action, drop last frame so action is valid for each row
        state_hand = np.concatenate([l_pose15, r_pose15], axis=1)  # [T,30]
        state_wrist = np.concatenate([l_t3, r_t3, l_rot6, r_rot6], axis=1)  # [T,18]
        if T <= 1:
            continue
        eff_T = T - 1
        state_hand_eff = state_hand[:-1]
        state_wrist_eff = state_wrist[:-1]
        action_hand_eff = state_hand[1:]
        action_wrist_eff = state_wrist[1:]
        cam_extr_eff = cam_extr_flat[:-1]
        cam_intr_eff = cam_intr_flat[:-1]
        image_eff = images[:-1]

        # Append to long datasets under data/
        n = state_hand_ds.shape[0]
        state_hand_ds.resize((n + eff_T, 30))
        state_wrist_ds.resize((n + eff_T, 18))
        action_hand_ds.resize((n + eff_T, 30))
        action_wrist_ds.resize((n + eff_T, 18))
        instruction_ds.resize((n + eff_T,))
        camera_extrinsic_ds.resize((n + eff_T, 16))
        camera_intrinsic_ds.resize((n + eff_T, 4))
        image_ds.resize((n + eff_T, 384, 384, 3))

        state_hand_ds[n:n+eff_T] = state_hand_eff
        state_wrist_ds[n:n+eff_T] = state_wrist_eff
        action_hand_ds[n:n+eff_T] = action_hand_eff
        action_wrist_ds[n:n+eff_T] = action_wrist_eff
        # Derive instruction text from filename or ancestor folder (<action,tool,object>)
        instr_text = derive_instruction(left_path, right_path)
        instruction_ds[n:n+eff_T] = np.full((eff_T,), instr_text, dtype=f"U{INSTRUCTION_MAX_CHARS}")
        camera_extrinsic_ds[n:n+eff_T] = cam_extr_eff.astype(np.float32)
        camera_intrinsic_ds[n:n+eff_T] = cam_intr_eff.astype(np.float32)
        image_ds[n:n+eff_T] = image_eff

        # Cumulative 1-based end index (under meta/episode_ends)
        end_index = n + eff_T
        episode_ends.append(int(end_index))
        presence.append(int(presence_code))
        total_frames += int(eff_T)
        total_episodes += 1
        print(total_episodes)

    # Write meta
    meta_grp.create_dataset("episode_ends", data=np.array(episode_ends, dtype=np.int64), dtype=np.int64, overwrite=True)
    meta_grp.create_dataset("presence", data=np.array(presence, dtype=np.int8), dtype=np.int8, overwrite=True)
    root.attrs["n_episodes"] = int(total_episodes)
    root.attrs["n_frames"] = int(total_frames)

    print(f"Done. Episodes: {total_episodes}, Total frames: {total_frames}")
    print(episode_ends)


def main():
    parser = argparse.ArgumentParser(description="Build TACO Zarr dataset (concatenated long sequences from Mano_Poses npy)")
    parser.add_argument("--data_root", type=str, required=True, help="TACO root or Mano_Poses directory")
    parser.add_argument("--output", type=str, required=True, help="Output Zarr path (directory)")
    args = parser.parse_args()

    build_taco_zarr(args.data_root, args.output)


if __name__ == "__main__":
    main() 