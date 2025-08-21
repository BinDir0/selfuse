'''
This script is used to build the HOI4D dataset in zarr format.

Usage:
python build_hoi4d_zarr.py --data_root /path/to/hoi4d --output_zarr /path/to/output.zarr --mano_root /path/to/mano

Cause HOI4D's frequency is 15Hz, we need to upsample it to 30Hz. 
In this code, we upsampled the hand pose and the extrinsics.
The image is used only for history observation, so we don't need to upsample it.
'''
#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import zarr
import json
import csv
import cv2
import glob

# Optional accelerators
try:
    import decord
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False


INSTRUCTION_PLACEHOLDER = "finish the task"
INSTRUCTION_MAX_CHARS = 512
HOI4D_DEF_ROOT = "/home/guantianrui/HOI4D/HOI4D-Instructions/definitions"
TASK_DEF_CSV = os.path.join(HOI4D_DEF_ROOT, "task", "task_definitions.csv")
LABEL_CSV = os.path.join(HOI4D_DEF_ROOT, "motion segmentation", "label.csv")


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


# --- Quaternion helpers for SLERP ---

def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float32)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = aa / angle
    half = angle * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),       2 * (xz + wy)],
        [2 * (xy + wz),       1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),       2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=np.float32)
    return R


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    q1 = q1.astype(np.float32)
    q2 = q2.astype(np.float32)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if 1.0 - dot < 1e-6:
        return (1.0 - t) * q1 + t * q2
    theta = np.arccos(dot)
    s1 = np.sin((1.0 - t) * theta)
    s2 = np.sin(t * theta)
    s = np.sin(theta)
    return (s1 / s) * q1 + (s2 / s) * q2


# --- Resampling helpers: upsample by 2x to 30Hz from 15Hz ---

def upsample_linear(arr: np.ndarray) -> np.ndarray:
    """Given [T,D], return [2T-1,D] by inserting midpoints (0.5*(i,i+1))."""
    T, D = arr.shape
    if T <= 1:
        return arr.copy()
    out = np.zeros((2 * T - 1, D), dtype=np.float32)
    out[0::2] = arr
    mids = 0.5 * (arr[:-1] + arr[1:])
    out[1::2] = mids
    return out


def upsample_quat_halfstep(aa_arr: np.ndarray) -> np.ndarray:
    """Given [T,3] axis-angle per frame, return [2T-1,3x?] rotation matrices via SLERP at 0.5.
    Output is [2T-1,3,3] rotation matrices.
    """
    T = aa_arr.shape[0]
    if T <= 1:
        R0 = axis_angle_to_rotmat(aa_arr[0]) if T == 1 else np.eye(3, dtype=np.float32)
        return np.expand_dims(R0, axis=0)
    quats = np.stack([axis_angle_to_quat(aa) for aa in aa_arr], axis=0)
    out = np.zeros((2 * T - 1, 4), dtype=np.float32)
    out[0::2] = quats
    for i in range(T - 1):
        q_mid = quat_slerp(quats[i], quats[i + 1], 0.5)
        out[2 * i + 1] = q_mid
    R_list = [quat_to_rotmat(out[i]) for i in range(out.shape[0])]
    return np.stack(R_list, axis=0)


def upsample_extrinsic_forward_fill(flat_extr: np.ndarray) -> np.ndarray:
    """Given [T,16] extrinsics (world->cam) per frame, return [2T-1,16] by repeating previous for mid frames."""
    T = flat_extr.shape[0]
    if T <= 1:
        return flat_extr.copy()
    out = np.zeros((2 * T - 1, 16), dtype=np.float32)
    out[0::2] = flat_extr
    out[1::2] = flat_extr[:-1]
    return out


def load_hoi4d_images_15hz(
    data_root: str,
    rel_key: str,
    img_size: int = 384,
    backend: str = "auto"
) -> np.ndarray:
    """Decode HOI4D RGB video at 15Hz and resize to [T, img_size, img_size, 3] uint8.
    Source: HOI4D_release/<rel_key>/align_rgb/image.mp4
    """
    vid_path = os.path.join(data_root, 'HOI4D_release', rel_key, 'align_rgb', 'image.mp4')
    if not os.path.isfile(vid_path):
        return np.zeros((0, img_size, img_size, 3), dtype=np.uint8)
    # Decide backend
    use_decord = (backend in ("auto", "decord")) and _HAS_DECORD

    # Decode to an array of NHWC RGB frames
    frames_nhwc: Optional[np.ndarray] = None
    if use_decord:
        try:
            # Decode with decord (CPU context is typically robust)
            vr = decord.VideoReader(vid_path)
            Tv = len(vr)
            if Tv == 0:
                return np.zeros((0, img_size, img_size, 3), dtype=np.uint8)
            batch = vr.get_batch(list(range(Tv)))  # NHWC RGB on CPU
            frames_nhwc = batch.asnumpy()
        except Exception:
            frames_nhwc = None

    if frames_nhwc is None:
        # Fallback to OpenCV
        cap = cv2.VideoCapture(vid_path)
        frames_list: List[np.ndarray] = []
        if not cap.isOpened():
            return np.zeros((0, img_size, img_size, 3), dtype=np.uint8)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None:
                    continue
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = np.repeat(frame[:, :, :1], 3, axis=2)
                frames_list.append(frame)
        finally:
            cap.release()
        if len(frames_list) == 0:
            return np.zeros((0, img_size, img_size, 3), dtype=np.uint8)
        frames_nhwc = np.stack(frames_list, axis=0)

    print(frames_nhwc.shape)
    Tv = frames_nhwc.shape[0]
    return np.stack([cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_AREA) for f in frames_nhwc], axis=0)


# --- Instruction mapping ---

def load_task_map(task_csv_path: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    try:
        with open(task_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # Try forms: id,name or name,id or includes 'T' prefix
                nums = [int(tok.replace('T', '').strip()) for tok in row if tok.strip().lstrip('T').isdigit()]
                names = [tok.strip() for tok in row if not tok.strip().lstrip('T').isdigit() and tok.strip()]
                if nums and names:
                    tid = nums[0]
                    name = names[0]
                    if tid not in mapping:
                        mapping[tid] = name
    except Exception:
        pass
    return mapping


def load_category_map(label_csv_path: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    try:
        with open(label_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # Heuristic: first int => id, next non-empty string => name
                id_candidates = [int(tok.strip()) for tok in row if tok.strip().isdigit()]
                if not id_candidates:
                    continue
                cid = id_candidates[0]
                # choose first non-numeric token
                name_tokens = [tok.strip() for tok in row if not tok.strip().isdigit() and tok.strip()]
                if name_tokens:
                    mapping[cid] = name_tokens[0]
    except Exception:
        pass
    # Fallback from README mapping if empty
    if not mapping:
        fallback = [
            '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
            'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
            'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
        ]
        for idx, name in enumerate(fallback):
            if name:
                mapping[idx] = name
    return mapping

# --- Action (event) based instruction generation from annotations ---
def load_action_segments(data_root: str, rel_key: str) -> Optional[List[Dict[str, object]]]:
    """Load action segments from HOI4D_annotations/<rel_key>/action/color.json.
    Expected fields per item: event, starttime/startTime, endtime/endTime (seconds or frame indices).
    """
    tail = os.path.join('action', 'color.json')
    action_path = _resolve_hoi4d_path_with_wildcard(os.path.join(data_root, 'HOI4D_annotations'), rel_key, tail)
    try:
        if not action_path:
            return None
        with open(action_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        # color.json may be a list or a dict containing a list under a key
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            # try common keys
            for k in ['events', 'actions', 'segments', 'data']:
                if k in obj and isinstance(obj[k], list):
                    items = obj[k]
                    break
            else:
                # treat dict itself as one item
                items = [obj]
        else:
            return None
        segs: List[Dict[str, object]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            ev = it.get('event', None)
            st = it.get('starttime', it.get('startTime', None))
            en = it.get('endtime', it.get('endTime', None))
            if ev is None or st is None or en is None:
                continue
            try:
                stf = float(st)
                enf = float(en)
            except Exception:
                continue
            segs.append({'event': str(ev), 'start': stf, 'end': enf})
        return segs if segs else None
    except Exception:
        return None


def build_instructions_for_episode(data_root: str, rel_key: str, T_frames: int, cat_map: Dict[int, str], task_map: Dict[int, str]) -> Optional[np.ndarray]:
    """Return per-frame instruction array of length T_frames using action segments if available.
    Format: "<event> the <object>." If no segments, return None.
    """
    segs = load_action_segments(data_root, rel_key)
    if not segs:
        return None
    # Object name from category id in rel_key if possible
    C_id, T_id = parse_ids_from_rel_key(rel_key)
    obj_name = cat_map.get(C_id, f"object_{C_id}") if C_id is not None else "object"
    # Determine if times are in seconds (likely <= 120) or frame indices
    max_end = max(float(s['end']) for s in segs)
    assume_seconds = max_end <= 120.0
    fps = 30.0
    instr = np.array([f"finish the task" for _ in range(T_frames)], dtype=f"U{INSTRUCTION_MAX_CHARS}")
    for s in segs:
        ev = str(s['event']).strip()
        s_raw = float(s['start'])
        e_raw = float(s['end'])
        if assume_seconds:
            s_idx = int(round(s_raw * fps))
            e_idx = int(round(e_raw * fps))
        else:
            s_idx = int(round(s_raw))
            e_idx = int(round(e_raw))
        s_idx = max(0, min(T_frames - 1, s_idx))
        e_idx = max(0, min(T_frames - 1, e_idx))
        if e_idx < s_idx:
            s_idx, e_idx = e_idx, s_idx
        text = f"{ev} the {obj_name}."
        instr[s_idx:e_idx + 1] = text
    return instr


def parse_ids_from_rel_key(rel_key: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract (C,T) integer ids from rel_key like ZY.../H1/C7/N.../S.../s.../T4"""
    C_id = None
    T_id = None
    parts = rel_key.replace('\\', '/').split('/')
    for p in parts:
        if p.startswith('C') and p[1:].isdigit():
            C_id = int(p[1:])
        if p.startswith('T') and p[1:].isdigit():
            T_id = int(p[1:])
    return C_id, T_id


def make_instruction_hoi4d(rel_key: str, task_map: Dict[int, str], cat_map: Dict[int, str]) -> str:
    C_id, T_id = parse_ids_from_rel_key(rel_key)
    obj = cat_map.get(C_id, f"object_{C_id}") if C_id is not None else "object"
    task = task_map.get(T_id, f"task_{T_id}") if T_id is not None else "operate"
    sent = f"{task} the {obj}."
    return sent[:INSTRUCTION_MAX_CHARS]


def collect_mano_episode_dirs(side_root: str) -> Dict[str, str]:
    """Collect all leaf T* directories that contain *_mano.npy and map to relative keys.

    Returns {rel_key: abs_path_to_T_dir}.
    """
    mapping: Dict[str, str] = {}
    for cur_root, dirs, files in os.walk(side_root):
        npy_files = [f for f in files if f.endswith('_mano.npy')]
        if len(npy_files) > 0:
            rel = os.path.relpath(cur_root, side_root)
            mapping[rel] = cur_root
    return mapping


def parse_frame_id_from_fname(fname: str) -> int:
    base = os.path.splitext(fname)[0]
    # Expect formats like: 123_mano
    parts = base.split('_')
    try:
        return int(parts[0])
    except Exception:
        return -1


def load_episode_from_mano_npys(ep_dir: str) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """Load per-frame *_mano.npy dicts for one side.

    Returns (frame_ids, pose15[T,15], aa3[T,3], trans[T,3]) sorted by frame id.
    Missing or malformed frames are skipped.
    """
    files = [f for f in os.listdir(ep_dir) if f.endswith('_mano.npy')]
    if not files:
        return [], np.zeros((0, 15), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    # Sort by parsed frame id
    files.sort(key=parse_frame_id_from_fname)

    frame_ids: List[int] = []
    pose_list: List[np.ndarray] = []
    aa3_list: List[np.ndarray] = []
    t3_list: List[np.ndarray] = []

    for f in files:
        fid = parse_frame_id_from_fname(f)
        if fid < 0:
            continue
        path = os.path.join(ep_dir, f)
        try:
            data = np.load(path, allow_pickle=True).item()
        except Exception as e:
            print(f"[WARN] failed to load {path}: {e}")
            continue
        if not all(k in data for k in ("pose_coeff", "global_rot", "trans")):
            print(f"[WARN] missing keys in {path}")
            continue
        pose = np.asarray(data["pose_coeff"], dtype=np.float32).reshape(-1)
        aa3 = np.asarray(data["global_rot"], dtype=np.float32).reshape(-1)
        t3 = np.asarray(data["trans"], dtype=np.float32).reshape(-1)
        if pose.shape[0] != 15 or aa3.shape[0] != 3 or t3.shape[0] != 3:
            print(f"[WARN] wrong dims in {path}: pose15={pose.shape}, aa3={aa3.shape}, t3={t3.shape}")
            continue
        frame_ids.append(fid)
        pose_list.append(pose)
        aa3_list.append(aa3)
        t3_list.append(t3)

    if len(frame_ids) == 0:
        return [], np.zeros((0, 15), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    # Already sorted by fid
    pose15 = np.stack(pose_list, axis=0)
    aa3 = np.stack(aa3_list, axis=0)
    t3 = np.stack(t3_list, axis=0)
    return frame_ids, pose15, aa3, t3


def _resolve_hoi4d_path_with_wildcard(base_root: str, rel_key: str, tail: str) -> str:
    """Resolve file path even if some segments (H/C/N/S/s/T) differ from rel_key.
    Strategy:
      - Parse tokens H,C,N,S,s,T from rel_key
      - If T exists, search '**/T{T}/{tail}', else search '**/{tail}' (may be slower)
      - Filter candidates by requiring other tokens present as '/Xval/' substrings
    Returns absolute file path or empty string if not found.
    """
    # 1) Exact path first
    exact = os.path.join(base_root, rel_key, tail)
    if os.path.isfile(exact):
        return exact
    # 2) Parse tokens
    parts = rel_key.replace('\\', '/').split('/')
    tok: Dict[str, Optional[str]] = {k: None for k in ['H', 'C', 'N', 'S', 's', 'T']}
    for p in parts:
        for k in list(tok.keys()):
            if p.startswith(k) and len(p) > 1 and p[1:].isdigit():
                tok[k] = p[1:]
    # 3) Build glob pattern (prefer T)
    if tok['T']:
        pattern = os.path.join(base_root, '**', f"T{tok['T']}", tail)
    else:
        pattern = os.path.join(base_root, '**', tail)
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return ''
    # 4) Filter by other tokens
    def ok(path: str) -> bool:
        sp = path.replace('\\', '/')
        for k in ['H', 'C', 'N', 'S', 's']:
            v = tok[k]
            if v is None:
                continue
            if f"/{k}{v}/" not in sp:
                return False
        return True
    filtered = [p for p in candidates if ok(p)]
    hits = filtered if filtered else candidates
    hits.sort(key=lambda x: len(x))
    return hits[0] if hits else ''


def collect_annotation_t_ids(ann_root: str) -> set:
    """Collect all T ids present anywhere under HOI4D_annotations for fast intersection filtering."""
    t_ids: set = set()
    for cur_root, dirs, files in os.walk(ann_root):
        base = os.path.basename(cur_root)
        if len(base) > 1 and base[0] == 'T' and base[1:].isdigit():
            try:
                t_ids.add(int(base[1:]))
            except Exception:
                continue
    return t_ids


def collect_annotation_rel_keys(ann_root: str) -> set:
    """Collect all rel_keys (path relative to ann_root) that end at a T* directory."""
    rel_keys: set = set()
    for cur_root, dirs, files in os.walk(ann_root):
        base = os.path.basename(cur_root)
        if len(base) > 1 and base[0] == 'T' and base[1:].isdigit():
            rel = os.path.relpath(cur_root, ann_root).replace('\\', '/')
            rel_keys.add(rel)
    return rel_keys


def collect_annotation_episode_dirs(ann_root: str) -> Dict[str, str]:
    """Collect all leaf T* directories under HOI4D_annotations and map to relative keys.

    Returns {rel_key: abs_path_to_T_dir}.
    """
    mapping: Dict[str, str] = {}
    for cur_root, dirs, files in os.walk(ann_root):
        base = os.path.basename(cur_root)
        if len(base) > 1 and base[0] == 'T' and base[1:].isdigit():
            rel = os.path.relpath(cur_root, ann_root).replace('\\', '/')
            mapping[rel] = cur_root
    return mapping


def episode_has_annotations(data_root: str, rel_key: str) -> bool:
    """Return True if this episode (identified by rel_key) exists under HOI4D_annotations.

    We check for presence of at least one expected file (action/color.json or 3Dseg/output.log)
    using the wildcard resolver to accommodate slight path differences.
    """
    tails = [
        os.path.join('action', 'color.json'),
        os.path.join('3Dseg', 'output.log'),
    ]
    for tail in tails:
        path = _resolve_hoi4d_path_with_wildcard(os.path.join(data_root, 'HOI4D_annotations'), rel_key, tail)
        if path:
            return True
    return False

def parse_open3d_output_log(output_log_path: str) -> Optional[np.ndarray]:
    """Parse Open3D camera trajectory output.log into [Tc,4,4] float32 if possible.
    Supports two formats:
      1) JSON-like dict with 'parameters' -> list of { 'extrinsic': [[4x4], ...] }
      2) Text format repeating blocks: a header line with three integers, then 4 lines of 4 floats (last row often 0 0 0 1)
    Returns None on failure.
    """
    try:
        with open(output_log_path, 'r', encoding='utf-8') as f:
            txt = f.read()
        # Try JSON-like first
        try:
            obj = json.loads(txt)
            params = obj.get('parameters', None)
            if isinstance(params, list) and len(params) > 0:
                mats: List[np.ndarray] = []
                for p in params:
                    extr = p.get('extrinsic', None) if isinstance(p, dict) else None
                    if extr is None:
                        continue
                    M = np.array(extr, dtype=np.float32)
                    if M.shape == (4, 4):
                        mats.append(M)
                if len(mats) > 0:
                    return np.stack(mats, axis=0)
        except Exception:
            pass
        # Fallback: text-based format
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        edges: List[Tuple[int, int, np.ndarray]] = []
        i = 0
        n = len(lines)
        while i < n:
            # Expect a header line: three integers
            hdr = lines[i].split()
            is_hdr = False
            if len(hdr) >= 3:
                try:
                    a = int(hdr[0]); b = int(hdr[1]); c = int(hdr[2])
                    is_hdr = True
                except Exception:
                    is_hdr = False
            if not is_hdr:
                i += 1
                continue
            # Ensure 4 following lines for a 4x4 matrix
            if i + 4 >= n:
                break
            rows: List[List[float]] = []
            ok = True
            for j in range(1, 5):
                try:
                    vals = [float(x) for x in lines[i + j].split()]
                    if len(vals) != 4:
                        ok = False
                        break
                    rows.append(vals)
                except Exception:
                    ok = False
                    break
            if ok:
                M = np.array(rows, dtype=np.float32)
                if M.shape == (4, 4):
                    edges.append((a, c, M))
                i += 5
            else:
                # If parsing failed, advance by one line to resynchronize
                i += 1
        if len(edges) > 0:
            # Accumulate relative transforms into absolute camera-to-world poses.
            max_idx = max(max(a, c) for a, c, _ in edges) + 1
            poses: List[np.ndarray] = [np.eye(4, dtype=np.float32) for _ in range(max_idx)]
            have: List[bool] = [False] * max_idx
            poses[0] = np.eye(4, dtype=np.float32)
            have[0] = True
            edges.sort(key=lambda e: e[0])
            for a, c, M in edges:
                if 0 <= a < max_idx and 0 <= c < max_idx and c == a + 1 and have[a]:
                    # M is relative transform from frame a to frame c: accumulate as P_c = P_a @ M
                    poses[c] = (poses[a].astype(np.float64) @ M.astype(np.float64)).astype(np.float32)
                    have[c] = True
            # Forward fill any missing poses
            for k in range(1, max_idx):
                if not have[k]:
                    poses[k] = poses[k - 1]
                    have[k] = True
            # Convert camera-to-world poses to world-to-camera extrinsics
            extrinsics: List[np.ndarray] = []
            for k in range(max_idx):
                try:
                    E = np.linalg.inv(poses[k].astype(np.float64)).astype(np.float32)
                except Exception:
                    E = np.linalg.pinv(poses[k].astype(np.float64)).astype(np.float32)
                extrinsics.append(E)
            return np.stack(extrinsics, axis=0)
        return None
    except Exception:
        return None


def load_hoi4d_episode_extrinsics(data_root: str, rel_key: str, T: int) -> np.ndarray:
    """Best-effort load camera extrinsics for HOI4D episode identified by rel_key.
    Tries HOI4D_annotations/<rel_key>/3Dseg/output.log first, then HOI4D_release/<rel_key>/3Dseg/output.log.
    If found and parsed, align length to T, else return zeros. Output shape [T,16].
    """
    tail = os.path.join('3Dseg', 'output.log')
    # Prefer annotations
    output_log = _resolve_hoi4d_path_with_wildcard(os.path.join(data_root, 'HOI4D_annotations'), rel_key, tail)
    if not output_log:
        return np.zeros((T, 16), dtype=np.float32)
    mats = parse_open3d_output_log(output_log)
    if mats is None or mats.ndim != 3 or mats.shape[1:] != (4, 4):
        return np.zeros((T, 16), dtype=np.float32)
    flat = mats.reshape(mats.shape[0], 16).astype(np.float32)
    if flat.shape[0] < T:
        pad = np.zeros((T - flat.shape[0], 16), dtype=np.float32)
        flat = np.concatenate([flat, pad], axis=0)
    elif flat.shape[0] > T:
        flat = flat[:T]
    flat = forward_fill_rows(flat)
    return flat 

def build_hoi4d_zarr(data_root: str, output_zarr: str, mano_root: str, image_backend: str = "auto"):
    """Build HOI4D Zarr from pre-converted *_mano.npy (15D PCA + aa3 + t3).

    - data/state/hand: [S,30]
    - data/state/wrist: [S,18]
    - data/action/hand: [S,30]
    - data/action/wrist: [S,18]
    - data/instruction: [S] (placeholder)
    - meta/episode_ends: [E] 1-based cumulative ends
    """
    right_root = os.path.join(data_root, 'mano_hand_pose', 'right_hand')
    left_root = os.path.join(data_root, 'mano_hand_pose', 'left_hand')

    if not os.path.isdir(right_root) and not os.path.isdir(left_root):
        raise FileNotFoundError(f"Neither right nor left hand roots found under {os.path.join(data_root, 'mano_hand_pose')}")

    # Zarr root
    os.makedirs(os.path.dirname(output_zarr), exist_ok=True)
    store = zarr.DirectoryStore(output_zarr)
    root = zarr.group(store=store, overwrite=True)

    data_grp = root.create_group('data')
    meta_grp = root.create_group('meta')

    state_grp = data_grp.create_group('state')
    action_grp = data_grp.create_group('action')

    state_hand_ds = state_grp.create_dataset('hand', shape=(0, 30), chunks=(65536, 30), dtype=np.float32, overwrite=True, maxshape=(None, 30))
    state_wrist_ds = state_grp.create_dataset('wrist', shape=(0, 18), chunks=(65536, 18), dtype=np.float32, overwrite=True, maxshape=(None, 18))
    action_hand_ds = action_grp.create_dataset('hand', shape=(0, 30), chunks=(65536, 30), dtype=np.float32, overwrite=True, maxshape=(None, 30))
    action_wrist_ds = action_grp.create_dataset('wrist', shape=(0, 18), chunks=(65536, 18), dtype=np.float32, overwrite=True, maxshape=(None, 18))
    instruction_ds = data_grp.create_dataset('instruction', shape=(0,), chunks=(65536,), dtype=f"U{INSTRUCTION_MAX_CHARS}", overwrite=True, maxshape=(None,))
    camera_extrinsic_ds = data_grp.create_dataset('extrinsic', shape=(0, 16), chunks=(65536, 16), dtype=np.float32, overwrite=True, maxshape=(None, 16))
    image_ds = data_grp.create_dataset('image', shape=(0, 384, 384, 3), chunks=(64, 384, 384, 3), dtype=np.uint8, overwrite=True, maxshape=(None, 384, 384, 3))

    # Collect episode directories and pair by relative key (relative to side root)
    right_eps = collect_mano_episode_dirs(right_root) if os.path.isdir(right_root) else {}
    left_eps = collect_mano_episode_dirs(left_root) if os.path.isdir(left_root) else {}
    ann_root = os.path.join(data_root, 'HOI4D_annotations')
    episode_eps = collect_annotation_episode_dirs(ann_root) if os.path.isdir(ann_root) else {}
    # (left | right) & episode, all paths guaranteed to exist because they are discovered via os.walk
    left_right_keys = set(right_eps.keys()) | set(left_eps.keys())
    keys = sorted(left_right_keys & set(episode_eps.keys()))
    print(f"Collected: left={len(left_eps)}, right={len(right_eps)}, episode={len(episode_eps)}; intersection={(len(keys))}")

    episode_ends: List[int] = []
    presence_codes: List[int] = []  # 0 none, 1 left, 2 right, 3 both
    total_frames = 0
    total_episodes = 0

    print(f"Processing {len(keys)} episodes after intersection")

    # Load instruction maps once
    task_map = load_task_map(TASK_DEF_CSV)
    cat_map = load_category_map(LABEL_CSV)

    for rel_key in keys:
        print(f"Processing episode: {rel_key}")
        right_dir = right_eps.get(rel_key)
        left_dir = left_eps.get(rel_key)

        # Load per-side frames from *_mano.npy
        r_ids, r_pose15_all, r_aa3_all, r_t3_all = ([], np.zeros((0,15), np.float32), np.zeros((0,3), np.float32), np.zeros((0,3), np.float32))
        l_ids, l_pose15_all, l_aa3_all, l_t3_all = ([], np.zeros((0,15), np.float32), np.zeros((0,3), np.float32), np.zeros((0,3), np.float32))
        if right_dir:
            r_ids, r_pose15_all, r_aa3_all, r_t3_all = load_episode_from_mano_npys(right_dir)
        if left_dir:
            l_ids, l_pose15_all, l_aa3_all, l_t3_all = load_episode_from_mano_npys(left_dir)

        frame_ids = sorted(set(r_ids) | set(l_ids))
        T = len(frame_ids)
        if T == 0:
            print(f"  Skip empty episode: {rel_key}")
            continue
        presence_code = (1 if len(l_ids) > 0 else 0) + (2 if len(r_ids) > 0 else 0)

        # Allocate aligned arrays (15Hz)
        l_pose15 = np.zeros((T, 15), dtype=np.float32)
        r_pose15 = np.zeros((T, 15), dtype=np.float32)
        l_t3_cam = np.zeros((T, 3), dtype=np.float32)
        r_t3_cam = np.zeros((T, 3), dtype=np.float32)
        l_R_cam = np.zeros((T, 3, 3), dtype=np.float32)
        r_R_cam = np.zeros((T, 3, 3), dtype=np.float32)

        r_idx = {fid: i for i, fid in enumerate(r_ids)}
        l_idx = {fid: i for i, fid in enumerate(l_ids)}

        # Fill aligned arrays per frame
        for ti, fid in enumerate(frame_ids):
            if fid in r_idx:
                j = r_idx[fid]
                r_pose15[ti] = r_pose15_all[j]
                r_t3_cam[ti] = r_t3_all[j]
                r_R_cam[ti] = axis_angle_to_rotmat(r_aa3_all[j])
            if fid in l_idx:
                j = l_idx[fid]
                l_pose15[ti] = l_pose15_all[j]
                l_t3_cam[ti] = l_t3_all[j]
                l_R_cam[ti] = axis_angle_to_rotmat(l_aa3_all[j])

        # Forward-fill to handle missing frames
        l_pose15 = forward_fill_rows(l_pose15)
        r_pose15 = forward_fill_rows(r_pose15)
        l_t3_cam = forward_fill_rows(l_t3_cam)
        r_t3_cam = forward_fill_rows(r_t3_cam)
        # For rotations, forward-fill by copying previous matrix on zero rows
        for i in range(1, T):
            if not l_R_cam[i].any():
                l_R_cam[i] = l_R_cam[i - 1]
            if not r_R_cam[i].any():
                r_R_cam[i] = r_R_cam[i - 1]

        # Load camera extrinsics from HOI4D_release if available (world->camera per frame at 15Hz)
        cam_extr_flat_15 = load_hoi4d_episode_extrinsics(data_root, rel_key, T)  # [T,16]
        # Load images at 15Hz
        imgs_15 = load_hoi4d_images_15hz(data_root, rel_key, img_size=384, backend=image_backend)  # [Timg,384,384,3]

        # Upsample to 30Hz
        l_pose15_30 = upsample_linear(l_pose15)
        r_pose15_30 = upsample_linear(r_pose15)
        # For rotations, we need original axis-angles; reconstruct from matrices is non-trivial; use SLERP from axis-angles directly
        # Recompute quats from cam-rot matrices: not needed, use original aa arrays if available; fallback using matrices -> approximate
        # Build aa arrays from original inputs for SLERP
        # Note: r_aa3_all and l_aa3_all aligned to r_ids/l_ids; we already converted to matrices above per frame IDS
        # To ensure consistency, derive aa arrays aligned to frame_ids from R_cam via log map is heavy; instead, reuse per-frame R_cam and do simple halfway slerp via quats reconstructed
        def mats_to_quats(R_seq: np.ndarray) -> np.ndarray:
            Tn = R_seq.shape[0]
            quats = np.zeros((Tn, 4), dtype=np.float32)
            for k in range(Tn):
                Rk = R_seq[k]
                tr = np.trace(Rk)
                if tr > 0.0:
                    S = np.sqrt(tr + 1.0) * 2.0
                    w = 0.25 * S
                    x = (Rk[2, 1] - Rk[1, 2]) / S
                    y = (Rk[0, 2] - Rk[2, 0]) / S
                    z = (Rk[1, 0] - Rk[0, 1]) / S
                else:
                    if Rk[0, 0] > Rk[1, 1] and Rk[0, 0] > Rk[2, 2]:
                        S = np.sqrt(1.0 + Rk[0, 0] - Rk[1, 1] - Rk[2, 2]) * 2.0
                        w = (Rk[2, 1] - Rk[1, 2]) / S
                        x = 0.25 * S
                        y = (Rk[0, 1] + Rk[1, 0]) / S
                        z = (Rk[0, 2] + Rk[2, 0]) / S
                    elif Rk[1, 1] > Rk[2, 2]:
                        S = np.sqrt(1.0 + Rk[1, 1] - Rk[0, 0] - Rk[2, 2]) * 2.0
                        w = (Rk[0, 2] - Rk[2, 0]) / S
                        x = (Rk[0, 1] + Rk[1, 0]) / S
                        y = 0.25 * S
                        z = (Rk[1, 2] + Rk[2, 1]) / S
                    else:
                        S = np.sqrt(1.0 + Rk[2, 2] - Rk[0, 0] - Rk[1, 1]) * 2.0
                        w = (Rk[1, 0] - Rk[0, 1]) / S
                        x = (Rk[0, 2] + Rk[2, 0]) / S
                        y = (Rk[1, 2] + Rk[2, 1]) / S
                        z = 0.25 * S
                quats[k] = np.array([w, x, y, z], dtype=np.float32)
            # Normalize
            norms = np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8
            quats = quats / norms
            return quats
        l_quat = mats_to_quats(l_R_cam)
        r_quat = mats_to_quats(r_R_cam)
        def upsample_quat_half(quats: np.ndarray) -> np.ndarray:
            Tn = quats.shape[0]
            if Tn <= 1:
                return quats.copy()
            out = np.zeros((2 * Tn - 1, 4), dtype=np.float32)
            out[0::2] = quats
            for i in range(Tn - 1):
                out[2 * i + 1] = quat_slerp(quats[i], quats[i + 1], 0.5)
            return out
        l_quat_30 = upsample_quat_half(l_quat)
        r_quat_30 = upsample_quat_half(r_quat)
        l_R_cam_30 = np.stack([quat_to_rotmat(q) for q in l_quat_30], axis=0)
        r_R_cam_30 = np.stack([quat_to_rotmat(q) for q in r_quat_30], axis=0)
        l_t3_cam_30 = upsample_linear(l_t3_cam)
        r_t3_cam_30 = upsample_linear(r_t3_cam)
        cam_extr_30 = upsample_extrinsic_forward_fill(cam_extr_flat_15)
        # print(cam_extr_flat_15)
        # Upsample images by forward-fill: insert previous frame in-between
        if imgs_15.shape[0] > 0:
            Timg = imgs_15.shape[0]
            imgs_30 = np.zeros((2 * Timg - 1, 384, 384, 3), dtype=np.uint8)
            imgs_30[0::2] = imgs_15
            imgs_30[1::2] = imgs_15[:-1]
            # Align to T30 if mismatch (pad/truncate)
            if imgs_30.shape[0] < cam_extr_30.shape[0]:
                pad = np.zeros((cam_extr_30.shape[0] - imgs_30.shape[0], 384, 384, 3), dtype=np.uint8)
                imgs_30 = np.concatenate([imgs_30, pad], axis=0)
            elif imgs_30.shape[0] > cam_extr_30.shape[0]:
                imgs_30 = imgs_30[:cam_extr_30.shape[0]]
        else:
            imgs_30 = np.zeros((cam_extr_30.shape[0], 384, 384, 3), dtype=np.uint8)

        # Transform wrist to world frame per frame using camera extrinsic (world->camera): x_w = R_cw * x_c + t_cw; R_w = R_cw * R_c
        def inv_extrinsic(flat16: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            M = flat16.reshape(4, 4)
            Rwc = M[:3, :3]
            twc = M[:3, 3]
            Rcw = Rwc.T
            tcw = -Rcw @ twc
            return Rcw.astype(np.float32), tcw.astype(np.float32)
        T30 = l_pose15_30.shape[0]
        l_t3_world_30 = np.zeros((T30, 3), dtype=np.float32)
        r_t3_world_30 = np.zeros((T30, 3), dtype=np.float32)
        l_R_world_30 = np.zeros((T30, 3, 3), dtype=np.float32)
        r_R_world_30 = np.zeros((T30, 3, 3), dtype=np.float32)
        for i in range(T30):
            Rcw, tcw = inv_extrinsic(cam_extr_30[i])
            l_t3_world_30[i] = Rcw @ l_t3_cam_30[i] + tcw
            r_t3_world_30[i] = Rcw @ r_t3_cam_30[i] + tcw
            l_R_world_30[i] = Rcw @ l_R_cam_30[i]
            r_R_world_30[i] = Rcw @ r_R_cam_30[i]

        # Convert rotations to 6D
        # print(f"Rcw: {Rcw}", f"tcw: {tcw}")
        l_rot6_world_30 = np.stack([rotmat_to_rot6(R) for R in l_R_world_30], axis=0)
        r_rot6_world_30 = np.stack([rotmat_to_rot6(R) for R in r_R_world_30], axis=0)
        # print(f"r_rot6_world_30: {r_rot6_world_30[1,:]}")

        # Final state/action at 30Hz
        state_hand = np.concatenate([l_pose15_30, r_pose15_30], axis=1)  # [T30,30]
        state_wrist = np.concatenate([l_t3_world_30, r_t3_world_30, l_rot6_world_30, r_rot6_world_30], axis=1)  # [T30,18]

        if T30 <= 1:
            continue
        eff_T = T30 - 1
        state_hand_eff = state_hand[:-1]
        state_wrist_eff = state_wrist[:-1]
        action_hand_eff = state_hand[1:]
        action_wrist_eff = state_wrist[1:]
        cam_extr_eff = cam_extr_30[:-1]
        image_eff = imgs_30[:-1]

        # Append to datasets
        n = state_hand_ds.shape[0]
        state_hand_ds.resize((n + eff_T, 30))
        state_wrist_ds.resize((n + eff_T, 18))
        action_hand_ds.resize((n + eff_T, 30))
        action_wrist_ds.resize((n + eff_T, 18))
        instruction_ds.resize((n + eff_T,))
        camera_extrinsic_ds.resize((n + eff_T, 16))
        image_ds.resize((n + eff_T, 384, 384, 3))

        state_hand_ds[n:n+eff_T] = state_hand_eff
        state_wrist_ds[n:n+eff_T] = state_wrist_eff
        action_hand_ds[n:n+eff_T] = action_hand_eff
        action_wrist_ds[n:n+eff_T] = action_wrist_eff
        # Instruction per frame using action segments if available
        try:
            instr_full = build_instructions_for_episode(data_root, rel_key, cam_extr_30.shape[0], cat_map, task_map)
        except Exception:
            instr_full = None
        if instr_full is not None and instr_full.shape[0] >= eff_T:
            instruction_ds[n:n+eff_T] = instr_full[:-1]
        else:
            instr_text = make_instruction_hoi4d(rel_key, task_map, cat_map)
            instruction_ds[n:n+eff_T] = np.full((eff_T,), instr_text, dtype=f"U{INSTRUCTION_MAX_CHARS}")
        camera_extrinsic_ds[n:n+eff_T] = cam_extr_eff.astype(np.float32)
        image_ds[n:n+eff_T] = image_eff
        print(f"instruction_ds: {instruction_ds[n+2]}")

        end_index = n + eff_T
        episode_ends.append(int(end_index))
        presence_codes.append(int(presence_code))
        total_frames += int(eff_T)
        print(f"presence_code: {presence_code}, T: {eff_T}")
        total_episodes += 1

        if (total_episodes % 200) == 0:
            print(f"Processed {total_episodes} episodes, frames so far: {total_frames}")

    # Meta
    meta_grp.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64), dtype=np.int64, overwrite=True)
    meta_grp.create_dataset('presence', data=np.array(presence_codes, dtype=np.int8), dtype=np.int8, overwrite=True)
    root.attrs['n_episodes'] = int(total_episodes)
    root.attrs['n_frames'] = int(total_frames)

    print(f"Done. Episodes: {total_episodes}, Total frames: {total_frames}")


def main():
    parser = argparse.ArgumentParser(description='Build HOI4D Zarr dataset from pre-converted *_mano.npy (unsampled, concatenated)')
    parser.add_argument('--data_root', type=str, default='/share_data/datasets/hoi4d', help='HOI4D root directory')
    parser.add_argument('--output', type=str, default='/share_data/datasets/hoi4d/hoi4d.zarr', help='Output Zarr path (directory)')
    args = parser.parse_args()

    build_hoi4d_zarr(args.data_root, args.output, mano_root='', image_backend='auto')


if __name__ == '__main__':
    main()

