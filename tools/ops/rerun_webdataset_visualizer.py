#!/usr/bin/env python3
"""Rerun viewer for inspecting WebDataset shards as a single episode timeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.pipeline.viewer_backend import EpisodeViewerBackend, scan_sample_summaries, select_episode_key
from tools.ops import webdataset_visualizer as wv

WORLD_ROOT = "/world"
CAMERA_ROOT = "/world/camera"
PINHOLE_ROOT = "/world/camera/pinhole"
IMAGE_ROOT = "/world/camera/pinhole/image"
LEFT_WORLD_ROOT = "/world/hands/left"
RIGHT_WORLD_ROOT = "/world/hands/right"
LEFT_IMAGE_ROOT = "/world/camera/pinhole/image/hands/left"
RIGHT_IMAGE_ROOT = "/world/camera/pinhole/image/hands/right"

# Match demo_offline.py hand overlay colors as closely as possible.
# demo_offline uses left=(200,100,128) and right=(202,152,53) on RGB frames.
LEFT_COLOR = np.array([200, 100, 128], dtype=np.uint8)
LEFT_LINE_COLOR = np.array([176, 82, 110], dtype=np.uint8)
RIGHT_COLOR = np.array([202, 152, 53], dtype=np.uint8)
RIGHT_LINE_COLOR = np.array([176, 128, 42], dtype=np.uint8)
MANO_FACE_EXTRA = np.array(
    [
        [92, 38, 234],
        [234, 38, 239],
        [38, 122, 239],
        [239, 122, 279],
        [122, 118, 279],
        [279, 118, 215],
        [118, 117, 215],
        [215, 117, 214],
        [117, 119, 214],
        [214, 119, 121],
        [119, 120, 121],
        [121, 120, 78],
        [120, 108, 78],
        [78, 108, 79],
    ],
    dtype=np.int32,
)


def load_rerun():
    try:
        import rerun as rr
    except ImportError as error:
        raise SystemExit(
            "rerun-sdk is not installed. Install viewer dependencies with:\n"
            "  pip install -r requirements-rerun-viewer.txt"
        ) from error
    return rr


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "episode"


def resolve_rrd_output_path(rrd_out: str | None, *, episode_key: str, render_mode: str) -> Path:
    if rrd_out:
        path = Path(rrd_out).expanduser()
    else:
        path = Path.cwd() / f"{_slugify(episode_key)}_{render_mode}.rrd"
    if path.suffix.lower() != ".rrd":
        path = path.with_suffix(".rrd")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def configure_recording(rr, *, output_mode: str, rrd_out: str | None, episode_key: str, render_mode: str) -> Path | None:
    rr.init(
        "hawor_webdataset_viewer",
        spawn=output_mode in {"online", "both"},
    )

    if output_mode not in {"offline", "both"}:
        return None

    rrd_path = resolve_rrd_output_path(
        rrd_out,
        episode_key=episode_key,
        render_mode=render_mode,
    )
    rr.save(rrd_path)
    return rrd_path


def send_default_blueprint(rr):
    try:
        import rerun.blueprint as rrb

        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=IMAGE_ROOT, name="Camera"),
                    rrb.Spatial3DView(origin=WORLD_ROOT, name="World"),
                )
            ),
            make_active=True,
        )
    except Exception as error:  # pragma: no cover - best effort for SDK drift
        print(f"Warning: failed to activate default Rerun blueprint: {error}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Rerun viewer for WebDataset shard samples")
    parser.add_argument("--input", required=True, help="Path to a .tar shard or directory containing .tar shards")
    parser.add_argument("--filter-key", default="", help="Initial substring filter on key / clip_id / instruction")
    parser.add_argument("--filter-presence", type=int, default=None, choices=[0, 1, 2, 3], help="Initial presence filter")
    parser.add_argument("--sample-limit", type=int, default=None, help="Only index the first N matched samples")
    parser.add_argument(
        "--episode-limit",
        "--episode_limit",
        dest="episode_limit",
        type=int,
        default=None,
        help="Only index the first N matched episodes during initial scan",
    )
    parser.add_argument("--clip-id", type=str, default=None, help="Exact clip_id to open")
    parser.add_argument("--episode-key", type=str, default=None, help="Exact episode key to open")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="1-based episode index after filters are applied; if omitted and multiple episodes match, prompts in the terminal.",
    )
    parser.add_argument(
        "--render-mode",
        default="keypoint",
        choices=["keypoint", "skeleton", "mesh"],
        help="Visualization mode for the selected episode.",
    )
    parser.add_argument(
        "--output-mode",
        default="online",
        choices=["online", "offline", "both"],
        help="Send the same recording to a live viewer, an offline .rrd, or both.",
    )
    parser.add_argument(
        "--rrd-out",
        type=str,
        default=None,
        help="Offline .rrd output path. Used for offline/both; defaults to ./<episode>_<render>.rrd",
    )
    parser.add_argument(
        "--descriptor-manifest",
        type=str,
        default=None,
        help="Deprecated compatibility flag. MANO replay now reads mano.npy directly from each sample.",
    )
    parser.add_argument("--mano-dir", type=str, default=None, help="Optional MANO model directory override")
    parser.add_argument("--mano-device", type=str, default="cpu", help="Device for MANO replay, e.g. cpu or cuda:0")
    return parser


def _decode_image_rgb(image_bytes: bytes) -> np.ndarray:
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode JPEG image bytes")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _hand_keypoint_points(frame: dict, side: str) -> np.ndarray:
    wrist = frame[f"{side}_wrist"].reshape(1, 3).astype(np.float32)
    tips = frame[f"{side}_tips"].astype(np.float32)
    return np.concatenate([wrist, tips], axis=0)


def _hand_keypoint_lines(points: np.ndarray) -> list[np.ndarray]:
    wrist = points[0]
    return [np.stack([wrist, tip], axis=0).astype(np.float32) for tip in points[1:]]


def _mano_line_strips(joints: np.ndarray) -> list[np.ndarray]:
    return [joints[[joint_a, joint_b]].astype(np.float32) for chain in wv.MANO_JOINT_TREE for joint_a, joint_b in chain]


def _project_hand(points_world: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray, image_shape) -> tuple[np.ndarray, np.ndarray]:
    uv, valid = wv._project_points(points_world, c2w, intrinsic)
    mask = wv._clip_uv_mask(uv, valid, image_shape)
    return uv.astype(np.float32), mask


def _filter_line_strips_2d(strips: list[np.ndarray], c2w: np.ndarray, intrinsic: np.ndarray, image_shape) -> list[np.ndarray]:
    filtered = []
    for strip in strips:
        uv, mask = _project_hand(strip, c2w, intrinsic, image_shape)
        if bool(np.all(mask)):
            filtered.append(uv)
    return filtered


def _log_empty_hand(rr, world_root: str, image_root: str, *, with_mesh: bool):
    rr.log(f"{world_root}/points", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
    rr.log(f"{world_root}/lines", rr.LineStrips3D(np.zeros((0, 2, 3), dtype=np.float32)))
    rr.log(f"{image_root}/points", rr.Points2D(np.zeros((0, 2), dtype=np.float32)))
    rr.log(f"{image_root}/lines", rr.LineStrips2D(np.zeros((0, 2, 2), dtype=np.float32)))
    if with_mesh:
        rr.log(
            f"{world_root}/mesh",
            rr.Mesh3D(
                vertex_positions=np.zeros((0, 3), dtype=np.float32),
                triangle_indices=np.zeros((0, 3), dtype=np.uint32),
            ),
        )


def _log_keypoint_hand(rr, side: str, world_root: str, image_root: str, points_world: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray, image_shape):
    color = LEFT_COLOR if side == "left" else RIGHT_COLOR
    line_color = LEFT_LINE_COLOR if side == "left" else RIGHT_LINE_COLOR
    world_lines = _hand_keypoint_lines(points_world)
    uv, mask = _project_hand(points_world, c2w, intrinsic, image_shape)
    image_lines = _filter_line_strips_2d(world_lines, c2w, intrinsic, image_shape)
    rr.log(f"{world_root}/points", rr.Points3D(points_world, colors=color))
    rr.log(f"{world_root}/lines", rr.LineStrips3D(world_lines, colors=line_color))
    rr.log(f"{image_root}/points", rr.Points2D(uv[mask], colors=color))
    rr.log(f"{image_root}/lines", rr.LineStrips2D(image_lines, colors=line_color))


def _log_skeleton_hand(rr, side: str, world_root: str, image_root: str, joints_world: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray, image_shape):
    color = LEFT_COLOR if side == "left" else RIGHT_COLOR
    line_color = LEFT_LINE_COLOR if side == "left" else RIGHT_LINE_COLOR
    world_lines = _mano_line_strips(joints_world)
    uv, mask = _project_hand(joints_world, c2w, intrinsic, image_shape)
    image_lines = _filter_line_strips_2d(world_lines, c2w, intrinsic, image_shape)
    rr.log(f"{world_root}/points", rr.Points3D(joints_world, colors=color))
    rr.log(f"{world_root}/lines", rr.LineStrips3D(world_lines, colors=line_color))
    rr.log(f"{image_root}/points", rr.Points2D(uv[mask], colors=color))
    rr.log(f"{image_root}/lines", rr.LineStrips2D(image_lines, colors=line_color))


def _log_mesh_hand(rr, side: str, world_root: str, image_root: str, verts_world: np.ndarray, joints_world: np.ndarray, faces: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray, image_shape):
    color = LEFT_COLOR if side == "left" else RIGHT_COLOR
    line_color = LEFT_LINE_COLOR if side == "left" else RIGHT_LINE_COLOR
    image_lines = _filter_line_strips_2d(_mano_line_strips(joints_world), c2w, intrinsic, image_shape)
    uv, mask = _project_hand(joints_world, c2w, intrinsic, image_shape)
    rr.log(
        f"{world_root}/mesh",
        rr.Mesh3D(
            vertex_positions=verts_world,
            triangle_indices=faces.astype(np.uint32),
            vertex_colors=np.repeat(color[None, :], verts_world.shape[0], axis=0),
        ),
    )
    rr.log(f"{world_root}/points", rr.Points3D(joints_world, colors=color))
    rr.log(f"{world_root}/lines", rr.LineStrips3D(_mano_line_strips(joints_world), colors=line_color))
    rr.log(f"{image_root}/points", rr.Points2D(uv[mask], colors=color))
    rr.log(f"{image_root}/lines", rr.LineStrips2D(image_lines, colors=line_color))


def _log_camera(rr, image_rgb: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray):
    height, width = image_rgb.shape[:2]
    image_from_camera = np.array(
        [
            [float(intrinsic[0]), 0.0, float(intrinsic[2])],
            [0.0, float(intrinsic[1]), float(intrinsic[3])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rr.log(
        CAMERA_ROOT,
        rr.Transform3D(
            translation=c2w[:3, 3],
            mat3x3=c2w[:3, :3],
        ),
    )
    rr.log(
        PINHOLE_ROOT,
        rr.Pinhole(
            image_from_camera=image_from_camera,
            width=width,
            height=height,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(IMAGE_ROOT, rr.Image(image_rgb))


def log_episode(rr, backend: EpisodeViewerBackend, frames, render_mode: str):
    faces_right = None
    faces_left = None
    if render_mode == "mesh":
        from hawor.utils.process import get_mano_faces

        faces_right = np.concatenate([get_mano_faces(), MANO_FACE_EXTRA], axis=0).astype(np.uint32)
        faces_left = faces_right[:, [0, 2, 1]]

    for index, frame in enumerate(frames):
        rr.set_time("frame_idx", sequence=index)
        image_rgb = _decode_image_rgb(frame.sample["image_bytes"])

        if render_mode == "keypoint":
            keypoint_frame = backend.build_keypoint_frame(frame.summary, frame.lowdim_array, frame.mano_array)
            c2w = keypoint_frame["c2w"]
            intrinsic = keypoint_frame["intrinsic"]
            left_points = _hand_keypoint_points(keypoint_frame, "left")
            right_points = _hand_keypoint_points(keypoint_frame, "right")
        else:
            if frame.mano_array is None:
                raise ValueError(f"Sample {frame.summary.key} is missing mano.npy, required for {render_mode} mode")
            mano_frame = backend.build_mano_frame(frame.summary, frame.lowdim_array, frame.mano_array)
            c2w = mano_frame["c2w"]
            intrinsic = mano_frame["intrinsic"]
            left_points = mano_frame["left_joints"]
            right_points = mano_frame["right_joints"]

        _log_camera(rr, image_rgb, c2w, intrinsic)
        left_present, right_present = wv._presence_flags(frame.presence)

        if render_mode == "keypoint":
            if left_present:
                _log_keypoint_hand(rr, "left", LEFT_WORLD_ROOT, LEFT_IMAGE_ROOT, left_points, c2w, intrinsic, image_rgb.shape)
            else:
                _log_empty_hand(rr, LEFT_WORLD_ROOT, LEFT_IMAGE_ROOT, with_mesh=False)
            if right_present:
                _log_keypoint_hand(rr, "right", RIGHT_WORLD_ROOT, RIGHT_IMAGE_ROOT, right_points, c2w, intrinsic, image_rgb.shape)
            else:
                _log_empty_hand(rr, RIGHT_WORLD_ROOT, RIGHT_IMAGE_ROOT, with_mesh=False)
            continue

        if render_mode == "skeleton":
            if left_present:
                _log_skeleton_hand(rr, "left", LEFT_WORLD_ROOT, LEFT_IMAGE_ROOT, left_points, c2w, intrinsic, image_rgb.shape)
            else:
                _log_empty_hand(rr, LEFT_WORLD_ROOT, LEFT_IMAGE_ROOT, with_mesh=False)
            if right_present:
                _log_skeleton_hand(rr, "right", RIGHT_WORLD_ROOT, RIGHT_IMAGE_ROOT, right_points, c2w, intrinsic, image_rgb.shape)
            else:
                _log_empty_hand(rr, RIGHT_WORLD_ROOT, RIGHT_IMAGE_ROOT, with_mesh=False)
            continue

        if left_present:
            _log_mesh_hand(
                rr,
                "left",
                LEFT_WORLD_ROOT,
                LEFT_IMAGE_ROOT,
                mano_frame["left_verts"],
                mano_frame["left_joints"],
                faces_left,
                c2w,
                intrinsic,
                image_rgb.shape,
            )
        else:
            _log_empty_hand(rr, LEFT_WORLD_ROOT, LEFT_IMAGE_ROOT, with_mesh=True)
        if right_present:
            _log_mesh_hand(
                rr,
                "right",
                RIGHT_WORLD_ROOT,
                RIGHT_IMAGE_ROOT,
                mano_frame["right_verts"],
                mano_frame["right_joints"],
                faces_right,
                c2w,
                intrinsic,
                image_rgb.shape,
            )
        else:
            _log_empty_hand(rr, RIGHT_WORLD_ROOT, RIGHT_IMAGE_ROOT, with_mesh=True)


def log_static_scene(rr):
    rr.log(
        f"{WORLD_ROOT}/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float32),
            vectors=np.eye(3, dtype=np.float32),
            colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
            radii=np.array([0.005, 0.005, 0.005], dtype=np.float32),
        ),
        static=True,
    )


def main():
    args = build_parser().parse_args()
    tar_paths = wv.resolve_tar_paths(args.input)
    summaries = scan_sample_summaries(
        tar_paths,
        sample_limit=args.sample_limit,
        episode_limit=args.episode_limit,
        filter_key=args.filter_key,
        filter_presence=args.filter_presence,
    )
    if not summaries:
        raise SystemExit("No samples matched the current filters.")

    selected_episode_key = select_episode_key(
        summaries,
        clip_id=args.clip_id,
        episode_key=args.episode_key,
        episode_index=args.episode_index,
    )
    backend = EpisodeViewerBackend(
        summaries,
        descriptor_manifest=args.descriptor_manifest,
        mano_dir=args.mano_dir,
        mano_device=args.mano_device,
    )

    frames = backend.load_episode_frames(selected_episode_key)
    if not frames:
        raise SystemExit(f"No frames found for episode {selected_episode_key}")

    rr = load_rerun()
    rrd_path = configure_recording(
        rr,
        output_mode=args.output_mode,
        rrd_out=args.rrd_out,
        episode_key=selected_episode_key,
        render_mode=args.render_mode,
    )
    send_default_blueprint(rr)
    log_static_scene(rr)
    log_episode(rr, backend, frames, args.render_mode)

    first = frames[0]
    print(
        f"Loaded episode={selected_episode_key} clip_id={first.summary.clip_id or '-'} "
        f"frames={len(frames)} render_mode={args.render_mode}",
        flush=True,
    )
    print(
        f"Input={Path(args.input).expanduser().resolve()} descriptor_manifest={args.descriptor_manifest or '-'}",
        flush=True,
    )
    if rrd_path is not None:
        print(f"Saved Rerun recording to: {rrd_path}", flush=True)


if __name__ == "__main__":
    main()
