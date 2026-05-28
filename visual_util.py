# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import numpy as np
import requests
import trimesh
from matplotlib import colormaps
from scipy.spatial.transform import Rotation


def predictions_to_glb(
    predictions: dict,
    conf_thres: float = 20.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    target_dir: str | None = None,
    max_points: int = 300000,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
    hand_data: dict | None = None,
    frame_indices: np.ndarray | None = None,
    hand_scale: float = 1.0,
    hand_auto_scale: bool = True,
    hand_max_frames: int | None = None,
) -> trimesh.Scene:
    """Convert VGGT-Omega camera/depth predictions to a GLB scene."""
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    conf_thres = max(2.0, float(conf_thres))

    points = predictions["world_points_from_depth"]
    conf = predictions["depth_conf"]
    if filter_depth_edges and "depth" in predictions:
        conf = conf.copy()
        conf[depth_edge(predictions["depth"][..., 0], rtol=depth_edge_rtol)] = 0.0
    images = predictions["images"]
    camera_matrices = predictions["extrinsic"]

    if mask_sky and target_dir is not None:
        conf = apply_sky_mask(conf, target_dir)

    vertices = points.reshape(-1, 3)
    colors = _images_to_rgb(images).reshape(-1, 3)
    colors = (colors * 255).clip(0, 255).astype(np.uint8)
    conf = conf.reshape(-1)

    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf)
    if conf_thres > 0 and np.any(mask):
        conf_threshold = np.percentile(conf[mask], conf_thres)
        mask &= conf >= conf_threshold
    mask &= conf > 1e-5

    if mask_black_bg:
        mask &= colors.sum(axis=1) >= 16
    if mask_white_bg:
        mask &= ~((colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240))

    vertices = vertices[mask]
    colors = colors[mask]
    vertices, colors = _limit_points(vertices, colors, max_points)

    if vertices.size == 0:
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        colors = np.array([[255, 255, 255]], dtype=np.uint8)
        scene_scale = 1.0
    else:
        lower = np.percentile(vertices, 5, axis=0)
        upper = np.percentile(vertices, 95, axis=0)
        scene_scale = float(np.linalg.norm(upper - lower))
        if scene_scale <= 0:
            scene_scale = 1.0

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=vertices, colors=colors))

    extrinsics = np.zeros((len(camera_matrices), 4, 4), dtype=np.float64)
    extrinsics[:, :3, :4] = camera_matrices
    extrinsics[:, 3, 3] = 1.0

    if show_cam:
        # Light gray at the earliest frame -> near-black at the latest; matches the
        # subdued look of the hand overlay instead of a saturated rainbow.
        cam_light = np.array([170, 170, 170])
        cam_dark = np.array([20, 20, 20])
        n_cam = len(extrinsics)
        for i, world_to_camera in enumerate(extrinsics):
            camera_to_world = np.linalg.inv(world_to_camera)
            t = i / max(n_cam - 1, 1)
            color = tuple(int(round(c)) for c in cam_light * (1.0 - t) + cam_dark * t)
            integrate_camera_into_scene(scene, camera_to_world, color, scene_scale)

    if hand_data is not None:
        add_hand_meshes(
            scene,
            predictions,
            hand_data,
            frame_indices,
            hand_scale=hand_scale,
            hand_auto_scale=hand_auto_scale,
            hand_max_frames=hand_max_frames,
            scene_scale=scene_scale,
        )

    return apply_scene_alignment(scene, extrinsics)


# One base hue per hand (left=pink, right=blue), lerped from a pale tint at the
# earliest kept frame to a saturated end at the latest. Soft like the reference.
HAND_PALETTES = {
    "left":  {"light": np.array([220, 235, 240]), "dark": np.array([ 70, 170, 210])},
    "right": {"light": np.array([245, 220, 230]), "dark": np.array([210, 130, 170])},
}


def _estimate_hand_scale(predictions: dict, frames_verts: list, frame_indices: np.ndarray) -> float:
    """Estimate metric->VGGT scale by matching hand depth to VGGT depth at the hand pixel.

    `frames_verts` is a list of (vggt_frame_idx, camera_space_vertices) pairs. The projection
    of a camera-space point is scale-invariant, so we can locate the hand pixel without knowing
    the scale, read VGGT's predicted depth there, and infer s = depth_vggt / depth_camera.
    """
    intrinsic = predictions.get("intrinsic")
    depth = predictions.get("depth")
    if intrinsic is None or depth is None:
        return 1.0
    depth = depth[..., 0] if depth.ndim == 4 else depth  # (S,H,W)
    conf = predictions.get("depth_conf")
    height, width = depth.shape[-2:]

    ratios = []
    for vggt_idx, verts in frames_verts:
        centroid = verts.mean(axis=0)
        z_cam = float(centroid[2])
        if not np.isfinite(z_cam) or z_cam <= 1e-3:
            continue
        K = intrinsic[vggt_idx]
        u = int(round(centroid[0] / z_cam * K[0, 0] + K[0, 2]))
        v = int(round(centroid[1] / z_cam * K[1, 1] + K[1, 2]))
        if not (0 <= u < width and 0 <= v < height):
            continue
        if conf is not None and conf[vggt_idx, v, u] <= 1e-5:
            continue
        depth_vggt = float(depth[vggt_idx, v, u])
        if np.isfinite(depth_vggt) and depth_vggt > 0:
            ratios.append(depth_vggt / z_cam)

    return float(np.median(ratios)) if ratios else 1.0


def add_hand_meshes(
    scene: trimesh.Scene,
    predictions: dict,
    hand_data: dict,
    frame_indices: np.ndarray | None,
    hand_scale: float = 1.0,
    hand_auto_scale: bool = True,
    hand_max_frames: int | None = None,
    scene_scale: float = 1.0,
    hand_spread_frac: float = 0.10,
    collision_iters: int = 60,
    collision_gap: float = 1.3,
) -> None:
    """Place per-frame camera-space hand meshes into the (VGGT world) scene.

    A camera-space vertex v maps to VGGT world via world = R_i^T (s * v - t_i), where
    [R_i | t_i] is VGGT's world->camera extrinsic for the frame and s converts the metric
    MANO hand into VGGT's normalized scale.
    """
    extrinsic = predictions["extrinsic"]  # (S, 3, 4)
    num_frames = extrinsic.shape[0]
    if frame_indices is None:
        frame_indices = np.arange(num_frames)
    frame_indices = np.asarray(frame_indices).astype(np.int64)

    # original-video-frame -> VGGT frame index (first match wins)
    orig_to_vggt = {}
    for vggt_idx, orig in enumerate(frame_indices[:num_frames]):
        orig_to_vggt.setdefault(int(orig), vggt_idx)

    placements = []  # (vggt_idx, vertices, faces, side)
    auto_inputs = []  # (vggt_idx, vertices) for scale estimation
    for side in ("left", "right"):
        frames = hand_data.get(f"{side}_frames")
        verts_all = hand_data.get(f"{side}_vertices")
        faces = hand_data.get(f"faces_{side}")
        if frames is None or verts_all is None or len(frames) == 0:
            continue
        for orig_frame, verts in zip(np.asarray(frames).astype(np.int64), verts_all):
            vggt_idx = orig_to_vggt.get(int(orig_frame))
            if vggt_idx is None or not np.isfinite(verts).all():
                continue
            placements.append((vggt_idx, np.asarray(verts, dtype=np.float64), faces, side))
            auto_inputs.append((vggt_idx, verts))

    if not placements:
        return

    # Auto-scale uses every available hand so the scalar stays stable regardless of
    # how many frames the user chooses to render.
    s = hand_scale
    if hand_auto_scale:
        s = _estimate_hand_scale(predictions, auto_inputs, frame_indices) * hand_scale

    # Optional subsample: keep at most `hand_max_frames` VGGT frames, evenly spaced
    # across the timeline. Picks frame indices (not placements) so left/right at the
    # same instant are kept together.
    if hand_max_frames is not None and hand_max_frames > 0:
        unique_idxs = sorted({p[0] for p in placements})
        if len(unique_idxs) > hand_max_frames:
            picks = np.linspace(0, len(unique_idxs) - 1, hand_max_frames).round().astype(int)
            kept = {unique_idxs[i] for i in picks}
            placements = [p for p in placements if p[0] in kept]

    # World-space direction of camera 0's "left" — used for the directional bias step.
    R0 = extrinsic[0, :3, :3]
    left_dir_world = R0.T @ np.array([-1.0, 0.0, 0.0])
    norm = float(np.linalg.norm(left_dir_world))
    left_dir_world = left_dir_world / norm if norm > 1e-9 else np.array([-1.0, 0.0, 0.0])

    # Step 1: transform each placement's camera-space vertices into VGGT world.
    items = []
    for vggt_idx, verts, faces, side in placements:
        rotation = extrinsic[vggt_idx, :3, :3]
        translation = extrinsic[vggt_idx, :3, 3]
        world_verts = (s * verts - translation) @ rotation  # R^T (s v - t) for row vectors
        items.append({"idx": int(vggt_idx), "verts": world_verts, "faces": np.asarray(faces), "side": side})

    # Step 2: directional bias — push left hands toward camera-left, right hands toward
    # camera-right, with deterministic per-frame randomness so they don't all stack on a
    # single line. This matches the natural left/right grouping in the reference image.
    for it in items:
        rng = np.random.default_rng(it["idx"] * 13 + (0 if it["side"] == "left" else 7))
        # left_dir_world points toward camera-left; left hand follows it, right hand against.
        sign = 1.0 if it["side"] == "left" else -1.0
        bias_mag = scene_scale * hand_spread_frac * rng.uniform(0.5, 2.0)
        perp = rng.standard_normal(3) * scene_scale * hand_spread_frac * 0.3
        it["verts"] = it["verts"] + sign * left_dir_world * bias_mag + perp

    # Step 3: collision relaxation — iteratively push apart any pair of placements whose
    # centroids are closer than `collision_gap * median(hand_radius)`. Each iteration
    # applies a rigid translation to the offending placements; we stop early when no
    # pair overlaps. O(n^2) per iter but n <= 2 * hand_max_frames, so trivial.
    n = len(items)
    if n >= 2 and collision_iters > 0:
        centroids = np.stack([it["verts"].mean(axis=0) for it in items])
        radii = np.array(
            [float(np.linalg.norm(it["verts"].max(0) - it["verts"].min(0))) * 0.5 for it in items]
        )
        sep = float(np.median(radii)) * float(collision_gap)
        if sep > 1e-6:
            for _ in range(collision_iters):
                shifts = np.zeros_like(centroids)
                moved = False
                for i in range(n):
                    for j in range(i + 1, n):
                        d = centroids[j] - centroids[i]
                        r = float(np.linalg.norm(d))
                        if r < sep:
                            push = (sep - r) * 0.5 + 1e-4
                            if r > 1e-6:
                                direction = d / r
                            else:
                                rng2 = np.random.default_rng(i * 1000 + j)
                                direction = rng2.standard_normal(3)
                                direction /= np.linalg.norm(direction) + 1e-9
                            shifts[i] -= direction * push
                            shifts[j] += direction * push
                            moved = True
                if not moved:
                    break
                centroids = centroids + shifts
                for it, sh in zip(items, shifts):
                    it["verts"] = it["verts"] + sh

    # Step 4: color and add to scene. Earliest kept frame -> palette light, latest -> dark.
    kept_idxs = sorted({it["idx"] for it in items})
    time_pos = {idx: i / max(len(kept_idxs) - 1, 1) for i, idx in enumerate(kept_idxs)}
    for it in items:
        pos = time_pos[it["idx"]]
        pal = HAND_PALETTES[it["side"]]
        rgb = tuple(int(round(c)) for c in pal["light"] * (1.0 - pos) + pal["dark"] * pos)
        mesh = trimesh.Trimesh(vertices=it["verts"], faces=it["faces"], process=False)
        mesh.visual.face_colors[:, :3] = rgb
        scene.add_geometry(mesh)


def _images_to_rgb(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def _limit_points(vertices: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices, colors
    indices = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
    return vertices[indices], colors[indices]


def depth_edge(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])

    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)

    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)

    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


def camera_trajectory_to_glb(
    predictions: dict,
    scatter_frac: float = 0.04,
    glyph_scale_frac: float = 1.0,
    tube_radius_frac: float = 0.004,
) -> trimesh.Scene:
    """Build a standalone GLB showing the camera trajectory only.

    Each VGGT frame is drawn as the same wireframe-cone glyph used in the main
    scene (small box + diagonals), placed at the camera's world position and
    oriented by the camera's world rotation. Consecutive positions are joined by
    a thin cylinder so the time order reads as a polyline. Centers get a
    deterministic per-index Gaussian scatter so a dense trajectory loosens up
    instead of stacking on a tight curve. Colors fade light gray -> near-black.
    """
    extrinsic = predictions["extrinsic"]
    n = len(extrinsic)
    if n == 0:
        return trimesh.Scene()

    extrinsics_h = np.zeros((n, 4, 4), dtype=np.float64)
    extrinsics_h[:, :3, :4] = extrinsic
    extrinsics_h[:, 3, 3] = 1.0

    cam_to_world = np.array([np.linalg.inv(E) for E in extrinsics_h])  # (n,4,4)
    cam_pos = cam_to_world[:, :3, 3].copy()

    if n >= 2:
        span = float(np.linalg.norm(cam_pos.max(0) - cam_pos.min(0)))
    else:
        span = 0.0
    if span < 1e-6:
        span = 1.0

    # Deterministic per-index Gaussian jitter.
    if scatter_frac > 0:
        sigma = span * scatter_frac
        for i in range(n):
            cam_pos[i] = cam_pos[i] + np.random.default_rng(int(i) * 257 + 11).standard_normal(3) * sigma
    cam_to_world[:, :3, 3] = cam_pos  # propagate scatter into the glyph transform

    scene = trimesh.Scene()
    cam_light = np.array([170, 170, 170])
    cam_dark = np.array([20, 20, 20])

    def lerp_gray(t: float) -> tuple:
        return tuple(int(round(c)) for c in cam_light * (1.0 - t) + cam_dark * t)

    # Use span as the implicit scale for integrate_camera_into_scene (frustum sized
    # relative to the trajectory itself rather than a point cloud that isn't there).
    glyph_scale = span * float(glyph_scale_frac)
    for i in range(n):
        t = i / max(n - 1, 1)
        integrate_camera_into_scene(scene, cam_to_world[i], lerp_gray(t), glyph_scale)

    r_tube = span * tube_radius_frac
    z_axis = np.array([0.0, 0.0, 1.0])
    for i in range(n - 1):
        a, b = cam_pos[i], cam_pos[i + 1]
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            continue
        v = seg / length
        dot = float(np.clip(np.dot(z_axis, v), -1.0, 1.0))
        if dot > 1.0 - 1e-9:
            R = np.eye(3)
        elif dot < -1.0 + 1e-9:
            R = Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()
        else:
            axis = np.cross(z_axis, v)
            axis = axis / (np.linalg.norm(axis) + 1e-9)
            R = Rotation.from_rotvec(axis * np.arccos(dot)).as_matrix()

        cyl = trimesh.creation.cylinder(radius=r_tube, height=length, sections=8)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (a + b) / 2.0
        cyl.apply_transform(T)
        cyl.visual.face_colors[:, :3] = lerp_gray((i + 0.5) / max(n - 1, 1))
        scene.add_geometry(cyl)

    return apply_scene_alignment(scene, extrinsics_h)


def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    cam_width = scene_scale * 0.025
    cam_height = scene_scale * 0.05

    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    complete_transform = transform @ get_opengl_conversion_matrix() @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices = transform_points(complete_transform, vertices)

    camera_mesh = trimesh.Trimesh(vertices=vertices, faces=compute_camera_faces(camera_cone_shape))
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(scene: trimesh.Scene, extrinsics: np.ndarray) -> trimesh.Scene:
    opengl_conversion_matrix = get_opengl_conversion_matrix()
    scene.apply_transform(np.linalg.inv(extrinsics[0]) @ opengl_conversion_matrix)
    return scene


def get_opengl_conversion_matrix() -> np.ndarray:
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int | None = None) -> np.ndarray:
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    return points[..., :dim].reshape(*initial_shape, dim)


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    faces = []
    num_vertices = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices

        faces.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces += [(v3, v2, v1) for v1, v2, v3 in faces]
    return np.array(faces)


def apply_sky_mask(conf: np.ndarray, target_dir: str) -> np.ndarray:
    image_dir = os.path.join(target_dir, "images")
    image_names = sorted(os.listdir(image_dir))
    height, width = conf.shape[-2:]
    masks = []
    skyseg_session = None

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(target_dir, "sky_masks", image_name)
        if os.path.exists(mask_path):
            sky_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            if not os.path.exists("skyseg.onnx"):
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                    "skyseg.onnx",
                )
            if skyseg_session is None:
                import onnxruntime

                skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
            sky_mask = segment_sky(image_path, skyseg_session, mask_path)

        if sky_mask.shape != (height, width):
            sky_mask = cv2.resize(sky_mask, (width, height))
        masks.append(sky_mask)

    return conf * (np.array(masks) > 0.1).astype(np.float32)


def segment_sky(image_path: str, onnx_session, mask_filename: str) -> np.ndarray:
    image = cv2.imread(image_path)
    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map)
    output_mask[result_map < 32] = 255

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size: list[int], image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, dsize=(input_size[0], input_size[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype=np.float32)
    image = (image / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)
    image = image.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: image})
    result = np.array(result).squeeze()
    result_min = np.min(result)
    result_max = np.max(result)
    if result_max > result_min:
        result = (result - result_min) / (result_max - result_min)
    else:
        result = np.zeros_like(result)
    return (result * 255).astype("uint8")


def download_file_from_url(url: str, filename: str) -> None:
    tmp_filename = f"{filename}.tmp"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(tmp_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    os.replace(tmp_filename, filename)
