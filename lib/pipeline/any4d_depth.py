"""Any4D depth inference helpers for stage3."""

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_exists(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Any4D] {what} not found: {path}")


def _env_flag_on(name: str, *, default_on: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default_on
    value = str(raw).strip().lower()
    if value == "":
        return default_on
    if value in ("0", "false", "no", "off"):
        return False
    return True


def resolve_any4d_paths(project_root=None, any4d_repo_root=None, checkpoint_path=None, resolution_set=None, use_amp=None):
    project_root = Path(project_root or PROJECT_ROOT).resolve()

    repo_root = any4d_repo_root or os.environ.get(
        "HAWOR_ANY4D_REPO_ROOT",
        str(project_root / "thirdparty" / "Any4D"),
    )
    repo_root = Path(repo_root).expanduser()
    if not repo_root.is_absolute():
        repo_root = (project_root / repo_root).resolve()

    checkpoint = checkpoint_path or os.environ.get(
        "HAWOR_ANY4D_CHECKPOINT_PATH",
        str(repo_root / "checkpoints" / "any4d_4v_combined.pth"),
    )
    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (project_root / checkpoint).resolve()

    if resolution_set is None:
        resolution_set = int(os.environ.get("HAWOR_ANY4D_RESOLUTION", "518"))
    if use_amp is None:
        use_amp = _env_flag_on("HAWOR_ANY4D_USE_AMP", default_on=True)

    return str(repo_root), str(checkpoint), int(resolution_set), bool(use_amp)


@contextlib.contextmanager
def _suppress_any4d_init_io():
    if os.environ.get("HAWOR_ANY4D_VERBOSE", "0").strip() == "1":
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _prepend_sys_path(path: str):
    if path not in sys.path:
        sys.path.insert(0, path)


def _any4d_config_dict(any4d_root: str):
    use_sdpa = _env_flag_on("HAWOR_ANY4D_USE_PYTORCH_SDPA", default_on=True)
    sdpa_override = (
        "+model.encoder.use_pytorch_sdpa=true"
        if use_sdpa
        else "+model.encoder.use_pytorch_sdpa=false"
    )
    return {
        "path": os.path.join(any4d_root, "configs", "train.yaml"),
        "config_overrides": [
            "machine=local",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            sdpa_override,
            "model/task=images_only",
        ],
    }


def _direct_frame_path(frame_source, frame_idx: int):
    image_paths = getattr(frame_source, "image_paths", None)
    if image_paths is None:
        return None
    if frame_idx < 0 or frame_idx >= len(image_paths):
        return None
    path = image_paths[frame_idx]
    return path if os.path.exists(path) else None


def _materialize_frame(frame_source, frame_idx: int, temp_dir: str):
    image = frame_source.get_frame(frame_idx, rgb=False)
    path = os.path.join(temp_dir, f"{int(frame_idx):06d}.png")
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"[Any4D] failed to materialize frame {frame_idx} to {path}")
    return path


def _resolve_image_paths(frame_source, frame_indices: Sequence[int], ref_frame_idx: int, temp_dir: Optional[str] = None):
    image_paths = []
    for frame_idx in [ref_frame_idx, *frame_indices]:
        path = _direct_frame_path(frame_source, frame_idx)
        if path is None:
            if temp_dir is None:
                raise RuntimeError(
                    f"[Any4D] frame_source does not expose direct paths for frame {frame_idx}, "
                    "and no temp_dir was provided for materialization."
                )
            path = _materialize_frame(frame_source, frame_idx, temp_dir)
        image_paths.append(path)
    return image_paths


@contextlib.contextmanager
def _build_image_paths(frame_source, frame_indices: Sequence[int], ref_frame_idx: int):
    with tempfile.TemporaryDirectory(prefix="hawor-any4d-") as temp_dir:
        yield _resolve_image_paths(frame_source, frame_indices, ref_frame_idx, temp_dir=temp_dir)


def _import_any4d_modules(repo_root: str):
    any4d_scripts_dir = os.path.join(repo_root, "scripts")
    _prepend_sys_path(repo_root)
    _prepend_sys_path(any4d_scripts_dir)

    import inference_test as any4d_inference_test
    from any4d.utils.image import load_images

    return any4d_inference_test, load_images


def _load_any4d_views(load_images, image_paths, resolution_set: int):
    return load_images(
        image_paths,
        resize_mode="fixed_mapping",
        resolution_set=resolution_set,
        norm_type="dinov2",
        patch_size=14,
        verbose=False,
        compute_moge_mask=False,
        binary_mask_path=None,
    )


def _predict_depths_from_views(any4d_inference_test, runner, views, frame_count: int):
    device = str(next(runner["model"].parameters()).device)
    pred_result = any4d_inference_test.sample_inference(
        model=runner["model"],
        views=views,
        device=device,
        use_amp=runner["use_amp"],
    )

    depth_list = []
    for target_i in range(frame_count):
        view_idx = 1 + target_i
        depth_z = (
            pred_result[f"pred{view_idx}"]["pts3d_cam"][..., 2:3][0]
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
        depth_list.append(depth_z.astype(np.float32))

    return np.stack(depth_list, axis=0)


def build_any4d_views(frame_source, frame_indices, runner=None, *, any4d_repo_root=None, checkpoint_path=None, resolution_set=None, use_amp=None, image_paths=None):
    frame_indices = [int(frame_idx) for frame_idx in frame_indices]
    if not frame_indices:
        raise ValueError("[Any4D] frame_indices is empty")

    runner = runner or build_any4d_runner(
        any4d_repo_root=any4d_repo_root,
        checkpoint_path=checkpoint_path,
        resolution_set=resolution_set,
        use_amp=use_amp,
    )

    load_images = runner["load_images"]
    if image_paths is None:
        ref_frame_idx = frame_indices[len(frame_indices) // 2]
        with _build_image_paths(frame_source, frame_indices, ref_frame_idx) as resolved_image_paths:
            return _load_any4d_views(load_images, resolved_image_paths, runner["resolution_set"])
    return _load_any4d_views(load_images, image_paths, runner["resolution_set"])


def predict_any4d_depths_from_views(frame_indices, views, runner=None, *, any4d_repo_root=None, checkpoint_path=None, resolution_set=None, use_amp=None):
    frame_indices = [int(frame_idx) for frame_idx in frame_indices]
    if not frame_indices:
        raise ValueError("[Any4D] frame_indices is empty")

    runner = runner or build_any4d_runner(
        any4d_repo_root=any4d_repo_root,
        checkpoint_path=checkpoint_path,
        resolution_set=resolution_set,
        use_amp=use_amp,
    )

    any4d_inference_test = runner["inference_module"]
    return _predict_depths_from_views(any4d_inference_test, runner, views, len(frame_indices))


def build_any4d_runner(any4d_repo_root=None, checkpoint_path=None, resolution_set=None, use_amp=None):
    repo_root, checkpoint_path, resolution_set, use_amp = resolve_any4d_paths(
        PROJECT_ROOT,
        any4d_repo_root,
        checkpoint_path,
        resolution_set,
        use_amp,
    )
    ensure_exists(repo_root, "Any4D repo root")
    ensure_exists(checkpoint_path, "Any4D checkpoint")

    any4d_inference_test, load_images = _import_any4d_modules(repo_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with _suppress_any4d_init_io():
        model = any4d_inference_test.init_inference_model(
            _any4d_config_dict(repo_root),
            checkpoint_path,
            device,
        )

    return {
        "model": model,
        "inference_module": any4d_inference_test,
        "load_images": load_images,
        "repo_root": repo_root,
        "checkpoint_path": checkpoint_path,
        "resolution_set": resolution_set,
        "use_amp": use_amp,
    }


def predict_any4d_depth_batch(frame_source, frame_indices, runner=None, *, any4d_repo_root=None, checkpoint_path=None, resolution_set=None, use_amp=None, image_paths=None):
    frame_indices = [int(frame_idx) for frame_idx in frame_indices]
    if not frame_indices:
        raise ValueError("[Any4D] frame_indices is empty")

    runner = runner or build_any4d_runner(
        any4d_repo_root=any4d_repo_root,
        checkpoint_path=checkpoint_path,
        resolution_set=resolution_set,
        use_amp=use_amp,
    )

    views = build_any4d_views(
        frame_source,
        frame_indices,
        runner=runner,
        any4d_repo_root=any4d_repo_root,
        checkpoint_path=checkpoint_path,
        resolution_set=resolution_set,
        use_amp=use_amp,
        image_paths=image_paths,
    )
    return predict_any4d_depths_from_views(
        frame_indices,
        views,
        runner=runner,
        any4d_repo_root=any4d_repo_root,
        checkpoint_path=checkpoint_path,
        resolution_set=resolution_set,
        use_amp=use_amp,
    )
