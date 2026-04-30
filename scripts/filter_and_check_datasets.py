#!/usr/bin/env python3
"""Unified dataset filtering and checking utility for EgoVLA.

This script consolidates the repository's existing dataset gates into one
standalone CLI:

  - wds:       scan VLA/VLM WebDataset shards and report bad samples
  - zarr-list: validate/filter zarr list files by episode count
  - hf-list:   validate/filter HF arrow/parquet dataset directories

The WebDataset checker intentionally reuses ``src.dataset.sanity_checks`` so
the thresholds match training-time data skips.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VALID_MAPPING_TYPES = {"human", "real_world"}


class DataSkipError(ValueError):
    """Fallback used before the training sanity module is lazily imported."""


class MissingOrInvalidFilesError(DataSkipError):
    """Fallback used before the training sanity module is lazily imported."""


def build_lowdim_slices(cameras):
    """Return HaWoR/EgoVLA lowdim slices for the requested camera list."""
    if not cameras or cameras[0] != "head":
        raise ValueError(f"cameras[0] must be 'head', got {cameras!r}")
    slices = {
        "wrist_state": (0, 18),
        "hand_state": (18, 48),
        "wrist_action": (48, 66),
        "hand_action": (66, 96),
    }
    offset = 96
    for cam in cameras:
        slices[f"{cam}_extrinsic"] = (offset, offset + 16)
        slices[f"{cam}_intrinsic"] = (offset + 16, offset + 20)
        offset += 20
    return slices


def load_training_checkers() -> None:
    """Compatibility hook kept so the copied script stays self-contained."""
    return None


class NonFiniteDataError(DataSkipError):
    pass


class OutlierDataError(DataSkipError):
    pass


class ExtrinsicInvalidError(DataSkipError):
    pass


class IntrinsicInvalidError(DataSkipError):
    pass


class InstructionInvalidError(DataSkipError):
    pass


class ImageQualityError(DataSkipError):
    pass


class DepthQualityError(DataSkipError):
    pass


class Rot6DInvalidError(DataSkipError):
    pass


class ExtremeStateActionDeltaError(DataSkipError):
    pass


def _rot6d_quality(rot6d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    finite = np.all(np.isfinite(arr), axis=1)
    col1 = arr[:, :3]
    col2 = arr[:, 3:6]
    n1 = np.linalg.norm(col1, axis=1, keepdims=True)
    n2 = np.linalg.norm(col2, axis=1, keepdims=True)
    unit1 = col1 / (n1 + 1e-8)
    unit2 = col2 / (n2 + 1e-8)
    orth = np.abs(np.sum(unit1 * unit2, axis=1))
    return finite, orth


def _rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    arr = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    a1 = arr[:, :3]
    a2 = arr[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    proj = np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=2)


def _rotation_delta_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rel = np.einsum("nij,njk->nik", _rot6d_to_rotmat(a), np.transpose(_rot6d_to_rotmat(b), (0, 2, 1)))
    traces = np.trace(rel, axis1=1, axis2=2)
    cos_theta = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_theta)


class DataChecker:
    POST_NORMALIZE_OUTLIER_THRESHOLD = 10.0

    def __init__(self, sanity_cfg: dict[str, Any] | None = None):
        cfg = dict(sanity_cfg or {})
        rot_cfg = dict(cfg.get("rot6d", {}) or {})
        delta_cfg = dict(cfg.get("state_action_delta", {}) or {})
        self.rot6d_orthogonality_threshold = float(rot_cfg.get("orthogonality_threshold", 0.1))
        self.delta_wrist_translation_threshold = float(delta_cfg.get("wrist_translation_threshold", 2.0))
        self.delta_wrist_rotation_threshold = float(delta_cfg.get("wrist_rotation_threshold", 10.0))
        self.delta_fingertips_displacement_threshold = float(delta_cfg.get("fingertips_displacement_threshold", 2.0))
        self.seen = 0

    def note_sample_seen(self) -> None:
        self.seen += 1

    def check(self, **fields) -> None:
        for key, value in fields.items():
            if value is None:
                continue
            if key == "sample_schema":
                self._check_sample_schema(*value)
            elif key == "intrinsic":
                self._check_intrinsic(value)
            elif key == "extrinsic":
                self._check_extrinsic(value)
            elif key == "instruction":
                self._check_instruction(*value)
            elif key == "finite":
                self._check_finite(value)
            elif key == "rot6d":
                self._check_rot6d(value)
            elif key == "state_action_delta":
                self._check_state_action_delta(*value)
            elif key == "image":
                self._check_image(value)
            elif key == "depth":
                self._check_depth(*value)
            elif key == "post_normalize":
                self._check_post_normalize(value)
            else:
                raise KeyError(f"unknown check field: {key}")

    @staticmethod
    def _check_sample_schema(sample: dict[str, Any], cfg: dict[str, Any]) -> None:
        for key in tuple(cfg.get("required_keys", ())):
            if key not in sample or sample.get(key) is None:
                raise MissingOrInvalidFilesError(f"missing required field: {key}")
        for key, dim in dict(cfg.get("expected_last_dim", {}) or {}).items():
            value = sample.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == 0 or arr.shape[-1] != int(dim):
                raise MissingOrInvalidFilesError(f"invalid {key} shape={tuple(arr.shape)} expected last_dim={dim}")
        meta_keys = tuple(cfg.get("required_meta_keys", ()))
        if meta_keys:
            meta = sample.get("meta.json")
            if not isinstance(meta, dict):
                raise MissingOrInvalidFilesError("missing or invalid meta.json mapping")
            for meta_key in meta_keys:
                if meta_key not in meta or meta.get(meta_key) is None:
                    raise MissingOrInvalidFilesError(f"meta missing required key: {meta_key}")

    @staticmethod
    def _check_intrinsic(intr: np.ndarray) -> None:
        arr = np.asarray(intr, dtype=np.float32)
        if arr.shape != (4,) or not np.isfinite(arr).all():
            raise IntrinsicInvalidError(f"invalid intrinsic shape/finite: {arr.shape}")
        if float(arr[0]) <= 0.0 or float(arr[1]) <= 0.0:
            raise IntrinsicInvalidError(f"non-positive focal: fx={arr[0]}, fy={arr[1]}")
        if float(arr[0]) >= 10000.0 or float(arr[1]) >= 10000.0:
            raise IntrinsicInvalidError(f"focal too large: fx={arr[0]}, fy={arr[1]}")

    @staticmethod
    def _check_extrinsic(ext: np.ndarray) -> None:
        arr = np.asarray(ext, dtype=np.float32)
        mats = arr.reshape(-1, 4, 4) if arr.ndim == 3 else arr.reshape(1, 4, 4)
        for mat in mats:
            if not np.isfinite(mat).all():
                raise ExtrinsicInvalidError("contains non-finite values")
            if not np.allclose(mat[3], [0.0, 0.0, 0.0, 1.0], atol=1e-3):
                raise ExtrinsicInvalidError(f"invalid last row: {mat[3].tolist()}")
            det = float(np.linalg.det(mat[:3, :3]))
            if abs(det - 1.0) > 0.1:
                raise ExtrinsicInvalidError(f"det(R)={det:.4g} deviates from 1.0")

    @staticmethod
    def _check_instruction(instruction, instruction_num) -> None:
        if instruction is None:
            raise InstructionInvalidError("instruction is None")
        try:
            n = int(instruction_num)
        except Exception as exc:
            raise InstructionInvalidError(f"invalid instruction_num={instruction_num!r}") from exc
        if n <= 0:
            raise InstructionInvalidError(f"instruction_num={n} <= 0")
        placeholders = {"none", "unknown", "n/a", "null", "todo"}

        def bad(item) -> bool:
            return not isinstance(item, str) or not item.strip() or item.strip().lower() in placeholders

        if isinstance(instruction, (list, tuple)):
            if all(bad(item) for item in instruction):
                raise InstructionInvalidError("all candidate instructions are empty/placeholders")
        elif bad(instruction):
            raise InstructionInvalidError(f"instruction is empty or placeholder: {instruction!r}")

    @staticmethod
    def _check_finite(values: dict[str, Any]) -> None:
        for name, value in values.items():
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.size and np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                raise NonFiniteDataError(f"Non-finite value in {name}: shape={arr.shape}")

    def _check_rot6d(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            arr = np.asarray(value)
            if arr.size == 0 or arr.ndim == 0 or arr.shape[-1] < 18:
                continue
            rot = np.concatenate([arr[..., 6:12].reshape(-1, 6), arr[..., 12:18].reshape(-1, 6)], axis=0)
            finite, orth = _rot6d_quality(rot)
            invalid = (~finite) | (orth > self.rot6d_orthogonality_threshold)
            if invalid.any():
                raise Rot6DInvalidError(f"{name}: invalid_rot6d={int(invalid.sum())}/{invalid.size}, max_orth_err={float(orth[invalid].max()):.4f}")

    def _check_state_action_delta(self, wrist_state, hand_state, wrist_action, hand_action) -> None:
        ws = np.asarray(wrist_state, dtype=np.float64).reshape(-1, wrist_state.shape[-1])
        wa = np.asarray(wrist_action, dtype=np.float64).reshape(-1, wrist_action.shape[-1])
        hs = np.asarray(hand_state, dtype=np.float64).reshape(-1, hand_state.shape[-1])
        ha = np.asarray(hand_action, dtype=np.float64).reshape(-1, hand_action.shape[-1])
        if min(ws.shape[0], wa.shape[0], hs.shape[0], ha.shape[0]) <= 0:
            return

        def metrics(a_ws, a_wa, a_hs, a_ha) -> tuple[float, float, float]:
            translation = float(np.linalg.norm(a_wa[:6] - a_ws[:6]))
            rotation = float(max(_rotation_delta_angle(a_wa[6:12], a_ws[6:12])[0], _rotation_delta_angle(a_wa[12:18], a_ws[12:18])[0]))
            fingertips = float(np.mean(np.linalg.norm(a_ha.reshape(10, 3) - a_hs.reshape(10, 3), axis=1)))
            return translation, rotation, fingertips

        checks = [metrics(ws[-1], wa[0], hs[-1], ha[0])]
        checks.extend(metrics(ws[i], ws[i + 1], hs[i], hs[i + 1]) for i in range(ws.shape[0] - 1))
        checks.extend(metrics(wa[i], wa[i + 1], ha[i], ha[i + 1]) for i in range(wa.shape[0] - 1))
        max_translation = max(item[0] for item in checks)
        max_rotation = max(item[1] for item in checks)
        max_fingertips = max(item[2] for item in checks)
        if (
            max_translation > self.delta_wrist_translation_threshold
            or max_rotation > self.delta_wrist_rotation_threshold
            or max_fingertips > self.delta_fingertips_displacement_threshold
        ):
            raise ExtremeStateActionDeltaError(
                f"state_action_delta_invalid: translation={max_translation:.4f}, rotation={max_rotation:.4f}, fingertips={max_fingertips:.4f}"
            )

    @staticmethod
    def _check_image(images: np.ndarray) -> None:
        arr = np.asarray(images)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ImageQualityError(f"unexpected image shape {arr.shape}")
        flat = arr.reshape(arr.shape[0], -1, 3)
        mean = flat.mean(axis=1)
        std = flat.std(axis=1)
        bad = (mean.min(axis=1) < 5.0) | (mean.max(axis=1) > 250.0) | (std.min(axis=1) < 3.0)
        if bad.any():
            raise ImageQualityError(f"image quality failed at frame {int(np.argmax(bad))}")

    @staticmethod
    def _check_depth(depth: np.ndarray | None, _clip_range) -> None:
        if depth is None:
            return
        arr = np.asarray(depth)
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            raise DepthQualityError("depth contains non-finite values")
        if float((arr > 0).mean()) < 0.05:
            raise DepthQualityError("depth valid fraction < 0.05")

    @classmethod
    def _check_post_normalize(cls, data: dict[str, Any]) -> None:
        for key in ("states", "actions"):
            arr = data.get(key)
            if arr is not None and np.asarray(arr).size and float(np.abs(arr).max()) > cls.POST_NORMALIZE_OUTLIER_THRESHOLD:
                raise OutlierDataError(f"post-normalize |{key}| too large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified dataset filtering/checking script for EgoVLA."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    wds_parser = subparsers.add_parser(
        "wds", help="Scan VLA/VLM WebDataset shards."
    )
    wds_parser.add_argument(
        "--kind",
        choices=("auto", "vla", "vlm"),
        default="auto",
        help="Sample schema to check. auto infers from tar member names.",
    )
    wds_parser.add_argument(
        "--shards",
        nargs="+",
        required=True,
        help="Shard paths or glob patterns.",
    )
    wds_parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Stop after this many samples across all shards. 0 scans all.",
    )
    wds_parser.add_argument(
        "--check-media",
        action="store_true",
        help="Decode image/depth payloads and run media checks.",
    )
    wds_parser.add_argument(
        "--check-depth",
        action="store_true",
        help="When --check-media is set, require/check depth.npy for VLA samples.",
    )
    wds_parser.add_argument(
        "--check-image-quality",
        action="store_true",
        help="Run brightness/contrast checks for VLM images too.",
    )
    wds_parser.add_argument(
        "--target-image-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=None,
        help="Optional resize size used before VLM image finite checks.",
    )
    wds_parser.add_argument(
        "--max-shard-fail-rate",
        type=float,
        default=0.0,
        help=(
            "Shard fail-rate threshold for --filtered-shards-output. "
            "Default keeps only shards with no failures."
        ),
    )
    add_common_outputs(wds_parser)
    wds_parser.add_argument(
        "--good-keys-output",
        default=None,
        help="Optional JSONL file with passing WDS sample keys.",
    )
    wds_parser.add_argument(
        "--bad-keys-output",
        default=None,
        help="Optional JSONL file with failing WDS sample keys and reasons.",
    )
    wds_parser.add_argument(
        "--filtered-shards-output",
        default=None,
        help="Optional text file listing shards whose fail rate is within threshold.",
    )

    zarr_parser = subparsers.add_parser(
        "zarr-list", help="Validate/filter zarr list entries."
    )
    zarr_parser.add_argument("--zarr-list", required=True, help="Input zarr list.")
    zarr_parser.add_argument(
        "--min-episodes",
        type=int,
        default=10,
        help="Minimum required episode count.",
    )
    zarr_parser.add_argument(
        "--filtered-output",
        default=None,
        help="Optional filtered zarr list output.",
    )
    add_common_outputs(zarr_parser)

    hf_parser = subparsers.add_parser(
        "hf-list", help="Validate/filter HF arrow/parquet dataset directories."
    )
    hf_parser.add_argument("--hf-list", required=True, help="Input HF list.")
    hf_parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="train",
        help="Split directory/directories to inspect.",
    )
    hf_parser.add_argument(
        "--filtered-output",
        default=None,
        help="Optional filtered HF list output.",
    )
    add_common_outputs(hf_parser)

    return parser.parse_args()


def add_common_outputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path. If omitted, report is printed to stdout.",
    )


def expand_paths(patterns: Iterable[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            out.extend(matches)
        elif not any(ch in pattern for ch in "*?["):
            out.append(pattern)
    return out


def write_report(report: dict[str, Any], path: str | None) -> None:
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def json_load_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, dict):
        return value
    raise MissingOrInvalidFilesError("missing or invalid meta.json")


def npy_load_maybe(value: Any) -> np.ndarray:
    if isinstance(value, bytes):
        return np.load(io.BytesIO(value))
    if isinstance(value, np.ndarray):
        return value
    raise MissingOrInvalidFilesError("missing or invalid npy payload")


def decode_image_bytes(value: Any) -> np.ndarray:
    from PIL import Image

    if value is None:
        raise MissingOrInvalidFilesError("missing image payload")
    if isinstance(value, bytes):
        return np.array(Image.open(io.BytesIO(value)).convert("RGB"), dtype=np.uint8)
    return np.array(value, dtype=np.uint8)


def resize_images_if_needed(images: np.ndarray, target_size: tuple[int, int] | None) -> np.ndarray:
    if target_size is None:
        return images
    target_h, target_w = target_size
    if images.shape[1] == target_h and images.shape[2] == target_w:
        return images
    from PIL import Image

    resized = []
    for frame in images:
        pil = Image.fromarray(frame)
        pil = pil.resize((target_w, target_h), Image.BILINEAR)
        resized.append(np.array(pil, dtype=np.uint8))
    return np.stack(resized, axis=0)


def infer_wds_kind(sample: dict[str, Any]) -> str:
    if "lowdim.npy" in sample:
        return "vla"
    if any(key.startswith("image_") and key.endswith(".jpg") for key in sample):
        return "vlm"
    raise MissingOrInvalidFilesError("cannot infer WDS sample kind")


def sample_locator(sample: dict[str, Any], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or {}
    return {
        "key": sample.get("__key__", ""),
        "shard": sample.get("__url__", ""),
        "dataset_name": meta.get("dataset_name", meta.get("source", "")),
        "episode_index": meta.get("episode_index", meta.get("sample_idx", "")),
    }


def lowdim_field(lowdim: np.ndarray, slices: dict[str, tuple[int, int]], key: str) -> np.ndarray:
    start, end = slices[key]
    if lowdim.ndim != 1 or lowdim.shape[0] < end:
        raise MissingOrInvalidFilesError(
            f"lowdim shape={tuple(lowdim.shape)} too short for {key}[{start}:{end}]"
        )
    return lowdim[start:end].astype(np.float32, copy=False)


def check_vla_wds_sample(
    sample: dict[str, Any],
    checker: Any,
    *,
    check_media: bool,
    check_depth: bool,
) -> None:
    meta = json_load_maybe(sample.get("meta.json"))
    lowdim = npy_load_maybe(sample.get("lowdim.npy"))

    cameras = meta.get("cameras", ["head"])
    if build_lowdim_slices is None:
        raise RuntimeError("training checker modules were not loaded")
    slices = build_lowdim_slices(cameras)
    wrist_state = lowdim_field(lowdim, slices, "wrist_state").reshape(1, -1)
    hand_state = lowdim_field(lowdim, slices, "hand_state").reshape(1, -1)
    wrist_action = lowdim_field(lowdim, slices, "wrist_action").reshape(1, -1)
    hand_action = lowdim_field(lowdim, slices, "hand_action").reshape(1, -1)
    extrinsic = lowdim_field(lowdim, slices, "head_extrinsic").reshape(4, 4)
    intrinsic = lowdim_field(lowdim, slices, "head_intrinsic")

    field_sample = {
        "wrist_state": wrist_state,
        "hand_state": hand_state,
        "wrist_action": wrist_action,
        "hand_action": hand_action,
        "extrinsic": extrinsic.reshape(-1),
        "intrinsic": intrinsic,
        "instruction": meta.get("instruction"),
        "instruction_num": meta.get("instruction_num"),
    }
    checker.check(
        sample_schema=(
            field_sample,
            {
                "required_keys": (
                    "wrist_state",
                    "hand_state",
                    "wrist_action",
                    "hand_action",
                    "extrinsic",
                    "intrinsic",
                    "instruction",
                    "instruction_num",
                ),
                "expected_last_dim": {
                    "wrist_state": 18,
                    "hand_state": 30,
                    "wrist_action": 18,
                    "hand_action": 30,
                    "extrinsic": 16,
                    "intrinsic": 4,
                },
            },
        )
    )
    checker.check(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        instruction=(meta.get("instruction"), meta.get("instruction_num")),
    )
    checker.check(
        finite={
            "wrist_state": wrist_state,
            "hand_state": hand_state,
            "wrist_action": wrist_action,
            "hand_action": hand_action,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        },
        rot6d={"wrist_state": wrist_state, "wrist_action": wrist_action},
        state_action_delta=(wrist_state, hand_state, wrist_action, hand_action),
    )

    if not check_media:
        return

    image = decode_image_bytes(sample.get("image.jpg"))
    checker.check(image=image[None], finite={"image": image})

    if check_depth:
        depth = npy_load_maybe(sample.get("depth.npy"))
        checker.check(depth=(depth, None), finite={"depth": depth})
    elif sample.get("depth.npy") is not None:
        depth = npy_load_maybe(sample.get("depth.npy"))
        checker.check(finite={"depth": depth})


def choose_vlm_text(meta: dict[str, Any], weights: tuple[float, float, float]) -> dict[str, Any]:
    texts = meta.get("texts")
    if not isinstance(texts, list) or not texts:
        raise MissingOrInvalidFilesError("meta.texts must be a non-empty list")

    rating_fields = (
        "formatting_ratings",
        "visual_dependency_ratings",
        "relevance_ratings",
    )
    ratings = []
    for field in rating_fields:
        value = meta.get(field)
        if not isinstance(value, list) or len(value) != len(texts):
            raise MissingOrInvalidFilesError(
                f"meta.{field} must be a list aligned with texts"
            )
        ratings.append(np.array([0 if item is None else item for item in value], dtype=np.float32))

    if len(texts) == 1:
        return texts[0]
    scores = ratings[0] * weights[0] + ratings[1] * weights[1] + ratings[2] * weights[2]
    return texts[int(np.argmax(scores))]


def check_vlm_wds_sample(
    sample: dict[str, Any],
    checker: Any,
    *,
    check_media: bool,
    check_image_quality: bool,
    target_image_size: tuple[int, int] | None,
) -> None:
    meta = json_load_maybe(sample.get("meta.json"))
    checker.check(
        sample_schema=(
            {"meta.json": meta},
            {
                "required_keys": ("meta.json",),
                "required_meta_keys": (
                    "texts",
                    "formatting_ratings",
                    "visual_dependency_ratings",
                    "relevance_ratings",
                ),
            },
        )
    )

    image_keys = sorted(
        key for key in sample if key.startswith("image_") and key.endswith(".jpg")
    )
    if not image_keys:
        raise MissingOrInvalidFilesError("missing required image_*.jpg fields")

    text = choose_vlm_text(meta, weights=(0.5, 0.5, 0.5))
    question = str(text.get("user", ""))
    answer = str(text.get("assistant", ""))
    checker.check(instruction=(question, 1))
    if not answer.strip():
        raise MissingOrInvalidFilesError("selected assistant answer is empty")

    if not check_media:
        return

    images = np.stack([decode_image_bytes(sample[key]) for key in image_keys], axis=0)
    processed = resize_images_if_needed(images, target_image_size)
    checker.check(finite={"images_processed": processed})
    if check_image_quality:
        checker.check(image=processed)


def scan_wds(args: argparse.Namespace) -> dict[str, Any]:
    import webdataset as wds

    load_training_checkers()
    if DataChecker is None:
        raise RuntimeError("failed to load training data checker")

    shard_paths = expand_paths(args.shards)
    if not shard_paths:
        raise SystemExit("No shards matched --shards")

    checker = DataChecker()
    target_size = tuple(args.target_image_size) if args.target_image_size else None
    started = time.time()
    total = 0
    passed = 0
    failed = 0
    reason_counts: Counter[str] = Counter()
    shard_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"samples": 0, "passed": 0, "failed": 0, "reasons": Counter()}
    )

    good_file = open(args.good_keys_output, "w", encoding="utf-8") if args.good_keys_output else None
    bad_file = open(args.bad_keys_output, "w", encoding="utf-8") if args.bad_keys_output else None
    try:
        for shard_path in shard_paths:
            if args.max_samples and total >= args.max_samples:
                break
            dataset = wds.WebDataset(str(shard_path), shardshuffle=False, empty_check=False)
            for sample in dataset:
                if args.max_samples and total >= args.max_samples:
                    break

                total += 1
                checker.note_sample_seen()
                stat = shard_stats[str(shard_path)]
                stat["samples"] += 1
                meta: dict[str, Any] | None = None
                try:
                    meta = json_load_maybe(sample.get("meta.json"))
                    kind = args.kind if args.kind != "auto" else infer_wds_kind(sample)
                    if kind == "vla":
                        check_vla_wds_sample(
                            sample,
                            checker,
                            check_media=args.check_media,
                            check_depth=args.check_depth,
                        )
                    elif kind == "vlm":
                        check_vlm_wds_sample(
                            sample,
                            checker,
                            check_media=args.check_media,
                            check_image_quality=args.check_image_quality,
                            target_image_size=target_size,
                        )
                    else:
                        raise MissingOrInvalidFilesError(f"unknown kind: {kind}")
                except DataSkipError as exc:
                    failed += 1
                    stat["failed"] += 1
                    reason = type(exc).__name__
                    reason_counts[reason] += 1
                    stat["reasons"][reason] += 1
                    if bad_file is not None:
                        record = sample_locator(sample, meta)
                        record.update({"reason": reason, "message": str(exc)})
                        bad_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as exc:
                    failed += 1
                    stat["failed"] += 1
                    reason = f"Unexpected{type(exc).__name__}"
                    reason_counts[reason] += 1
                    stat["reasons"][reason] += 1
                    if bad_file is not None:
                        record = sample_locator(sample, meta)
                        record.update({"reason": reason, "message": str(exc)})
                        bad_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    passed += 1
                    stat["passed"] += 1
                    if good_file is not None:
                        good_file.write(
                            json.dumps(sample_locator(sample, meta), ensure_ascii=False) + "\n"
                        )
    finally:
        if good_file is not None:
            good_file.close()
        if bad_file is not None:
            bad_file.close()

    shard_reports = []
    kept_shards = []
    for shard_path in shard_paths:
        stat = shard_stats[str(shard_path)]
        samples = int(stat["samples"])
        failures = int(stat["failed"])
        fail_rate = failures / samples if samples else 0.0
        if samples and fail_rate <= args.max_shard_fail_rate:
            kept_shards.append(str(shard_path))
        shard_reports.append(
            {
                "shard": str(shard_path),
                "samples": samples,
                "passed": int(stat["passed"]),
                "failed": failures,
                "fail_rate": fail_rate,
                "reasons": dict(stat["reasons"]),
            }
        )

    if args.filtered_shards_output:
        Path(args.filtered_shards_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.filtered_shards_output).write_text(
            "\n".join(kept_shards) + ("\n" if kept_shards else ""),
            encoding="utf-8",
        )

    return {
        "command": "wds",
        "kind": args.kind,
        "elapsed_sec": time.time() - started,
        "shards_total": len(shard_paths),
        "samples_total": total,
        "passed": passed,
        "failed": failed,
        "fail_rate": failed / total if total else 0.0,
        "reason_counts": dict(reason_counts),
        "filtered_shards": {
            "threshold": args.max_shard_fail_rate,
            "kept": len(kept_shards),
            "output": args.filtered_shards_output,
        },
        "shards": shard_reports,
    }


def parse_zarr_list_line(raw: str, line_no: int) -> tuple[str, str, str | None]:
    parts = raw.split()
    if len(parts) == 1:
        return parts[0], "human", None
    if len(parts) == 2:
        zarr_path, mapping_type = parts
        if mapping_type not in VALID_MAPPING_TYPES:
            raise ValueError("mapping_type must be 'human' or 'real_world'")
        return zarr_path, mapping_type, None
    if len(parts) == 3:
        zarr_path, mapping_type, target_name = parts
        if mapping_type not in VALID_MAPPING_TYPES:
            raise ValueError("mapping_type must be 'human' or 'real_world'")
        if "/" in target_name or "\\" in target_name:
            raise ValueError("target name must be a simple directory name")
        return zarr_path, mapping_type, target_name
    raise ValueError(
        f"line {line_no}: expected '<zarr_path> [human|real_world] [target_name]'"
    )


def count_zarr_episodes(zarr_path: str) -> int:
    import zarr

    try:
        root = zarr.open(zarr_path, mode="r")
        return int(len(root["meta/episode_ends"]))
    except Exception:
        try:
            root = zarr.open_consolidated(zarr_path, mode="r")
            return int(len(root["meta/episode_ends"]))
        except Exception as exc:
            raise ValueError(f"failed to read meta/episode_ends: {exc}") from exc


def scan_zarr_list(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.zarr_list)
    if not input_path.exists():
        raise SystemExit(f"zarr list not found: {input_path}")

    seen_names: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    kept_lines: list[str] = []
    reason_counts: Counter[str] = Counter()

    for line_no, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        record: dict[str, Any] = {"line_no": line_no, "raw": stripped}
        try:
            zarr_path, mapping_type, target_name = parse_zarr_list_line(stripped, line_no)
            path = Path(zarr_path)
            if not path.exists():
                raise ValueError(f"zarr path does not exist: {zarr_path}")
            dataset_name = path.stem
            if dataset_name in seen_names:
                raise ValueError(
                    f"duplicate dataset name '{dataset_name}' first seen on line {seen_names[dataset_name]}"
                )
            seen_names[dataset_name] = line_no
            if target_name is None:
                target_name = dataset_name
            episodes = count_zarr_episodes(zarr_path)
            keep = episodes >= args.min_episodes
            reason = "ok" if keep else "too_few_episodes"
            if keep:
                kept_lines.append(f"{zarr_path} {mapping_type} {target_name}")
        except Exception as exc:
            episodes = None
            keep = False
            reason = type(exc).__name__
            record["message"] = str(exc)

        reason_counts[reason] += 1
        record.update(
            {
                "episodes": episodes,
                "keep": keep,
                "reason": reason,
            }
        )
        records.append(record)

    if args.filtered_output:
        Path(args.filtered_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.filtered_output).write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""),
            encoding="utf-8",
        )

    return {
        "command": "zarr-list",
        "input": str(input_path),
        "min_episodes": args.min_episodes,
        "entries_total": len(records),
        "kept": sum(1 for item in records if item["keep"]),
        "dropped": sum(1 for item in records if not item["keep"]),
        "reason_counts": dict(reason_counts),
        "filtered_output": args.filtered_output,
        "entries": records,
    }


def load_hf_list(path: Path) -> list[tuple[str, str | None, str]]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        dataset_path = parts[0]
        dataset_name = parts[1] if len(parts) > 1 else None
        entries.append((dataset_path, dataset_name, stripped))
    return entries


def collect_hf_files(dataset_path: str, split: str) -> tuple[list[str], list[str]]:
    files = []
    empty = []
    for ext in ("arrow", "parquet"):
        for file_path in sorted(glob.glob(f"{dataset_path}/{split}/*.{ext}")):
            if Path(file_path).stat().st_size > 0:
                files.append(file_path)
            else:
                empty.append(file_path)
    return files, empty


def scan_hf_list(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.hf_list)
    if not input_path.exists():
        raise SystemExit(f"HF list not found: {input_path}")

    splits = ["train", "test"] if args.split == "both" else [args.split]
    records = []
    kept_lines = []
    reason_counts: Counter[str] = Counter()

    for dataset_path, dataset_name, raw in load_hf_list(input_path):
        split_info = {}
        total_files = 0
        total_empty = 0
        for split in splits:
            valid_files, empty_files = collect_hf_files(dataset_path, split)
            split_info[split] = {
                "valid_files": len(valid_files),
                "empty_files": len(empty_files),
            }
            total_files += len(valid_files)
            total_empty += len(empty_files)

        exists = Path(dataset_path).exists()
        keep = exists and total_files > 0
        if not exists:
            reason = "path_missing"
        elif total_files == 0:
            reason = "no_nonempty_files"
        elif total_empty > 0:
            reason = "ok_with_empty_files_skipped"
        else:
            reason = "ok"
        reason_counts[reason] += 1
        if keep:
            kept_lines.append(raw)
        records.append(
            {
                "dataset_path": dataset_path,
                "dataset_name": dataset_name or Path(dataset_path).name,
                "keep": keep,
                "reason": reason,
                "splits": split_info,
            }
        )

    if args.filtered_output:
        Path(args.filtered_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.filtered_output).write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""),
            encoding="utf-8",
        )

    return {
        "command": "hf-list",
        "input": str(input_path),
        "split": args.split,
        "entries_total": len(records),
        "kept": sum(1 for item in records if item["keep"]),
        "dropped": sum(1 for item in records if not item["keep"]),
        "reason_counts": dict(reason_counts),
        "filtered_output": args.filtered_output,
        "entries": records,
    }


def main() -> None:
    args = parse_args()
    if args.command == "wds":
        report = scan_wds(args)
    elif args.command == "zarr-list":
        report = scan_zarr_list(args)
    elif args.command == "hf-list":
        report = scan_hf_list(args)
    else:
        raise AssertionError(args.command)
    write_report(report, args.report)


if __name__ == "__main__":
    main()
