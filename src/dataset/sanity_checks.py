"""Skip-error classes, check helpers, and the DataChecker entry point that
ties them to per-worker skip stats and a stdout logger."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Mapping

import numpy as np
import torch


class DataSkipError(ValueError):
    """Base class for intentionally skip-on-fail sanity violations.

    Stage wrappers catch DataSkipError as a labeled skip; any other
    exception propagates with stage + sample context attached so
    unexpected upstream failures (corrupt tar, schema bugs, decode
    errors) surface loudly instead of being silently dropped.
    """


class NonFiniteDataError(DataSkipError):
    """Raised when dataset preprocessing produces a non-finite numeric value."""


class OutlierDataError(DataSkipError):
    """Raised when dataset produces a finite but extreme value (post-normalize outlier)."""


class ExtrinsicInvalidError(DataSkipError):
    """Raised when a 4x4 homogeneous extrinsic matrix fails structural validity."""


class IntrinsicInvalidError(DataSkipError):
    """Raised when [fx, fy, cx, cy] intrinsic fails structural validity."""


class InstructionInvalidError(DataSkipError):
    """Raised when the language instruction is None / empty / a placeholder, or
    when instruction_num is 0 (which would make np.random.randint(0, 0) crash)."""


class ImageQualityError(DataSkipError):
    """Raised when an RGB image frame fails brightness/contrast sanity checks."""


class DepthQualityError(DataSkipError):
    """Raised when a depth map fails finite/valid-fraction sanity checks."""


def check_extrinsic_valid(
    ext: np.ndarray,
    *,
    tol_det: float = 0.1,
    tol_last_row: float = 1e-3,
) -> None:
    """Structural sanity check for a 4x4 homogeneous transform.

    Does NOT bound the translation norm (world coordinate frame is arbitrary;
    chunk-local translation is bounded by check_chunk_translation_valid instead).
    """
    if ext.shape != (4, 4):
        raise ExtrinsicInvalidError(f"shape={ext.shape} != (4, 4)")
    if not np.all(np.isfinite(ext)):
        raise ExtrinsicInvalidError("contains non-finite values")
    last_row = ext[3, :]
    if not np.allclose(last_row, [0.0, 0.0, 0.0, 1.0], atol=tol_last_row):
        raise ExtrinsicInvalidError(
            f"last row {last_row.tolist()} deviates from [0, 0, 0, 1] (tol={tol_last_row})"
        )
    det = float(np.linalg.det(ext[:3, :3]))
    if abs(det - 1.0) > tol_det:
        raise ExtrinsicInvalidError(
            f"det(R)={det:.4g} deviates from 1.0 (tol={tol_det})"
        )


def check_intrinsic_valid(
    intr: np.ndarray,
    *,
    max_focal: float = 10000.0,
) -> None:
    """Structural sanity check for a [fx, fy, cx, cy] intrinsic.

    Does NOT bound cx/cy — data augmentation (random_resized_crop) shifts the
    principal point, and a tight bound here causes false positives.
    """
    if intr.shape != (4,):
        raise IntrinsicInvalidError(f"shape={intr.shape} != (4,)")
    if not np.all(np.isfinite(intr)):
        raise IntrinsicInvalidError("contains non-finite values")
    fx, fy = float(intr[0]), float(intr[1])
    if fx <= 0 or fy <= 0:
        raise IntrinsicInvalidError(f"non-positive focal: fx={fx:.4g}, fy={fy:.4g}")
    if fx >= max_focal or fy >= max_focal:
        raise IntrinsicInvalidError(
            f"focal too large: fx={fx:.4g}, fy={fy:.4g} (max={max_focal})"
        )


_INSTRUCTION_PLACEHOLDERS = frozenset({"none", "unknown", "n/a", "null", "todo"})


def check_instruction_valid(
    instruction,
    instruction_num,
) -> None:
    """Instruction content sanity.

    Accepts very short instructions (e.g. "pick") so long as they are non-empty
    and not a known placeholder. instruction_num must be > 0 so downstream
    np.random.randint(0, instruction_num) does not crash.
    """
    if instruction is None:
        raise InstructionInvalidError("instruction is None")
    n = int(instruction_num) if instruction_num is not None else 0
    if n <= 0:
        raise InstructionInvalidError(f"instruction_num={n} <= 0")

    def _bad_string(s) -> bool:
        if not isinstance(s, str):
            return True
        stripped = s.strip()
        if not stripped:
            return True
        if stripped.lower() in _INSTRUCTION_PLACEHOLDERS:
            return True
        return False

    if isinstance(instruction, (list, tuple)):
        if all(_bad_string(x) for x in instruction):
            raise InstructionInvalidError(
                f"all {len(instruction)} candidate instructions are empty or placeholders"
            )
    else:
        if _bad_string(instruction):
            raise InstructionInvalidError(f"instruction is empty or placeholder: {instruction!r}")


def check_image_quality(
    images: np.ndarray,
    *,
    min_mean: float = 5.0,
    max_mean: float = 250.0,
    min_std: float = 3.0,
) -> None:
    """Robust per-frame brightness/contrast sanity for uint8 RGB video.

    Drops all-black / all-white frames via mean bounds, near-uniform frames via
    std floor. Does NOT run frozen-frame detection: history_pad_mode='repeat'
    and future_frame_pad_mode='repeat' legitimately duplicate frames.

    images: [N, H, W, 3] uint8.
    """
    if images is None:
        return
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ImageQualityError(f"unexpected shape {images.shape}, expected [N, H, W, 3]")
    # Per-channel per-frame stats; keep min/max across channels so a dark
    # or blown-out channel is still caught.
    flat = images.reshape(images.shape[0], -1, 3)
    per_frame_mean = flat.mean(axis=1)  # [N, 3]
    per_frame_std = flat.std(axis=1)    # [N, 3]
    m_min = per_frame_mean.min(axis=1)  # [N]
    m_max = per_frame_mean.max(axis=1)  # [N]
    s_min = per_frame_std.min(axis=1)   # [N]

    mean_bad = (m_min < min_mean) | (m_max > max_mean)
    std_bad = s_min < min_std
    any_bad = mean_bad | std_bad
    if not any_bad.any():
        return

    # Frame-major precedence preserved: report the earliest failing frame,
    # and prefer the mean violation when both fire on the same frame.
    n = int(np.argmax(any_bad))
    if mean_bad[n]:
        raise ImageQualityError(
            f"frame[{n}] channel mean=[{float(m_min[n]):.2f}, {float(m_max[n]):.2f}] "
            f"outside [{min_mean}, {max_mean}]"
        )
    raise ImageQualityError(
        f"frame[{n}] channel std_min={float(s_min[n]):.2f} < {min_std}"
    )


def check_depth_quality(
    depth: np.ndarray | None,
    clip_range,
    *,
    min_valid_fraction: float = 0.05,
) -> None:
    """Finite + minimum valid-coverage sanity for a depth map.

    Does NOT enforce non-negative (sensor noise allows slight negatives), max
    range (the downstream depth_clip_range clamps it), or median bounds
    (sparse depth sensors produce a wide distribution).
    """
    if depth is None:
        return
    if not np.all(np.isfinite(depth)):
        raise DepthQualityError("contains non-finite values")
    valid = (depth > 0).mean()
    if valid < min_valid_fraction:
        raise DepthQualityError(
            f"valid_fraction={valid:.3f} < {min_valid_fraction}"
        )




def build_sample_context(sample: Mapping[str, Any]) -> str:
    """Build a compact, human-readable sample identifier string."""
    fields: list[str] = []

    dataset_name = sample.get("dataset_name", sample.get("source", "unknown"))
    dataset_name = _to_python_scalar(dataset_name)
    if dataset_name is not None:
        fields.append(f"dataset={dataset_name}")

    episode_index = sample.get("episode_index", sample.get("sample_idx"))
    episode_index = _to_python_scalar(episode_index)
    if episode_index is not None:
        fields.append(f"episode={episode_index}")

    sample_key = sample.get("__key__")
    sample_key = _to_python_scalar(sample_key)
    if sample_key is not None:
        fields.append(f"key={sample_key}")

    # shard: webdataset tar path (basename only for brevity); propagated
    # through sliding_window_compose -> build_sample_from_window.
    shard_url = sample.get("shard_url")
    if shard_url is None:
        shard_url = sample.get("__url__")
    shard_url = _to_python_scalar(shard_url)
    if shard_url is not None:
        fields.append(f"shard={os.path.basename(str(shard_url))}")

    return ", ".join(fields) if fields else "dataset=unknown"


def ensure_mapping_finite(values: Mapping[str, Any]) -> None:
    """Validate that all numeric arrays or tensors in a mapping are finite.

    Per-sample shard / episode / key context is added later by
    DataChecker.log_skip when the exception bubbles to preprocess_fn, so the
    raised message here only carries the field-level diagnostic.
    """
    for name, value in values.items():
        if value is None:
            continue
        summary = summarize_non_finite(name=name, value=value)
        if summary is not None:
            raise NonFiniteDataError(f"Non-finite value: {summary}")


def summarize_non_finite(name: str, value: Any) -> str | None:
    """Return a short diagnostic string when a numeric value is non-finite."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or value.dtype == torch.bool:
            return None
        if not torch.is_floating_point(value) and not torch.is_complex(value):
            return None

        finite_mask = torch.isfinite(value)
        if bool(finite_mask.all()):
            return None

        nan_count = int(torch.isnan(value).sum().item())
        inf_count = int(torch.isinf(value).sum().item())
        finite_values = value[finite_mask]
        return _format_summary(
            name=name,
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            nan_count=nan_count,
            inf_count=inf_count,
            finite_values=finite_values,
        )

    if isinstance(value, np.ndarray):
        if value.size == 0 or value.dtype == np.bool_:
            return None
        if not np.issubdtype(value.dtype, np.floating) and not np.issubdtype(value.dtype, np.complexfloating):
            return None

        finite_mask = np.isfinite(value)
        if bool(finite_mask.all()):
            return None

        nan_count = int(np.isnan(value).sum())
        inf_count = int(np.isinf(value).sum())
        finite_values = value[finite_mask]
        return _format_summary(
            name=name,
            shape=value.shape,
            dtype=str(value.dtype),
            nan_count=nan_count,
            inf_count=inf_count,
            finite_values=finite_values,
        )

    return None


def _format_summary(
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    nan_count: int,
    inf_count: int,
    finite_values: Any,
) -> str:
    finite_count = _count_values(finite_values)
    if finite_count == 0:
        return (
            f"field={name}, shape={shape}, dtype={dtype}, "
            f"nan_count={nan_count}, inf_count={inf_count}, finite_stats=none"
        )

    if isinstance(finite_values, torch.Tensor):
        finite_values = finite_values.detach().float()
        finite_min = float(finite_values.min().item())
        finite_max = float(finite_values.max().item())
        finite_mean = float(finite_values.mean().item())
    else:
        finite_values = finite_values.astype(np.float32, copy=False)
        finite_min = float(finite_values.min())
        finite_max = float(finite_values.max())
        finite_mean = float(finite_values.mean())

    return (
        f"field={name}, shape={shape}, dtype={dtype}, "
        f"nan_count={nan_count}, inf_count={inf_count}, "
        f"finite_min={finite_min:.6e}, finite_max={finite_max:.6e}, finite_mean={finite_mean:.6e}"
    )


def _to_python_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return value.reshape(-1)[0].item()
    return value


def _count_values(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, np.ndarray):
        return int(value.size)
    return 0


LOGGER_NAME = "dataset"
_ENV_LEVEL = "DATASET_LOG_LEVEL"
_CONFIGURED_PIDS: set[int] = set()


def configure_logger() -> None:
    """Idempotent per-pid setup of the data-skip logger.

    Safe to call from worker_init_fn on spawn; no-op on fork once the parent
    has run it.
    """
    pid = os.getpid()
    if pid in _CONFIGURED_PIDS:
        return

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(os.environ.get(_ENV_LEVEL, "INFO").upper())

    has_stdout_handler = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        for h in logger.handlers
    )
    if not has_stdout_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [pid=%(process)d] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    # FSDP / torchrun / transformers root handlers would duplicate records.
    logger.propagate = False

    _CONFIGURED_PIDS.add(pid)


def get_data_logger() -> logging.Logger:
    configure_logger()
    return logging.getLogger(LOGGER_NAME)


class DataChecker:
    """Per-Dataset data-quality dispatcher holding skip counters and logger.

    ``check(**fields)`` runs all checks applicable to the supplied fields;
    callers stay decoupled from the specific checks performed.
    """

    # Normal post-normalize values sit in roughly [-1, 1] (q01/q99 scaling);
    # 10x margin admits legitimate edge cases but rejects pinv-blowup outputs.
    POST_NORMALIZE_OUTLIER_THRESHOLD = 10.0

    _DISPATCH = {
        "intrinsic": "_apply_intrinsic",
        "extrinsic": "_apply_extrinsic",
        "instruction": "_apply_instruction",
        "image": "_apply_image",
        "depth": "_apply_depth",
        "finite": "_apply_finite",
        "post_normalize": "_apply_post_normalize",
    }

    def __init__(self) -> None:
        self.seen = 0
        self.skipped = 0
        self._log = get_data_logger()

    def note_sample_seen(self) -> None:
        self.seen += 1

    def note_sample_skipped(self) -> tuple[int, int, float]:
        self.skipped += 1
        ratio = 100.0 * self.skipped / self.seen if self.seen > 0 else 0.0
        return self.skipped, self.seen, ratio

    def current_stats(self) -> tuple[int, int]:
        return self.skipped, self.seen

    def log_skip(self, worker_id: int, exc: BaseException, sample) -> None:
        try:
            ctx = build_sample_context(sample)
        except Exception:
            ctx = "dataset=?, episode=?, key=?, shard=?"
        skipped, seen, pct = self.note_sample_skipped()
        self._log.info(
            "DATA_SKIP worker=%s reason=%s %s msg=%s stats=skipped=%d/seen=%d (%.2f%%)",
            worker_id, type(exc).__name__, ctx, exc, skipped, seen, pct,
        )

    def check(self, **fields) -> None:
        """Run all checks applicable to the given fields, raising on first failure.

        Recognized field keys (any subset; ``None`` values skipped):
            intrinsic   : np.ndarray of shape (4,) [fx, fy, cx, cy]
            extrinsic   : np.ndarray (4, 4) for one matrix, or (K, 4, 4) for a
                          batch (e.g. future_extrinsic per frame)
            instruction : tuple (instruction, instruction_num)
            image       : np.ndarray [N, H, W, 3] uint8
            depth       : tuple (depth_array_or_none, clip_range)
        """
        for key, val in fields.items():
            if val is None:
                continue
            method_name = self._DISPATCH.get(key)
            if method_name is None:
                raise KeyError(f"DataChecker: unknown field '{key}'")
            getattr(self, method_name)(val)

    @staticmethod
    def _apply_intrinsic(intr) -> None:
        check_intrinsic_valid(intr)

    @staticmethod
    def _apply_extrinsic(ext) -> None:
        if ext.ndim == 3:
            for ext_k in ext:
                check_extrinsic_valid(ext_k)
        else:
            check_extrinsic_valid(ext)

    @staticmethod
    def _apply_instruction(instruction_tuple) -> None:
        instruction, instruction_num = instruction_tuple
        check_instruction_valid(instruction, instruction_num)

    @staticmethod
    def _apply_image(images) -> None:
        check_image_quality(images)

    @staticmethod
    def _apply_depth(depth_tuple) -> None:
        depth, clip_range = depth_tuple
        check_depth_quality(depth, clip_range)

    @staticmethod
    def _apply_finite(mapping) -> None:
        ensure_mapping_finite(mapping)

    @classmethod
    def _apply_post_normalize(cls, data) -> None:
        for field in ("states", "actions"):
            arr = data.get(field)
            if arr is None or arr.size == 0:
                continue
            amax = float(np.abs(arr).max())
            if amax > cls.POST_NORMALIZE_OUTLIER_THRESHOLD:
                raise OutlierDataError(
                    f"post-normalize |{field}|={amax:.3e} > {cls.POST_NORMALIZE_OUTLIER_THRESHOLD}"
                )


def current_worker_id() -> int:
    """Return the DataLoader worker id, or -1 in the main process."""
    info = torch.utils.data.get_worker_info()
    return -1 if info is None else int(info.id)


def attach_sample_ctx(fn, *, checker: DataChecker):
    """Wrap ``fn(sample) -> sample`` into a generator usable as a wds
    ``pipeline.compose(...)`` stage with two error policies:

    - ``DataSkipError``: routed through ``checker.log_skip`` and the
      sample is dropped silently. This is the existing per-sample skip
      path used by preprocess sanity checks.
    - any other exception: re-raised as ``RuntimeError`` with the sample
      locator (dataset / episode / key / shard) attached, with the
      original exception preserved as ``__cause__`` so the traceback
      still pinpoints the failing line.

    The wrapper exists so the call site only needs to opt in once
    (``checker=...`` on the pipeline builder); per-stage error glue
    stays inside the builder.
    """
    def stage_iter(src):
        for sample in src:
            try:
                result = fn(sample)
            except DataSkipError as e:
                checker.log_skip(current_worker_id(), e, sample)
            except Exception as e:
                raise RuntimeError(
                    f"data error on {build_sample_context(sample)}"
                ) from e
            else:
                yield result
    return stage_iter
