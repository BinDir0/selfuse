"""
Post-collator data visualization for LegendVLA.

For each sample in a real batch from UnifiedWdsDataset (after the full
collator + Qwen3-VL processor pass), this script:

  1. De-patchifies pixel_values_videos (and the breast half when present)
     back to viewable (T, H, W, 3) uint8 frames. These are the exact pixels
     the ViT consumes — any cropping / resize / mean-std normalization done
     by the processor is reversed here only for display.
  2. Un-normalizes states / actions with the training normalizer, reproduces
     absolute wrist / fingertip positions in the head-camera frame (same
     geometry helpers used by rerun_inference_vis.py and debug_dataloader_batch),
     projects them with the head intrinsic and overlays markers on the last
     head frame. Breast view gets the same overlay using breast_intrinsic.
  3. Dumps instruction text and the rendered chat template for text sanity.

Output: one self-contained HTML per preset under --output-dir/<preset>/index.html,
with all PNG figures embedded as base64. `scp` the file to a local machine and
open in any browser; no asset directory to ship along.

# ─────────────────────────── Basic usage ────────────────────────────────────
# Always run from the project root so `python -m scripts.viz_loaded_data` can
# import `src.*`. The config loader composes
# `src/config/experiment/legendvla_qwen3_vl.yaml` by default; dataset /
# normalizer / shape paths inside that config must be reachable.
#
# Minimal head-only run (single-GPU, one DataLoader worker):
#     python -m scripts.viz_loaded_data \
#         --preset head-only \
#         --num-samples 6 \
#         --num-workers 0 \
#         --shuffle-buffer 16
#
# Dual-view run (sources only from shard subsets that actually ship a breast
# camera; the preset sets load_breast=True and narrows
# vla_wds_datasets to those subsets automatically):
#     python -m scripts.viz_loaded_data \
#         --preset with-breast \
#         --num-samples 6 \
#         --num-workers 0 \
#         --shuffle-buffer 16
#
# Override normalizer location (default: cfg.training.normalizer_path):
#     python -m scripts.viz_loaded_data \
#         --preset head-only \
#         --normalizer-path /efs-exp/.../normalizer.pkl
#
# Output produced (per run):
#     outputs/viz/head_only/index.html     # ~10 MB, self-contained
#     outputs/viz/with_breast/index.html   # ~10-20 MB, self-contained
# To view: `scp <remote>:outputs/viz/<preset>/index.html .` then open locally.
#
# Notes:
# * --shuffle-buffer 16 keeps startup at a few seconds. The production default
#   is 16384, which fills for minutes before the first batch is yielded.
# * --num-workers 0 keeps everything in-process for easy debugging; bump to 2-4
#   once the pipeline is confirmed working end-to-end.
# * with-breast currently requires shards that store `breast_image.jpg`; the
#   preset filters to dataset names listed inline (see load_config), edit that
#   allow-list if new dual-view shards land elsewhere.
# ────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Reuse state/action projection helpers from the eval-style debug script.
from src.tests.debug_dataloader_batch import (
    build_combined_overlay,
    build_fingertip_geometry,
    draw_sequence_overlay,
    project_points,
    select_normalizer_fields,
)
from src.dataset.data_transforms import get_absolute_action


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["head-only", "with-breast"],
        default="head-only",
        help="head-only: load_breast=False (default recipe). "
             "with-breast: force load_breast=True and filter to teleop_xiaozi.",
    )
    parser.add_argument(
        "--config-path",
        default="src/config/experiment/legendvla_qwen3_vl.yaml",
        help="Experiment config YAML (compose path relative to src/config).",
    )
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument(
        "--normalizer-path",
        default=None,
        help="Override training.normalizer_path. Default: whatever the config holds.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/viz",
        help="Directory to place the HTML and supporting PNGs.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=16,
        help="Override vla_dataset.shuffle_buffer for faster startup.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hydra config assembly
# ---------------------------------------------------------------------------

def load_config(args: argparse.Namespace):
    """Compose the Hydra experiment config, then patch for viz.

    Keeps all real dataset paths / normalizer / shape_meta intact. Only the
    shuffle buffer and per-subset breast toggles change.
    """
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)

    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    with hydra.initialize(version_base=None, config_path="../src/config"):
        cfg = hydra.compose(config_name=f"experiment/{config_name}")

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = {"runtime": {"output_dir": "outputs"}, "job": {"num": 0, "name": "viz"}}

    # Register the `hydra:` interpolation resolver AFTER hydra.initialize (hydra
    # itself registers one at import that calls HydraConfig.get(), which raises
    # outside of @hydra.main). replace=True swaps to a fallback that returns ""
    # when HydraConfig is unset, matching src/tests/debug_dataloader_batch.py.
    def hydra_resolver(path: str):
        from hydra.core.hydra_config import HydraConfig
        try:
            value = OmegaConf.select(HydraConfig.get(), path)
        except Exception:
            return ""
        return "" if value is None else value
    OmegaConf.register_new_resolver("hydra", hydra_resolver, replace=True)

    OmegaConf.resolve(cfg)

    # Speed-up: small shuffle buffer so first batch comes out in seconds.
    cfg.dataset.vla_dataset.shuffle_buffer = int(args.shuffle_buffer)
    if cfg.dataset.get("vlm_dataset") is not None:
        cfg.dataset.vlm_dataset.shuffle_buffer = int(args.shuffle_buffer)

    if args.preset == "with-breast":
        cfg.dataset.vla_dataset.load_breast = True
        # Filter VLA subsets to those that declare a breast camera so every
        # sampled batch exercises the dual-view branch. teleop_xiaozi is the
        # only production dataset with breast shards today.
        filtered = []
        for entry in cfg.vla_wds_datasets:
            if entry.name in {"teleop_xiaozi"}:
                filtered.append(entry)
        if not filtered:
            raise RuntimeError(
                "with-breast preset requires at least one dataset named 'teleop_xiaozi' "
                "in vla_wds_datasets; none found in the current config."
            )
        cfg.vla_wds_datasets = filtered
        # Disable VLM interleaving for a focused breast batch — we still want to
        # confirm the VLA path's breast handling, not VLM.
        cfg.dataset.vlm_dataset = None
    else:
        cfg.dataset.vla_dataset.load_breast = False

    # Single-process DataLoader settings are applied in build_dataloader.
    return cfg


# ---------------------------------------------------------------------------
# Dataset + dataloader
# ---------------------------------------------------------------------------

def build_dataloader(cfg, num_samples: int, num_workers: int, normalizer_override: str | None):
    """Instantiate the full UnifiedWdsDataset + collator, return a DataLoader."""
    data_collator = hydra.utils.instantiate(cfg.data_collator)
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataset.vla_dataset.set_collator(data_collator)
    if dataset.vlm_dataset is not None:
        dataset.vlm_dataset.set_collator(data_collator)

    normalizer_path = normalizer_override or cfg.training.normalizer_path
    normalizer = None
    if normalizer_path is not None:
        with open(normalizer_path, "rb") as fh:
            normalizer = pickle.load(fh)
    if normalizer is not None:
        dataset.vla_dataset.set_normalizer(normalizer)

    loader = DataLoader(
        dataset=dataset,
        collate_fn=dataset.get_collator(),
        batch_size=num_samples,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )
    return loader, normalizer


# ---------------------------------------------------------------------------
# Image de-patchify
# ---------------------------------------------------------------------------

# Qwen3-VL video processor config (matches preprocessor_config.json on disk).
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2


def depatchify_video(
    patches: torch.Tensor,
    grid_thw: torch.Tensor,
) -> np.ndarray:
    """Rebuild one video's (T, H, W, 3) uint8 tensor from flat Qwen3-VL patches.

    Qwen3-VL video processor flattens each video into per-patch feature rows:
        row = channels * temporal_patch_size * patch_size * patch_size
    grouped in (T_patch, H_patch, W_patch) order. We reverse that group, undo
    the mean-std normalization, and clip to uint8 for display.

    Reference:
        https://huggingface.co/docs/transformers/model_doc/qwen2_vl
        (same patch layout as Qwen2-VL; Qwen3-VL inherits the video processor)
    """
    assert grid_thw.shape == (3,), f"grid_thw must be (3,), got {tuple(grid_thw.shape)}"
    t_p, h_p, w_p = [int(x) for x in grid_thw.tolist()]
    expected_rows = t_p * h_p * w_p
    if patches.shape[0] != expected_rows:
        raise ValueError(
            f"patch count mismatch: patches={patches.shape[0]} vs T*H*W={expected_rows}"
        )
    feat_dim = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
    if patches.shape[1] != feat_dim:
        raise ValueError(
            f"patch feature dim {patches.shape[1]} != expected {feat_dim} "
            f"(3 channels * {TEMPORAL_PATCH_SIZE} tps * {PATCH_SIZE}x{PATCH_SIZE})"
        )

    video = patches.float().view(
        t_p, h_p, w_p, 3, TEMPORAL_PATCH_SIZE, PATCH_SIZE, PATCH_SIZE,
    )
    # (T_p, tps, H_p, P, W_p, P, C) then flatten T and spatial dims.
    video = video.permute(0, 4, 1, 5, 2, 6, 3).contiguous()
    video = video.view(t_p * TEMPORAL_PATCH_SIZE, h_p * PATCH_SIZE, w_p * PATCH_SIZE, 3)
    arr = video.cpu().numpy()
    # Inverse of (x - mean) / std, which Qwen2VL applies on [0, 1] inputs.
    arr = arr * IMAGE_STD[None, None, None, :] + IMAGE_MEAN[None, None, None, :]
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return arr


def split_per_sample_videos(
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.Tensor,
    is_vla_mask: np.ndarray,
    load_breast: bool,
):
    """Return per-sample dict with {"head": (T,H,W,3), "breast": ... | None}.

    Grid rows are laid out per sample: one row when breast is absent for that
    sample (VLM sample in token-padded mode still emits a video), two rows
    (head first, then breast) when load_breast=True and the sample has
    a breast shard. We use `is_vla_mask` + the existence of breast slots
    (inferred from total rows) to split.
    """
    # Each video's patch-row count is cumulative. build offsets.
    rows_per_video = (video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]).tolist()
    offsets = [0]
    for r in rows_per_video:
        offsets.append(offsets[-1] + int(r))

    num_videos = video_grid_thw.shape[0]
    num_samples = is_vla_mask.shape[0]
    videos_per_sample = num_videos // num_samples
    if num_videos % num_samples != 0:
        # Mixed case is unusual; fall back to "head only per sample" and let the
        # caller check shapes.
        videos_per_sample = 1

    per_sample = []
    for i in range(num_samples):
        start_v = i * videos_per_sample
        head_patches = pixel_values_videos[offsets[start_v]: offsets[start_v + 1]]
        head_video = depatchify_video(head_patches, video_grid_thw[start_v])
        entry = {"head": head_video}
        if load_breast and videos_per_sample >= 2:
            breast_patches = pixel_values_videos[
                offsets[start_v + 1]: offsets[start_v + 2]
            ]
            entry["breast"] = depatchify_video(breast_patches, video_grid_thw[start_v + 1])
        else:
            entry["breast"] = None
        per_sample.append(entry)
    return per_sample


# ---------------------------------------------------------------------------
# State / action decode
# ---------------------------------------------------------------------------

def decode_states_actions(
    batch: dict,
    sample_idx: int,
    normalizer,
    use_relative_action: bool,
) -> dict:
    """Un-normalize + return absolute state/action sequences for overlay.

    Mirrors the debug_dataloader_batch flow exactly so the state/action viz
    matches what rerun_inference_vis.py draws.
    """
    state_normalizer, action_normalizer = select_normalizer_fields(
        normalizer, use_relative_action,
    )

    n_states = int(batch["n_states"][sample_idx].item())
    n_actions = int(batch["n_actions"][sample_idx].item())

    states_norm = batch["states"][sample_idx, :n_states]
    actions_norm = batch["actions"][sample_idx, :n_actions]

    if state_normalizer is not None and n_states > 0:
        states_abs = state_normalizer.unnormalize(states_norm).detach().cpu().numpy()
    else:
        states_abs = states_norm.detach().cpu().numpy()

    if action_normalizer is not None and n_actions > 0:
        actions_raw = action_normalizer.unnormalize(actions_norm).detach().cpu().numpy()
    else:
        actions_raw = actions_norm.detach().cpu().numpy()

    if use_relative_action and n_states > 0 and n_actions > 0:
        actions_abs = get_absolute_action(states_abs[-1], actions_raw)
    else:
        actions_abs = actions_raw.copy()

    return {
        "n_states": n_states,
        "n_actions": n_actions,
        "states_abs": states_abs,
        "actions_abs": actions_abs,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def png_bytes(img: np.ndarray) -> bytes:
    """Encode a (H, W, 3) RGB uint8 array to PNG bytes."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # cv2.imencode expects BGR.
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode PNG failed")
    return buf.tobytes()


def img_tag(img: np.ndarray, max_width: int = 320) -> str:
    """Return a <img> tag with inline base64 data URI."""
    data = base64.b64encode(png_bytes(img)).decode("ascii")
    return (
        f'<img style="max-width:{max_width}px;margin:2px;border:1px solid #ccc" '
        f'src="data:image/png;base64,{data}" />'
    )


def frame_strip(video: np.ndarray, label: str, stride: int = 1) -> str:
    """Render a T-frame strip as a horizontal row of PNG <img> tags."""
    imgs = [img_tag(video[t]) for t in range(0, video.shape[0], stride)]
    return (
        f'<div class="strip"><div class="strip-label">{label} '
        f'({video.shape[0]}x{video.shape[1]}x{video.shape[2]})</div>'
        f'<div class="row">{"".join(imgs)}</div></div>'
    )


def sample_section(
    index: int,
    head_video: np.ndarray,
    breast_video: np.ndarray | None,
    state_overlay: np.ndarray,
    action_overlay: np.ndarray,
    combined_overlay: np.ndarray,
    breast_overlay: np.ndarray | None,
    meta: dict,
) -> str:
    parts = [f"<section><h2>Sample #{index}</h2>"]
    parts.append(f"<pre class='meta'>{meta['header']}</pre>")
    parts.append(frame_strip(head_video, "Head RGB (after processor, un-normalized)"))
    if breast_video is not None:
        parts.append(frame_strip(breast_video, "Breast RGB (after processor, un-normalized)"))
    parts.append("<div class='strip'><div class='strip-label'>State / action overlay on last head frame</div><div class='row'>")
    parts.append(img_tag(state_overlay))
    parts.append(img_tag(action_overlay))
    parts.append(img_tag(combined_overlay))
    parts.append("</div></div>")
    if breast_overlay is not None:
        parts.append(
            "<div class='strip'><div class='strip-label'>State+action combined overlay on last breast frame</div>"
            f"<div class='row'>{img_tag(breast_overlay)}</div></div>"
        )
    parts.append(f"<pre class='meta'>{meta['body']}</pre></section>")
    return "\n".join(parts)


def render_html(preset: str, sections: list[str], summary: str) -> str:
    """Wrap sections in a single-file HTML skeleton."""
    style = """
    body { font-family: -apple-system, sans-serif; margin: 16px; background: #fafafa; }
    h1 { margin-bottom: 4px; }
    h2 { margin-top: 24px; padding: 4px 6px; background: #eee; }
    section { border: 1px solid #ddd; padding: 8px 12px; margin: 12px 0; background: white; }
    .row { display: flex; flex-wrap: wrap; }
    .strip { margin: 8px 0; }
    .strip-label { font-weight: bold; color: #444; margin: 4px 0; font-size: 13px; }
    pre.meta { background: #f4f4f4; padding: 6px 8px; font-size: 12px; white-space: pre-wrap; word-break: break-all; }
    header { margin-bottom: 12px; }
    """
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>viz_loaded_data {preset}</title>
<style>{style}</style></head><body>
<header>
  <h1>LegendVLA post-collator visualization — {preset}</h1>
  <pre class="meta">{summary}</pre>
</header>
{"".join(sections)}
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir) / args.preset.replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args)
    loader, normalizer = build_dataloader(cfg, args.num_samples, args.num_workers, args.normalizer_path)

    print(f"[viz] fetching one batch of {args.num_samples} samples ...")
    batch = next(iter(loader))
    print(f"[viz] batch keys: {sorted(batch.keys())}")
    print(f"[viz] pixel_values_videos: {tuple(batch['pixel_values_videos'].shape) if batch.get('pixel_values_videos') is not None else None}")
    print(f"[viz] video_grid_thw: {batch.get('video_grid_thw')}")

    is_vla_mask = batch["is_vla_data"].cpu().numpy().astype(bool)
    use_relative_action = bool(getattr(cfg.dataset.vla_dataset, "use_relative_action", True))

    video_grid_thw = batch["video_grid_thw"]
    pixel_values_videos = batch["pixel_values_videos"]
    if pixel_values_videos is None or video_grid_thw is None:
        print("[viz] WARNING: pixel_values_videos is None (likely image-only batch); cannot render RGB frames.")
        return
    per_sample = split_per_sample_videos(
        pixel_values_videos, video_grid_thw, is_vla_mask,
        load_breast=(args.preset == "with-breast"),
    )

    sections = []
    for i in range(len(per_sample)):
        if not bool(is_vla_mask[i]):
            # VLM sample — no state/action overlay.
            head = per_sample[i]["head"]
            sections.append(
                f"<section><h2>Sample #{i} (VLM)</h2>"
                f"{frame_strip(head, 'Head RGB (VLM sample)')}</section>"
            )
            continue

        sa = decode_states_actions(batch, i, normalizer, use_relative_action)
        intrinsic = batch["intrinsic"][i].cpu().numpy().astype(np.float32)
        last_head_frame = per_sample[i]["head"][-1]
        presence = int(batch.get("presence", torch.full((len(per_sample),), 3))[i]) if "presence" in batch else 3

        state_overlay = draw_sequence_overlay(
            last_head_frame, sa["states_abs"], intrinsic,
            base_color=(64, 180, 255), presence=presence,
        )
        action_overlay = draw_sequence_overlay(
            last_head_frame, sa["actions_abs"], intrinsic,
            base_color=(255, 140, 64), presence=presence,
        )
        combined = build_combined_overlay(
            last_head_frame, sa["states_abs"], sa["actions_abs"], intrinsic, presence,
        )

        breast_video = per_sample[i].get("breast")
        breast_overlay = None
        if breast_video is not None and "breast_intrinsic" in batch:
            breast_intrinsic = batch["breast_intrinsic"][i].cpu().numpy().astype(np.float32)
            last_breast = breast_video[-1]
            breast_overlay = build_combined_overlay(
                last_breast, sa["states_abs"], sa["actions_abs"],
                breast_intrinsic, presence,
            )

        dataset_name = batch.get("dataset_name", ["?"] * len(per_sample))
        dataset_name_i = dataset_name[i] if isinstance(dataset_name, list) else (
            dataset_name[i] if hasattr(dataset_name, "__getitem__") else "?"
        )
        header = (
            f"dataset_name = {dataset_name_i}\n"
            f"is_vla_data  = {bool(is_vla_mask[i])}\n"
            f"n_states     = {sa['n_states']}\n"
            f"n_actions    = {sa['n_actions']}\n"
            f"head intrin  = fx={intrinsic[0]:.2f} fy={intrinsic[1]:.2f} cx={intrinsic[2]:.2f} cy={intrinsic[3]:.2f}\n"
            + (
                "breast intrin= "
                f"fx={batch['breast_intrinsic'][i, 0].item():.2f} "
                f"fy={batch['breast_intrinsic'][i, 1].item():.2f} "
                f"cx={batch['breast_intrinsic'][i, 2].item():.2f} "
                f"cy={batch['breast_intrinsic'][i, 3].item():.2f}\n"
                if breast_video is not None and "breast_intrinsic" in batch else ""
            )
        )
        body_parts = []
        if "debug_texts" in batch:
            body_parts.append("rendered prompt:\n" + str(batch["debug_texts"][i])[:1000])
        body_parts.append(
            f"states_abs[:3] (un-normalized, camera frame, 48-D) =\n{np.round(sa['states_abs'][:3], 3)}"
        )
        body_parts.append(
            f"actions_abs[:3] =\n{np.round(sa['actions_abs'][:3], 3)}"
        )
        meta = {"header": header, "body": "\n\n".join(body_parts)}

        sections.append(sample_section(
            i, per_sample[i]["head"], breast_video,
            state_overlay, action_overlay, combined, breast_overlay, meta,
        ))

    summary = (
        f"preset       = {args.preset}\n"
        f"config       = {args.config_path}\n"
        f"batch_size   = {args.num_samples}\n"
        f"PATCH_SIZE   = {PATCH_SIZE}  TEMPORAL_PATCH_SIZE = {TEMPORAL_PATCH_SIZE}\n"
        f"mean / std   = {IMAGE_MEAN.tolist()} / {IMAGE_STD.tolist()}\n"
        f"VLA / VLM    = {int(is_vla_mask.sum())} / {int((~is_vla_mask).sum())}\n"
    )
    html = render_html(args.preset, sections, summary)
    out_path = output_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[viz] wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    sys.exit(main())
