"""
Part 0: Per-Dataset Audit.

Iterate every VLA / VLM sub-dataset, verify sample schema, value health,
distribution, and (optionally) collator round-trip.  Generate per-dataset
visualizations for human / agent review.

Runs on: CPU
Requires: webdataset + (optional) normalizer.pkl + (optional) model-path

Usage:
    # Auto-discover from a data root directory
    #   expects: {data_root}/vla/{dataset_name}/*.tar
    #            {data_root}/vlm/{dataset_name}/*.tar
    python -m src.tests.full_chain_verification.part0_dataset_audit \
        --data-root ~/data \
        [--normalizer-path /path/to/normalizer.pkl] \
        [--model-path /path/to/Qwen3-VL-2B-Instruct] \
        [--max-samples 200]

    # Auto-discover from config YAMLs (uses paths in vla_wds.yaml / vlm_wds.yaml)
    python -m src.tests.full_chain_verification.part0_dataset_audit \
        --auto

    # Manual dataset list
    python -m src.tests.full_chain_verification.part0_dataset_audit \
        --vla-dataset egodex '/data/egodex/shard-{00000..00001}.tar' \
        --vlm-dataset FineVision '/data/vlm/FineVision/shard-{00000..00001}.tar'
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset.wds_dataset import LOWDIM_SLICES
from src.tests.full_chain_verification.utils import (
    PhaseReport,
    assert_check,
    get_output_dir,
    safe_import_plt,
)

OUTPUT_PART = "part0"

# -- Lowdim field names and slices ------------------------------------------

FIELD_NAMES = list(LOWDIM_SLICES.keys())
FIELD_SLICES = list(LOWDIM_SLICES.values())


# ── helpers ────────────────────────────────────────────────────────────────

def load_normalizer(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def iter_wds_raw(shard_url: str, max_samples: int, decode: str = "l"):
    """Yield raw WDS samples. decode="l" for numpy, "pil" for PIL images."""
    import webdataset as wds
    # WebDataset doesn't understand shell glob '*'; expand to file list first
    if any(c in shard_url for c in ("*", "?", "[")):
        urls = expand_shard_pattern(shard_url)
        if not urls:
            raise FileNotFoundError(f"No files matched pattern: {shard_url}")
        shard_url = urls
    dataset = wds.WebDataset(shard_url).decode(decode)
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        yield sample


def expand_shard_pattern(pattern: str) -> list[str]:
    """Expand a glob/brace pattern into actual file paths."""
    import braceexpand
    expanded = []
    for p in braceexpand.braceexpand(pattern):
        expanded.extend(sorted(glob.glob(p)))
    return expanded


def discover_from_data_root(data_root: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Scan {data_root}/vla/*/ and {data_root}/vlm/*/ for datasets.

    Returns (vla_list, vlm_list) where each entry is (name, shard_pattern).
    Shard pattern is a glob string matching all .tar files in the sub-directory.
    """
    root = Path(data_root).expanduser().resolve()
    vla_list, vlm_list = [], []

    vla_dir = root / "vla"
    if vla_dir.is_dir():
        for ds_dir in sorted(vla_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            tars = sorted(ds_dir.glob("*.tar"))
            if not tars:
                print(f"  WARN: {ds_dir.name} has no .tar files, skipping")
                continue
            # Build a brace-expansion pattern for webdataset
            if len(tars) == 1:
                pattern = str(tars[0])
            else:
                pattern = str(ds_dir / "*.tar")
            vla_list.append((ds_dir.name, pattern))

    vlm_dir = root / "vlm"
    if vlm_dir.is_dir():
        for ds_dir in sorted(vlm_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            tars = sorted(ds_dir.glob("*.tar"))
            if not tars:
                print(f"  WARN: {ds_dir.name} has no .tar files, skipping")
                continue
            if len(tars) == 1:
                pattern = str(tars[0])
            else:
                pattern = str(ds_dir / "*.tar")
            vlm_list.append((ds_dir.name, pattern))

    return vla_list, vlm_list


def auto_discover_datasets() -> tuple[list[dict], list[dict]]:
    """Read VLA and VLM dataset configs and return (vla_list, vlm_list).

    Each entry: {"name": str, "shard_urls": str | list[str], "type": "vla"|"vlm"}
    """
    from omegaconf import OmegaConf

    project_root = Path(__file__).resolve().parents[3]
    vla_path = project_root / "src/config/dataset_paths/vla_wds.yaml"
    vlm_path = project_root / "src/config/dataset_paths/vlm_wds.yaml"

    vla_list, vlm_list = [], []

    if vla_path.exists():
        raw = OmegaConf.load(vla_path)
        base = OmegaConf.to_container(raw, resolve=False)
        wds_base = base.get("wds_base_dir", "")
        for ds in base.get("vla_wds_datasets", []):
            urls = ds["shard_urls"]
            if isinstance(urls, str):
                urls = urls.replace("${wds_base_dir}", wds_base)
            elif isinstance(urls, list):
                urls = [u.replace("${wds_base_dir}", wds_base) for u in urls]
            vla_list.append({"name": ds["name"], "shard_urls": urls, "type": "vla"})

    if vlm_path.exists():
        raw = OmegaConf.load(vlm_path)
        base = OmegaConf.to_container(raw, resolve=False)
        wds_base_vlm = base.get("wds_base_dir", "")
        for ds in base.get("vlm_wds_datasets", []):
            urls = ds["shard_urls"]
            if isinstance(urls, str):
                urls = urls.replace("${wds_base_dir}", wds_base_vlm)
            elif isinstance(urls, list):
                urls = [u.replace("${wds_base_dir}", wds_base_vlm) for u in urls]
            vlm_list.append({"name": ds["name"], "shard_urls": urls, "type": "vlm"})

    return vla_list, vlm_list


# ── VLA dataset checks ────────────────────────────────────────────────────

def audit_vla_dataset(
    name: str,
    shard_url: str,
    max_samples: int,
    normalizer=None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all checks on a single VLA dataset. Returns audit result dict."""
    result = {
        "name": name,
        "type": "vla",
        "n_loaded": 0,
        "checks": {},
        "field_stats": {},
        "issues": [],
    }

    # Load samples
    samples = []
    try:
        for s in iter_wds_raw(shard_url, max_samples):
            samples.append(s)
    except Exception as e:
        result["issues"].append(f"shard_load_error: {e}")
        return result

    result["n_loaded"] = len(samples)
    if not samples:
        result["issues"].append("no samples loaded")
        return result

    # -- Schema check --
    required_keys = {"lowdim.npy", "__key__"}
    missing_keys = []
    no_image = 0
    for i, s in enumerate(samples):
        for k in required_keys:
            if k not in s:
                missing_keys.append(f"sample {i}: missing {k}")
        if not any(k.endswith((".png", ".jpg", ".jpeg")) for k in s):
            no_image += 1
    result["checks"]["schema"] = {
        "missing_keys": missing_keys[:10],
        "no_image_count": no_image,
        "pass": len(missing_keys) == 0,
    }

    # -- Lowdim shape & NaN/Inf --
    lowdims = []
    shape_issues = []
    nan_count, inf_count, allzero_count = 0, 0, 0
    for i, s in enumerate(samples):
        ld = s.get("lowdim.npy")
        if ld is None:
            continue
        if ld.shape != (116,):
            shape_issues.append(f"sample {i}: shape={ld.shape}")
            continue
        lowdims.append(ld)
        if np.any(np.isnan(ld)):
            nan_count += 1
        if np.any(np.isinf(ld)):
            inf_count += 1
        if np.all(ld == 0):
            allzero_count += 1

    result["checks"]["lowdim_shape"] = {
        "issues": shape_issues[:10],
        "pass": len(shape_issues) == 0,
    }
    result["checks"]["nan_inf"] = {
        "nan_samples": nan_count,
        "inf_samples": inf_count,
        "allzero_samples": allzero_count,
        "pass": nan_count == 0 and inf_count == 0,
    }

    if not lowdims:
        result["issues"].append("no valid lowdim arrays")
        return result

    lowdims = np.stack(lowdims)  # [N, 116]

    # -- Per-field statistics --
    for field_name, (start, end) in LOWDIM_SLICES.items():
        vals = lowdims[:, start:end]
        stats = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "nan_dims": int(np.any(np.isnan(vals), axis=0).sum()),
            "zero_dims": int(np.all(vals == 0, axis=0).sum()),
        }
        result["field_stats"][field_name] = stats

    # -- Action/state specific checks --
    wrist_actions = lowdims[:, LOWDIM_SLICES["wrist_action"][0]:LOWDIM_SLICES["wrist_action"][1]]
    hand_actions = lowdims[:, LOWDIM_SLICES["hand_action"][0]:LOWDIM_SLICES["hand_action"][1]]
    all_actions = np.concatenate([wrist_actions, hand_actions], axis=1)
    action_range = float(np.max(np.abs(all_actions)))

    result["checks"]["action_range"] = {
        "max_abs": action_range,
        "warn": action_range > 100,
        "pass": action_range < 1e6,
    }

    # -- Normalizer check --
    # When normalizer uses "actions" key it was fitted on *relative* actions
    # (after coordinate transform + get_relative_action).  Raw absolute actions
    # from the shard have a fundamentally different distribution, so the
    # outlier check is only indicative — mark it info-only rather than fail.
    if normalizer is not None:
        try:
            import torch
            if "actions" in normalizer.params_dict:
                key = "actions"
            elif "motions" in normalizer.params_dict:
                key = "motions"
            else:
                key = list(normalizer.params_dict.keys())[0]
            is_relative = key == "actions"
            actions_tensor = torch.tensor(all_actions, dtype=torch.float32)
            normalized = normalizer[key](actions_tensor).numpy()
            outlier_ratio = float(np.mean(np.abs(normalized) > 5))
            result["checks"]["normalizer"] = {
                "key_used": key,
                "is_relative_normalizer": is_relative,
                "outlier_ratio_gt5": outlier_ratio,
                "normalized_range": [float(normalized.min()), float(normalized.max())],
                # Relative normalizer applied to raw absolute actions: always pass
                # (mismatch is expected, only used for visualization reference).
                # Unified ("motions") normalizer applied directly: enforce threshold.
                "pass": True if is_relative else outlier_ratio < 0.1,
            }
        except Exception as e:
            result["checks"]["normalizer"] = {"error": str(e), "pass": False}

    # -- Visualizations --
    if out_dir is not None:
        ds_dir = out_dir / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        plot_vla_distributions(lowdims, name, ds_dir)
        plot_vla_image_grid(samples[:16], name, ds_dir)
        plot_vla_nan_heatmap(lowdims, name, ds_dir)
        if normalizer is not None and "normalizer" in result["checks"] and "error" not in result["checks"]["normalizer"]:
            import torch
            key = result["checks"]["normalizer"]["key_used"]
            actions_tensor = torch.tensor(all_actions, dtype=torch.float32)
            normalized = normalizer[key](actions_tensor).numpy()
            plot_vla_normalized_dist(normalized, name, ds_dir)

    return result


# ── VLM dataset checks ────────────────────────────────────────────────────

def audit_vlm_dataset(
    name: str,
    shard_url: str,
    max_samples: int,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all checks on a single VLM dataset."""
    result = {
        "name": name,
        "type": "vlm",
        "n_loaded": 0,
        "checks": {},
        "issues": [],
    }

    samples = []
    try:
        for s in iter_wds_raw(shard_url, max_samples, decode="pil"):
            samples.append(s)
    except Exception as e:
        result["issues"].append(f"shard_load_error: {e}")
        return result

    result["n_loaded"] = len(samples)
    if not samples:
        result["issues"].append("no samples loaded")
        return result

    # -- Schema check --
    no_image, no_meta, decode_err = 0, 0, 0
    text_lengths = []
    for i, s in enumerate(samples):
        has_image = any(k.endswith((".jpg", ".png")) and k.startswith("image_") for k in s)
        has_meta = "meta.json" in s
        if not has_image:
            no_image += 1
        if not has_meta:
            no_meta += 1
            continue

        meta = s["meta.json"]
        if not isinstance(meta, dict):
            decode_err += 1
            continue

        texts = meta.get("texts", [])
        if texts:
            for t in texts:
                if isinstance(t, dict):
                    q_len = len(str(t.get("user", "")))
                    a_len = len(str(t.get("assistant", "")))
                    text_lengths.append({"question": q_len, "answer": a_len})

    result["checks"]["schema"] = {
        "no_image": no_image,
        "no_meta": no_meta,
        "decode_err": decode_err,
        "pass": no_image == 0 and no_meta == 0 and decode_err == 0,
    }

    if text_lengths:
        q_lens = [t["question"] for t in text_lengths]
        a_lens = [t["answer"] for t in text_lengths]
        result["checks"]["text_stats"] = {
            "n_qa_pairs": len(text_lengths),
            "q_len_mean": float(np.mean(q_lens)),
            "q_len_max": int(np.max(q_lens)),
            "a_len_mean": float(np.mean(a_lens)),
            "a_len_max": int(np.max(a_lens)),
            "empty_answer": int(sum(1 for a in a_lens if a == 0)),
            "pass": sum(1 for a in a_lens if a == 0) == 0,
        }

    # -- Visualizations --
    if out_dir is not None:
        ds_dir = out_dir / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        plot_vlm_image_grid(samples[:16], name, ds_dir)
        if text_lengths:
            plot_vlm_text_length_hist(text_lengths, name, ds_dir)

    return result


# ── Visualization functions ───────────────────────────────────────────────

def plot_vla_distributions(lowdims: np.ndarray, name: str, ds_dir: Path) -> None:
    """Per-field boxplot + per-dim histograms for lowdim values."""
    plt = safe_import_plt()
    if plt is None:
        return

    # Per-field boxplot
    fig, axes = plt.subplots(1, len(FIELD_NAMES), figsize=(3 * len(FIELD_NAMES), 5))
    if len(FIELD_NAMES) == 1:
        axes = [axes]
    for ax, field_name in zip(axes, FIELD_NAMES):
        start, end = LOWDIM_SLICES[field_name]
        vals = lowdims[:, start:end].flatten()
        ax.boxplot(vals, vert=True)
        ax.set_title(field_name, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{name}: per-field distributions (N={len(lowdims)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(ds_dir / "field_boxplots.png", dpi=120)
    plt.close(fig)

    # Per-dim histograms for actions (48D)
    action_start = LOWDIM_SLICES["wrist_action"][0]
    action_end = LOWDIM_SLICES["hand_action"][1]
    action_data = lowdims[:, action_start:action_end]
    n_dims = action_data.shape[1]
    n_cols = 8
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2))
    axes = axes.flatten()
    for d in range(n_dims):
        ax = axes[d]
        vals = action_data[:, d]
        ax.hist(vals, bins=50, alpha=0.7, edgecolor="none")
        ax.set_title(f"act[{d}]", fontsize=7)
        ax.tick_params(labelsize=6)
    for d in range(n_dims, len(axes)):
        axes[d].set_visible(False)
    fig.suptitle(f"{name}: action dimensions histogram (N={len(lowdims)})", fontsize=11)
    fig.tight_layout()
    fig.savefig(ds_dir / "action_dim_hist.png", dpi=120)
    plt.close(fig)


def plot_vla_normalized_dist(normalized: np.ndarray, name: str, ds_dir: Path) -> None:
    """Histogram of normalized action values."""
    plt = safe_import_plt()
    if plt is None:
        return

    n_dims = normalized.shape[1]
    n_cols = 8
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2))
    axes = axes.flatten()
    for d in range(n_dims):
        ax = axes[d]
        ax.hist(normalized[:, d], bins=50, alpha=0.7, color="steelblue", edgecolor="none")
        ax.axvline(x=-5, color="red", linestyle="--", linewidth=0.5)
        ax.axvline(x=5, color="red", linestyle="--", linewidth=0.5)
        ax.set_title(f"norm[{d}]", fontsize=7)
        ax.tick_params(labelsize=6)
    for d in range(n_dims, len(axes)):
        axes[d].set_visible(False)
    fig.suptitle(f"{name}: normalized action distribution", fontsize=11)
    fig.tight_layout()
    fig.savefig(ds_dir / "normalized_action_hist.png", dpi=120)
    plt.close(fig)


def plot_vla_nan_heatmap(lowdims: np.ndarray, name: str, ds_dir: Path) -> None:
    """Heatmap of NaN / zero / extreme values across samples x dims."""
    plt = safe_import_plt()
    if plt is None:
        return

    N = min(len(lowdims), 200)
    data = lowdims[:N]
    # Build indicator: 0=normal, 1=zero, 2=extreme(|x|>100), 3=NaN
    indicator = np.zeros_like(data, dtype=np.float32)
    indicator[data == 0] = 1
    indicator[np.abs(data) > 100] = 2
    indicator[np.isnan(data)] = 3

    fig, ax = plt.subplots(figsize=(14, max(4, N * 0.04)))
    im = ax.imshow(indicator, aspect="auto", interpolation="nearest",
                   cmap="RdYlGn_r", vmin=0, vmax=3)
    ax.set_xlabel("dimension (0-115)")
    ax.set_ylabel("sample index")
    ax.set_title(f"{name}: value health (green=ok, yellow=zero, orange=extreme, red=NaN)")
    # Field boundaries
    for field_name, (start, _) in LOWDIM_SLICES.items():
        ax.axvline(x=start - 0.5, color="white", linewidth=0.5, alpha=0.6)
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], label="0=ok 1=zero 2=|x|>100 3=NaN")
    fig.tight_layout()
    fig.savefig(ds_dir / "nan_heatmap.png", dpi=120)
    plt.close(fig)


def plot_vla_image_grid(samples: list[dict], name: str, ds_dir: Path) -> None:
    """Grid of sample images."""
    plt = safe_import_plt()
    if plt is None:
        return

    images = []
    for s in samples:
        for key in s:
            if key.endswith((".png", ".jpg", ".jpeg")):
                img = s[key]
                if hasattr(img, "convert"):  # PIL Image
                    img = np.array(img.convert("RGB"))
                if isinstance(img, np.ndarray):
                    images.append(img)
                break
        if len(images) >= 16:
            break

    if not images:
        return

    n = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < n:
            img = images[i]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(f"#{i}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{name}: sample images", fontsize=11)
    fig.tight_layout()
    fig.savefig(ds_dir / "sample_images.png", dpi=120)
    plt.close(fig)


def plot_vlm_image_grid(samples: list[dict], name: str, ds_dir: Path) -> None:
    """Grid of VLM sample images with Q/A text."""
    plt = safe_import_plt()
    if plt is None:
        return

    entries = []
    for s in samples:
        img_keys = sorted(k for k in s if k.startswith("image_") and k.endswith((".jpg", ".png")))
        if not img_keys:
            continue
        img = s[img_keys[0]]
        meta = s.get("meta.json", {})
        texts = meta.get("texts", [{}])
        text = texts[0] if texts else {}
        q = str(text.get("user", ""))[:60]
        a = str(text.get("assistant", ""))[:60]
        entries.append({"image": img, "q": q, "a": a})
        if len(entries) >= 12:
            break

    if not entries:
        return

    n = len(entries)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < n:
            e = entries[i]
            img = e["image"]
            if hasattr(img, "convert"):
                img = np.array(img.convert("RGB"))
            ax.imshow(img)
            ax.set_title(f"Q: {e['q']}", fontsize=6, wrap=True)
            ax.set_xlabel(f"A: {e['a']}", fontsize=6, wrap=True)
        ax.axis("off")
    fig.suptitle(f"{name}: VLM sample images", fontsize=11)
    fig.tight_layout()
    fig.savefig(ds_dir / "vlm_sample_images.png", dpi=120)
    plt.close(fig)


def plot_vlm_text_length_hist(text_lengths: list[dict], name: str, ds_dir: Path) -> None:
    """Histogram of question and answer text lengths."""
    plt = safe_import_plt()
    if plt is None:
        return

    q_lens = [t["question"] for t in text_lengths]
    a_lens = [t["answer"] for t in text_lengths]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(q_lens, bins=30, alpha=0.7, color="steelblue", edgecolor="none")
    ax1.set_title("Question length (chars)")
    ax1.set_xlabel("length")
    ax2.hist(a_lens, bins=30, alpha=0.7, color="coral", edgecolor="none")
    ax2.set_title("Answer length (chars)")
    ax2.set_xlabel("length")
    fig.suptitle(f"{name}: text length distribution (N={len(text_lengths)})", fontsize=11)
    fig.tight_layout()
    fig.savefig(ds_dir / "text_length_hist.png", dpi=120)
    plt.close(fig)


# ── Collator round-trip ───────────────────────────────────────────────────

def test_collator_roundtrip(
    name: str,
    shard_url: str,
    model_path: str,
    max_samples: int = 5,
) -> dict[str, Any]:
    """Load real samples, push through collator, check no errors."""
    from src.dataset.wds_dataset import LOWDIM_SLICES
    from src.tests.full_chain_verification.part2_collator_tokenization import build_collator
    import torch

    collator, _, _ = build_collator(model_path)
    result = {"pass": True, "errors": []}

    try:
        samples = list(iter_wds_raw(shard_url, max_samples))
    except Exception as e:
        return {"pass": False, "errors": [f"load: {e}"]}

    for i, raw in enumerate(samples):
        try:
            ld = raw["lowdim.npy"]
            fields = {k: ld[s:e] for k, (s, e) in LOWDIM_SLICES.items()}
            # Build a sample dict compatible with collator
            img = next((raw[k] for k in raw if k.endswith((".png", ".jpg", ".jpeg"))), None)
            if img is None:
                continue
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            sample = {
                "images": torch.tensor(img, dtype=torch.uint8).unsqueeze(0),
                "instruction": "test instruction",
                "intrinsic": torch.tensor(fields["intrinsic"], dtype=torch.float32),
                "active_views": ["head"],
                "view_mask": torch.tensor([True, False], dtype=torch.bool),
                "vision_type": "video",
                "video_fps": torch.tensor(5.0),
                "states": torch.tensor(fields["wrist_state"], dtype=torch.float32).unsqueeze(0),
                "actions": torch.tensor(
                    np.concatenate([fields["wrist_action"], fields["hand_action"]]),
                    dtype=torch.float32,
                ).unsqueeze(0),
                "n_states": torch.tensor(1, dtype=torch.long),
                "n_actions": torch.tensor(1, dtype=torch.long),
                "actions_valid_mask": torch.ones(1, 48, dtype=torch.bool),
                "is_vla_data": torch.tensor(True, dtype=torch.bool),
            }
            batch = collator.collate_raw([sample])
            assert "input_ids" in batch
        except Exception as e:
            result["errors"].append(f"sample {i}: {e}")

    if result["errors"]:
        result["pass"] = False
    return result


# ── Report generation ─────────────────────────────────────────────────────

def format_summary_line(r: dict) -> str:
    """One-line summary for a dataset audit result."""
    status = "PASS" if all(
        c.get("pass", True) for c in r.get("checks", {}).values()
    ) and not r.get("issues") else "FAIL"
    n = r["n_loaded"]
    issues = r.get("issues", [])
    check_fails = [k for k, v in r.get("checks", {}).items() if not v.get("pass", True)]
    detail = ""
    if issues:
        detail = f" issues={issues[:3]}"
    if check_fails:
        detail += f" failed_checks={check_fails}"
    return f"[{status}] {r['name']} ({r['type']}, N={n}){detail}"


# ── Main ──────────────────────────────────────────────────────────────────

def run_all(
    vla_datasets: list[tuple[str, str]] | None = None,
    vlm_datasets: list[tuple[str, str]] | None = None,
    data_root: str | None = None,
    auto: bool = False,
    normalizer_path: str | None = None,
    model_path: str | None = None,
    max_samples: int = 200,
) -> list[dict]:
    out_dir = get_output_dir(OUTPUT_PART)
    print(f"\n=== Part 0: Per-Dataset Audit ===\n")
    print(f"Output: {out_dir}\n")

    normalizer = load_normalizer(normalizer_path) if normalizer_path else None

    # Build dataset list
    all_vla: list[tuple[str, str]] = list(vla_datasets or [])
    all_vlm: list[tuple[str, str]] = list(vlm_datasets or [])

    if data_root:
        root_vla, root_vlm = discover_from_data_root(data_root)
        all_vla.extend(root_vla)
        all_vlm.extend(root_vlm)
        print(f"  Discovered from {data_root}: {len(root_vla)} VLA, {len(root_vlm)} VLM datasets\n")

    if auto:
        auto_vla, auto_vlm = auto_discover_datasets()
        for ds in auto_vla:
            urls = ds["shard_urls"]
            if isinstance(urls, list):
                urls = " ".join(urls)
            all_vla.append((ds["name"], urls))
        for ds in auto_vlm:
            urls = ds["shard_urls"]
            if isinstance(urls, list):
                urls = " ".join(urls)
            all_vlm.append((ds["name"], urls))

    results = []

    # VLA datasets
    for name, shard_url in all_vla:
        print(f"  Auditing VLA: {name} ...")
        try:
            r = audit_vla_dataset(name, shard_url, max_samples, normalizer, out_dir)
            if model_path:
                r["collator_test"] = test_collator_roundtrip(name, shard_url, model_path, max_samples=3)
            results.append(r)
            print(f"    {format_summary_line(r)}")
        except Exception as e:
            print(f"    [ERROR] {name}: {e}")
            traceback.print_exc()
            results.append({"name": name, "type": "vla", "n_loaded": 0, "issues": [str(e)], "checks": {}})

    # VLM datasets
    for name, shard_url in all_vlm:
        print(f"  Auditing VLM: {name} ...")
        try:
            r = audit_vlm_dataset(name, shard_url, max_samples, out_dir)
            results.append(r)
            print(f"    {format_summary_line(r)}")
        except Exception as e:
            print(f"    [ERROR] {name}: {e}")
            traceback.print_exc()
            results.append({"name": name, "type": "vlm", "n_loaded": 0, "issues": [str(e)], "checks": {}})

    # Save JSON report
    report_path = out_dir / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 60}")
    print("  DATASET AUDIT SUMMARY")
    print(f"{'=' * 60}")
    total_pass = 0
    for r in results:
        line = format_summary_line(r)
        print(f"  {line}")
        if "PASS" in line:
            total_pass += 1
    print(f"\n  Total: {total_pass}/{len(results)} passed")
    print(f"  Report: {report_path}")
    print(f"  Plots:  {out_dir}/{{dataset_name}}/")
    print(f"{'=' * 60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 0: Per-dataset audit")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root dir with vla/ and vlm/ sub-dirs (e.g. ~/data)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover datasets from config YAMLs")
    parser.add_argument("--vla-dataset", nargs=2, action="append", default=[],
                        metavar=("NAME", "SHARD_URL"),
                        help="Manual VLA dataset: name shard_pattern (repeatable)")
    parser.add_argument("--vlm-dataset", nargs=2, action="append", default=[],
                        metavar=("NAME", "SHARD_URL"),
                        help="Manual VLM dataset: name shard_pattern (repeatable)")
    parser.add_argument("--normalizer-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Qwen3-VL path for collator round-trip test")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    results = run_all(
        vla_datasets=[tuple(d) for d in args.vla_dataset],
        vlm_datasets=[tuple(d) for d in args.vlm_dataset],
        data_root=args.data_root,
        auto=args.auto,
        normalizer_path=args.normalizer_path,
        model_path=args.model_path,
        max_samples=args.max_samples,
    )
    all_pass = all(
        all(c.get("pass", True) for c in r.get("checks", {}).values()) and not r.get("issues")
        for r in results
    )
    exit(0 if all_pass else 1)
