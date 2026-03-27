#!/usr/bin/env python3
"""Export one real dataloader batch for offline inspection.

This script runs the actual dataset -> collator -> tokenizer path used by the
project, then saves raw samples, processed samples, tokenized outputs, and
fingertip projections for each sample in one batch.
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import pickle
import sys
from datetime import datetime
from typing import Any

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.data_transforms import get_absolute_action
from src.utils.geometry import transform_hand_points_from_wrist_to_camera_frame

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "now", lambda fmt: datetime.now().strftime(fmt), replace=True
)


def _register_offline_hydra_resolver(mock_hydra: Any) -> None:
    """Bind ${hydra:...} to mock_hydra. Must run after hydra.compose — compose re-registers the default resolver."""

    def _hydra_resolver(key: str) -> Any:
        val = OmegaConf.select(mock_hydra, key)
        return "" if val is None else val

    OmegaConf.register_new_resolver("hydra", _hydra_resolver, replace=True)


def _offline_hydra_stub(experiment_stem: str, output_dir: str) -> Any:
    return OmegaConf.create(
        {
            "runtime": {
                "output_dir": output_dir,
                "choices": {"experiment": experiment_stem},
            },
            "job": {"num": 0, "name": experiment_stem},
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one real dataloader batch and export debug artifacts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/experiment/legendvla_qwen3_vl.yaml",
        help="Hydra experiment config path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory used to save the exported batch.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=4,
        help="Number of samples in the exported batch.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to inspect.",
    )
    parser.add_argument(
        "--dataset_kind",
        type=str,
        default="vla",
        choices=["vla", "unified"],
        help="Whether to inspect the pure VLA stream or the unified train stream.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker count for debugging.",
    )
    parser.add_argument(
        "--normalizer_path",
        type=str,
        default=None,
        help="Override cfg.training.normalizer_path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for sampling in debug mode.",
    )
    return parser.parse_args()


def load_config(
    config_path: pathlib.Path,
    *,
    hydra_output_dir: pathlib.Path | str | None = None,
):
    """Compose Hydra config without @hydra.main; stub ${hydra:...} for OmegaConf.resolve.

    hydra_output_dir:
        Value for ${hydra:runtime.output_dir} (e.g. script --output_dir). Defaults to <repo>/outputs.
    """
    config_path = config_path.resolve()
    experiment_stem = config_path.stem
    out = (
        pathlib.Path(hydra_output_dir).resolve()
        if hydra_output_dir is not None
        else (REPO_ROOT / "outputs")
    )
    raw_cfg = OmegaConf.load(config_path)
    if "defaults" not in raw_cfg:
        cfg = raw_cfg
    else:
        config_dir = config_path.parent.parent.resolve()
        config_name = f"{config_path.parent.name}/{config_path.stem}"
        with hydra.initialize_config_dir(
            config_dir=str(config_dir),
            version_base=None,
        ):
            cfg = hydra.compose(config_name=config_name)

    OmegaConf.set_struct(cfg, False)
    mock_hydra = _offline_hydra_stub(experiment_stem, str(out))
    _register_offline_hydra_resolver(mock_hydra)
    cfg.hydra = OmegaConf.to_container(mock_hydra)

    OmegaConf.resolve(cfg)
    return cfg


def tensor_to_numpy(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return value


def copy_for_save(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: copy_for_save(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_for_save(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_for_save(item) for item in value)
    return value


def summarize_value(value: Any):
    if isinstance(value, torch.Tensor):
        out: dict[str, Any] = {
            "type": "torch.Tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        # Scalars (e.g. n_states, n_actions, is_vla_data) have shape []; JSON looked "empty" without value.
        if value.numel() == 1:
            raw = value.detach().cpu().item()
            if isinstance(raw, bool):
                out["value"] = raw
            elif torch.is_floating_point(value):
                out["value"] = float(raw)
            else:
                out["value"] = int(raw)
        return out
    if isinstance(value, np.ndarray):
        out = {
            "type": "np.ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        if value.size == 1:
            out["value"] = value.reshape(-1)[0].item()
        return out
    if isinstance(value, dict):
        return {key: summarize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "preview": [summarize_value(item) for item in value[:4]],
        }
    return value


def save_json(path: pathlib.Path, payload: Any):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_text(path: pathlib.Path, content: str):
    path.write_text(content, encoding="utf-8")


def save_image(path: pathlib.Path, rgb_image: np.ndarray):
    image = np.asarray(rgb_image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def select_normalizer_fields(normalizer, use_relative_action: bool):
    if normalizer is None:
        return None, None
    if use_relative_action:
        return normalizer["states"], normalizer["actions"]
    motion_field = normalizer["motions"]
    return motion_field, motion_field


def project_points(points_3d: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsic.astype(np.float32).tolist()
    xyz = np.asarray(points_3d, dtype=np.float32)
    z = np.clip(xyz[:, 2], 1e-6, None)
    u = fx * (xyz[:, 0] / z) + cx
    v = fy * (xyz[:, 1] / z) + cy
    return np.stack([u, v], axis=1)


def build_fingertip_geometry(motion: np.ndarray) -> dict[str, np.ndarray]:
    motion = np.asarray(motion, dtype=np.float32)
    wrist = motion[:18]
    hand_wrist = motion[18:]
    hand_camera = transform_hand_points_from_wrist_to_camera_frame(
        hand_wrist.reshape(1, -1),
        wrist.reshape(1, -1),
    ).reshape(-1)
    return {
        "left_wrist": wrist[:3],
        "right_wrist": wrist[3:6],
        "left_tips": hand_camera[:15].reshape(5, 3),
        "right_tips": hand_camera[15:].reshape(5, 3),
    }


def color_with_strength(color: tuple[int, int, int], index: int, total: int) -> tuple[int, int, int]:
    if total <= 1:
        return color
    strength = 0.35 + 0.65 * float(index + 1) / float(total)
    return tuple(int(channel * strength) for channel in color)


def draw_hand_pose(
    image: np.ndarray,
    geometry: dict[str, np.ndarray],
    intrinsic: np.ndarray,
    color: tuple[int, int, int],
    presence: int,
    thickness: int = 2,
) -> np.ndarray:
    canvas = image.copy()
    show_left_hand = presence in [1, 3]
    show_right_hand = presence in [2, 3]

    def draw_one_hand(wrist_key: str, tips_key: str):
        wrist_2d = project_points(geometry[wrist_key][None, :], intrinsic)[0]
        tips_2d = project_points(geometry[tips_key], intrinsic)
        wrist_xy = tuple(np.round(wrist_2d).astype(int))
        if 0 <= wrist_xy[0] < canvas.shape[1] and 0 <= wrist_xy[1] < canvas.shape[0]:
            cv2.circle(canvas, wrist_xy, 6, color, -1)
        for tip in tips_2d:
            tip_xy = tuple(np.round(tip).astype(int))
            if 0 <= tip_xy[0] < canvas.shape[1] and 0 <= tip_xy[1] < canvas.shape[0]:
                cv2.line(canvas, wrist_xy, tip_xy, color, thickness)
                cv2.circle(canvas, tip_xy, 4, color, -1)

    if show_left_hand:
        draw_one_hand("left_wrist", "left_tips")
    if show_right_hand:
        draw_one_hand("right_wrist", "right_tips")
    return canvas


def draw_sequence_overlay(
    background: np.ndarray,
    sequence: np.ndarray,
    intrinsic: np.ndarray,
    base_color: tuple[int, int, int],
    presence: int,
) -> np.ndarray:
    canvas = background.copy()
    for index, motion in enumerate(sequence):
        geometry = build_fingertip_geometry(motion)
        canvas = draw_hand_pose(
            canvas,
            geometry,
            intrinsic,
            color_with_strength(base_color, index, len(sequence)),
            presence,
        )
    return canvas


def build_combined_overlay(
    background: np.ndarray,
    states_absolute: np.ndarray,
    actions_absolute: np.ndarray,
    intrinsic: np.ndarray,
    presence: int,
) -> np.ndarray:
    canvas = background.copy()
    if len(states_absolute) > 0:
        canvas = draw_sequence_overlay(
            canvas,
            states_absolute,
            intrinsic,
            base_color=(64, 180, 255),
            presence=presence,
        )
    if len(actions_absolute) > 0:
        canvas = draw_sequence_overlay(
            canvas,
            actions_absolute,
            intrinsic,
            base_color=(255, 140, 64),
            presence=presence,
        )
    return canvas


def prepare_dataset(cfg, args):
    cfg.data_collator.debug_capture_texts = True
    cfg.dataset.vla_dataset.debug_capture_raw_sample = True
    cfg.dataset.vla_dataset.debug_capture_processed_sample = True

    data_collator = hydra.utils.instantiate(cfg.data_collator)

    if args.dataset_kind == "vla":
        dataset = hydra.utils.instantiate(cfg.dataset.vla_dataset)
        dataset.set_collator(data_collator)
    else:
        dataset = hydra.utils.instantiate(cfg.dataset)
        dataset.vla_dataset.set_collator(data_collator)
        if dataset.vlm_dataset is not None:
            dataset.vlm_dataset.set_collator(data_collator)

    normalizer_path = args.normalizer_path or cfg.training.normalizer_path
    normalizer = None
    if normalizer_path is not None:
        with open(normalizer_path, "rb") as file_obj:
            normalizer = pickle.load(file_obj)

    if args.dataset_kind == "vla":
        if normalizer is not None:
            dataset.set_normalizer(normalizer)
        if args.split == "val":
            dataset = dataset.get_validation_dataset()
    else:
        if normalizer is not None:
            dataset.vla_dataset.set_normalizer(normalizer)
        if args.split == "val":
            dataset = dataset.get_validation_dataset()

    if hasattr(dataset, "batch_size"):
        dataset.batch_size = args.sample_count

    return dataset, data_collator, normalizer, normalizer_path


def build_dataloader(cfg, dataset, args):
    loader_cfg_ref = cfg.val_dataloader.loader if args.split == "val" else cfg.dataloader.loader
    loader_cfg = dict(OmegaConf.to_container(loader_cfg_ref, resolve=True))
    loader_cfg["batch_size"] = args.sample_count
    loader_cfg["num_workers"] = args.num_workers
    loader_cfg["pin_memory"] = False
    loader_cfg["shuffle"] = False
    loader_cfg["drop_last"] = False
    if args.num_workers <= 0:
        loader_cfg.pop("persistent_workers", None)
        loader_cfg.pop("prefetch_factor", None)
    else:
        loader_cfg["persistent_workers"] = bool(loader_cfg.get("persistent_workers", True))
    return DataLoader(
        dataset=dataset,
        collate_fn=dataset.get_collator(),
        **loader_cfg,
    )


def build_sample_export(
    sample_index: int,
    batch: dict[str, Any],
    tokenizer,
    state_normalizer,
    action_normalizer,
    use_relative_action: bool,
    motion_type: str,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    if motion_type != "fingertips":
        raise NotImplementedError(
            f"Overlay export is only implemented for fingertips. Got: {motion_type}"
        )

    sample_dir = output_dir / f"sample_{sample_index:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    raw_sample = None
    if "debug_raw_sample" in batch:
        raw_sample = batch["debug_raw_sample"][sample_index]
    processed_sample = None
    if "debug_processed_sample" in batch:
        processed_sample = batch["debug_processed_sample"][sample_index]

    if raw_sample is not None:
        torch.save(copy_for_save(raw_sample), sample_dir / "raw_sample.pt")
        save_json(sample_dir / "raw_sample_summary.json", summarize_value(raw_sample))
    if processed_sample is not None:
        torch.save(copy_for_save(processed_sample), sample_dir / "processed_sample.pt")
        save_json(sample_dir / "processed_sample_summary.json", summarize_value(processed_sample))

    input_ids = batch["input_ids"][sample_index]
    attention_mask = batch["attention_mask"][sample_index].bool()
    labels = batch["labels"][sample_index]
    valid_input_ids = input_ids[attention_mask].detach().cpu()
    valid_labels = labels[attention_mask].detach().cpu()
    token_ids = valid_input_ids.tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    save_text(sample_dir / "decoded_from_input_ids.txt", decoded_text)
    save_json(sample_dir / "token_ids.json", token_ids)
    save_json(sample_dir / "tokens.json", tokens)
    save_json(sample_dir / "labels_valid.json", valid_labels.tolist())

    full_text = batch.get("debug_full_texts", [None] * len(batch["input_ids"]))[sample_index]
    prompt_text = batch.get("debug_prompt_texts", [None] * len(batch["input_ids"]))[sample_index]
    full_messages = batch.get("debug_full_messages", [None] * len(batch["input_ids"]))[sample_index]
    prompt_messages = batch.get("debug_prompt_messages", [None] * len(batch["input_ids"]))[sample_index]
    if full_text is not None:
        save_text(sample_dir / "full_text.txt", full_text)
    if prompt_text is not None:
        save_text(sample_dir / "prompt_text.txt", prompt_text)
    if full_messages is not None:
        save_json(sample_dir / "full_messages.json", copy_for_save(full_messages))
    if prompt_messages is not None:
        save_json(sample_dir / "prompt_messages.json", copy_for_save(prompt_messages))

    n_states = int(batch["n_states"][sample_index].item())
    n_actions = int(batch["n_actions"][sample_index].item())
    presence = 3
    intrinsic = None
    current_frame = None

    if processed_sample is not None:
        if processed_sample.get("intrinsic") is not None:
            intrinsic = tensor_to_numpy(processed_sample["intrinsic"]).astype(np.float32)
            np.save(sample_dir / "intrinsic.npy", intrinsic)
        if processed_sample.get("images") is not None:
            images = tensor_to_numpy(processed_sample["images"])
            for frame_index, frame in enumerate(images):
                save_image(sample_dir / f"rgb_{frame_index:03d}.png", frame)
            current_frame = images[-1]
            save_image(sample_dir / "current_frame.png", current_frame)
        if processed_sample.get("presence") is not None:
            presence = int(tensor_to_numpy(processed_sample["presence"]).reshape(-1)[0])
    elif raw_sample is not None:
        intrinsic = tensor_to_numpy(raw_sample["intrinsic"]).astype(np.float32)
        np.save(sample_dir / "intrinsic.npy", intrinsic)
        images = tensor_to_numpy(raw_sample["image"])
        for frame_index, frame in enumerate(images):
            save_image(sample_dir / f"rgb_{frame_index:03d}.png", frame)
        current_frame = images[-1]
        save_image(sample_dir / "current_frame.png", current_frame)
        if raw_sample.get("presence") is not None:
            presence = int(tensor_to_numpy(raw_sample["presence"]).reshape(-1)[0])

    states_after_norm_full = batch["states"][sample_index].detach().cpu().numpy()
    actions_after_norm_full = batch["actions"][sample_index].detach().cpu().numpy()
    actions_valid_mask = batch["actions_valid_mask"][sample_index].detach().cpu().numpy()

    np.save(sample_dir / "states_after_norm_full.npy", states_after_norm_full)
    np.save(sample_dir / "actions_after_norm_full.npy", actions_after_norm_full)
    np.save(sample_dir / "actions_valid_mask.npy", actions_valid_mask)

    states_after_norm_valid = states_after_norm_full[:n_states]
    actions_after_norm_valid = actions_after_norm_full[:n_actions]
    np.save(sample_dir / "states_after_norm_valid.npy", states_after_norm_valid)
    np.save(sample_dir / "actions_after_norm_valid.npy", actions_after_norm_valid)

    if state_normalizer is not None and n_states > 0:
        states_before_norm_valid = state_normalizer.unnormalize(
            batch["states"][sample_index, :n_states]
        ).detach().cpu().numpy()
    else:
        states_before_norm_valid = states_after_norm_valid.copy()

    if action_normalizer is not None and n_actions > 0:
        actions_before_norm_valid = action_normalizer.unnormalize(
            batch["actions"][sample_index, :n_actions]
        ).detach().cpu().numpy()
    else:
        actions_before_norm_valid = actions_after_norm_valid.copy()

    np.save(sample_dir / "states_before_norm_valid.npy", states_before_norm_valid)
    np.save(sample_dir / "actions_before_norm_valid.npy", actions_before_norm_valid)

    if use_relative_action and len(states_before_norm_valid) > 0 and len(actions_before_norm_valid) > 0:
        actions_absolute_valid = get_absolute_action(
            states_before_norm_valid[-1],
            actions_before_norm_valid,
        )
    else:
        actions_absolute_valid = actions_before_norm_valid.copy()

    np.save(sample_dir / "actions_absolute_valid.npy", actions_absolute_valid)

    states_absolute_valid = states_before_norm_valid.copy()

    state_overlay_path = None
    action_overlay_path = None
    combined_overlay_path = None
    if intrinsic is not None and current_frame is not None:
        state_overlay = draw_sequence_overlay(
            current_frame,
            states_absolute_valid,
            intrinsic,
            base_color=(64, 180, 255),
            presence=presence,
        )
        action_overlay = draw_sequence_overlay(
            current_frame,
            actions_absolute_valid,
            intrinsic,
            base_color=(255, 140, 64),
            presence=presence,
        )
        combined_overlay = build_combined_overlay(
            current_frame,
            states_absolute_valid,
            actions_absolute_valid,
            intrinsic,
            presence,
        )
        state_overlay_path = sample_dir / "states_overlay.png"
        action_overlay_path = sample_dir / "actions_overlay.png"
        combined_overlay_path = sample_dir / "combined_overlay.png"
        save_image(state_overlay_path, state_overlay)
        save_image(action_overlay_path, action_overlay)
        save_image(combined_overlay_path, combined_overlay)

    metadata = {
        "sample_index": sample_index,
        "n_states": n_states,
        "n_actions": n_actions,
        "presence": presence,
        "is_vla_data": bool(batch["is_vla_data"][sample_index].item()),
        "answer_start_idx": int(batch["answer_start_idx"][sample_index].item()),
        "full_text_path": "full_text.txt" if full_text is not None else None,
        "prompt_text_path": "prompt_text.txt" if prompt_text is not None else None,
        "decoded_text_path": "decoded_from_input_ids.txt",
        "current_frame_path": "current_frame.png" if current_frame is not None else None,
        "states_overlay_path": state_overlay_path.name if state_overlay_path is not None else None,
        "actions_overlay_path": action_overlay_path.name if action_overlay_path is not None else None,
        "combined_overlay_path": combined_overlay_path.name if combined_overlay_path is not None else None,
    }

    if processed_sample is not None:
        if "instruction" in processed_sample:
            metadata["instruction"] = processed_sample["instruction"]
        if "dataset_name" in processed_sample:
            metadata["dataset_name"] = processed_sample["dataset_name"]
        if "episode_index" in processed_sample:
            metadata["episode_index"] = int(tensor_to_numpy(processed_sample["episode_index"]).reshape(-1)[0])
    elif raw_sample is not None:
        if "instruction" in raw_sample:
            metadata["instruction_candidates"] = raw_sample["instruction"]
        if "dataset_name" in raw_sample:
            metadata["dataset_name"] = raw_sample["dataset_name"]
        if "episode_index" in raw_sample:
            metadata["episode_index"] = int(tensor_to_numpy(raw_sample["episode_index"]).reshape(-1)[0])

    save_json(sample_dir / "metadata.json", metadata)
    return metadata


def write_html_report(output_dir: pathlib.Path, manifest: dict[str, Any], sample_entries: list[dict[str, Any]]):
    sections = [
        "<html><head><meta charset='utf-8'><title>LegendVLA Batch Debug</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;}pre{white-space:pre-wrap;background:#f6f8fa;padding:12px;border-radius:8px;}img{max-width:420px;border:1px solid #ddd;margin:6px 12px 6px 0;}code{background:#f1f1f1;padding:2px 4px;border-radius:4px;}section{margin-bottom:32px;}</style>",
        "</head><body>",
        "<h1>LegendVLA Batch Debug Report</h1>",
        "<h2>Manifest</h2>",
        f"<pre>{html.escape(json.dumps(manifest, indent=2, ensure_ascii=False))}</pre>",
    ]

    for sample in sample_entries:
        sample_dir = f"sample_{sample['sample_index']:03d}"
        title = f"Sample {sample['sample_index']:03d}"
        instruction = sample.get("instruction") or sample.get("instruction_candidates")
        sections.append("<section>")
        sections.append(f"<h2>{html.escape(title)}</h2>")
        sections.append(f"<p><b>Instruction:</b> {html.escape(str(instruction))}</p>")
        sections.append(
            f"<p><b>n_states:</b> {sample['n_states']} &nbsp; <b>n_actions:</b> {sample['n_actions']} &nbsp; <b>presence:</b> {sample['presence']}</p>"
        )
        if sample.get("current_frame_path"):
            sections.append(f"<img src='{sample_dir}/{sample['current_frame_path']}' alt='current frame'>")
        if sample.get("states_overlay_path"):
            sections.append(f"<img src='{sample_dir}/{sample['states_overlay_path']}' alt='states overlay'>")
        if sample.get("actions_overlay_path"):
            sections.append(f"<img src='{sample_dir}/{sample['actions_overlay_path']}' alt='actions overlay'>")
        if sample.get("combined_overlay_path"):
            sections.append(f"<img src='{sample_dir}/{sample['combined_overlay_path']}' alt='combined overlay'>")
        if sample.get("full_text_path"):
            full_text = (output_dir / sample_dir / sample["full_text_path"]).read_text(encoding="utf-8")
            sections.append("<h3>Full Text</h3>")
            sections.append(f"<pre>{html.escape(full_text)}</pre>")
        if sample.get("prompt_text_path"):
            prompt_text = (output_dir / sample_dir / sample["prompt_text_path"]).read_text(encoding="utf-8")
            sections.append("<h3>Prompt Text</h3>")
            sections.append(f"<pre>{html.escape(prompt_text)}</pre>")
        decoded_text = (output_dir / sample_dir / sample["decoded_text_path"]).read_text(encoding="utf-8")
        sections.append("<h3>Decoded Tokens</h3>")
        sections.append(f"<pre>{html.escape(decoded_text)}</pre>")
        sections.append("</section>")

    sections.append("</body></html>")
    (output_dir / "report.html").write_text("\n".join(sections), encoding="utf-8")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = pathlib.Path(args.config)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path, hydra_output_dir=output_dir)
    dataset, data_collator, normalizer, normalizer_path = prepare_dataset(cfg, args)
    dataloader = build_dataloader(cfg, dataset, args)

    motion_type = dataset.motion_type if args.dataset_kind == "vla" else dataset.vla_dataset.motion_type
    use_relative_action = dataset.use_relative_action if args.dataset_kind == "vla" else dataset.vla_dataset.use_relative_action

    if motion_type != "fingertips":
        raise NotImplementedError(
            f"Only fingertips visualization is implemented right now. Got: {motion_type}"
        )

    state_normalizer, action_normalizer = select_normalizer_fields(normalizer, use_relative_action)

    batch = next(iter(dataloader))
    torch.save(copy_for_save(batch), output_dir / "batch.pt")

    tokenizer = data_collator.batch_processor.tokenizer
    sample_entries = []
    for sample_index in range(batch["input_ids"].shape[0]):
        sample_entries.append(
            build_sample_export(
                sample_index=sample_index,
                batch=batch,
                tokenizer=tokenizer,
                state_normalizer=state_normalizer,
                action_normalizer=action_normalizer,
                use_relative_action=use_relative_action,
                motion_type=motion_type,
                output_dir=output_dir,
            )
        )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "dataset_kind": args.dataset_kind,
        "split": args.split,
        "sample_count": args.sample_count,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "normalizer_path": normalizer_path,
        "motion_type": motion_type,
        "use_relative_action": bool(use_relative_action),
        "batch_keys": sorted(batch.keys()),
    }
    save_json(output_dir / "manifest.json", manifest)
    write_html_report(output_dir, manifest, sample_entries)

    print(f"Saved batch debug artifacts to: {output_dir}")
    print(f"Open report: {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
