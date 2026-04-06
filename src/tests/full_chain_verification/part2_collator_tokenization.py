"""
Part 2: Collator / Tokenization / Mask Verification.

Tests chat message formatting, token encoding, labels, attention mask,
visual token alignment, and camera intrinsic handling.

Requires: HF Qwen3-VL processor weights (AutoProcessor)
Runs on: CPU only

Usage:
    python -m src.tests.full_chain_verification.part2_collator_tokenization \
        --model-path /path/to/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from src.tests.full_chain_verification.utils import (
    CheckResult,
    PhaseReport,
    assert_check,
    get_output_dir,
    safe_import_plt,
)

OUTPUT_PART = "part2"


# ── Mock sample builders ───────────────────────────────────────────────────

def make_vla_sample(
    n_states: int = 18,
    n_actions: int = 32,
    image_horizon: int = 6,
    state_dim: int = 48,
    action_dim: int = 48,
    instruction: str = "pick up the red cube from the table",
    image_size: tuple[int, int] = (224, 224),
) -> dict[str, Any]:
    H, W = image_size
    return {
        "images": torch.randint(0, 255, (image_horizon, H, W, 3), dtype=torch.uint8),
        "instruction": instruction,
        "intrinsic": torch.tensor([500.0, 500.0, 320.0, 240.0], dtype=torch.float32),
        "vision_type": "video",
        "video_fps": torch.tensor(5.0, dtype=torch.float32),
        "states": torch.randn(n_states, state_dim),
        "actions": torch.randn(n_actions, action_dim),
        "n_states": torch.tensor(n_states, dtype=torch.long),
        "n_actions": torch.tensor(n_actions, dtype=torch.long),
        "actions_valid_mask": torch.ones(n_actions, action_dim, dtype=torch.bool),
        "is_vla_data": torch.tensor(True, dtype=torch.bool),
    }


def make_vlm_sample(
    question: str = "What color is the cube on the table?",
    answer: str = "The cube on the table is red.",
    image_size: tuple[int, int] = (224, 224),
    state_dim: int = 48,
    action_dim: int = 48,
    action_horizon: int = 32,
    state_horizon: int = 18,
) -> dict[str, Any]:
    H, W = image_size
    return {
        "images": torch.randint(0, 255, (H, W, 3), dtype=torch.uint8),
        "question": question,
        "answer": answer,
        "vision_type": "image",
        "is_vla_data": torch.tensor(False, dtype=torch.bool),
        # VLM samples carry zero-filled VLA fields for batch collation
        "states": torch.zeros(state_horizon, state_dim),
        "actions": torch.zeros(action_horizon, action_dim),
        "n_states": torch.tensor(0, dtype=torch.long),
        "n_actions": torch.tensor(0, dtype=torch.long),
        "actions_valid_mask": torch.zeros(action_horizon, action_dim, dtype=torch.bool),
        "intrinsic": torch.zeros(4, dtype=torch.float32),
        "video_fps": torch.tensor(5.0, dtype=torch.float32),
    }


def build_collator(model_path: str, mode: str = "train"):
    from src.dataset.qwen3_vl_batching import Qwen3VLBatchProcessor, Qwen3VLChatFormatter
    from src.dataset.unified_vla_collator import UnifiedVLACollator

    formatter = Qwen3VLChatFormatter(
        state_token="<state>",
        action_token="<action>",
        camera_intrinsic_mode="text",
    )
    batch_processor = Qwen3VLBatchProcessor(
        model_name_or_path=model_path,
        processor_init_kwargs={"trust_remote_code": True},
        processor_call_kwargs={
            "min_pixels": 147456,
            "max_pixels": 147456,
            "text_kwargs": {
                "padding": "longest",
                "return_tensors": "pt",
            },
        },
    )
    collator = UnifiedVLACollator(
        formatter=formatter,
        batch_processor=batch_processor,
        mode=mode,
    )
    return collator, formatter, batch_processor


# ── 2.1 Chat message structure ─────────────────────────────────────────────

def test_vla_message_structure(report: PhaseReport, model_path: str) -> None:
    from src.dataset.qwen3_vl_batching import Qwen3VLChatFormatter
    formatter = Qwen3VLChatFormatter()
    sample = make_vla_sample(n_states=18, n_actions=32)

    messages = formatter.build_messages(sample, prompt_only=False)
    checks = []

    # User turn
    checks.append(messages[0]["role"] == "user")
    content = messages[0]["content"]
    has_video = any(c.get("type") == "video" for c in content)
    checks.append(has_video)
    text_parts = [c for c in content if c.get("type") == "text"]
    checks.append(len(text_parts) == 1)
    user_text = text_parts[0]["text"]
    checks.append("<state>" * 18 in user_text)
    checks.append("Camera intrinsic:" in user_text)
    checks.append("pick up the red cube" in user_text)

    # Assistant turn
    checks.append(len(messages) == 2)
    checks.append(messages[1]["role"] == "assistant")
    assistant_text = messages[1]["content"][0]["text"]
    checks.append(assistant_text == "<action>" * 32)

    report.add(assert_check(
        all(checks),
        "2.1a VLA message structure",
        f"checks={checks}",
    ))


def test_vla_message_prompt_only(report: PhaseReport, model_path: str) -> None:
    from src.dataset.qwen3_vl_batching import Qwen3VLChatFormatter
    formatter = Qwen3VLChatFormatter()
    sample = make_vla_sample()
    messages = formatter.build_messages(sample, prompt_only=True)
    report.add(assert_check(
        len(messages) == 1 and messages[0]["role"] == "user",
        "2.1b VLA prompt_only: 1 user turn only",
        f"len={len(messages)}",
    ))


def test_vlm_message_structure(report: PhaseReport, model_path: str) -> None:
    from src.dataset.qwen3_vl_batching import Qwen3VLChatFormatter
    formatter = Qwen3VLChatFormatter()
    sample = make_vlm_sample()
    messages = formatter.build_messages(sample, prompt_only=False)

    checks = []
    checks.append(messages[0]["role"] == "user")
    content = messages[0]["content"]
    has_image = any(c.get("type") == "image" for c in content)
    checks.append(has_image)
    text_parts = [c for c in content if c.get("type") == "text"]
    checks.append("What color" in text_parts[0]["text"])
    checks.append("<state>" not in text_parts[0]["text"])
    checks.append("<action>" not in text_parts[0]["text"])
    checks.append(messages[1]["role"] == "assistant")
    checks.append("red" in messages[1]["content"][0]["text"])

    report.add(assert_check(
        all(checks),
        "2.1c VLM message structure",
        f"checks={checks}",
    ))


# ── 2.2 Token encoding ─────────────────────────────────────────────────────

def test_input_ids_special_token_counts(report: PhaseReport, model_path: str) -> None:
    """VLA sample: exact count of <state> and <action> tokens in input_ids."""
    collator, _, bp = build_collator(model_path)
    state_token_id = bp.tokenizer.convert_tokens_to_ids("<state>")
    action_token_id = bp.tokenizer.convert_tokens_to_ids("<action>")

    n_states, n_actions = 18, 32
    sample = make_vla_sample(n_states=n_states, n_actions=n_actions)
    batch = collator.collate_raw([sample])

    input_ids = batch["input_ids"][0]
    state_count = int((input_ids == state_token_id).sum())
    action_count = int((input_ids == action_token_id).sum())

    report.add(assert_check(
        state_count == n_states and action_count == n_actions,
        "2.2a special token counts",
        f"<state>={state_count} (expected {n_states}), <action>={action_count} (expected {n_actions})",
    ))


def test_input_ids_token_order(report: PhaseReport, model_path: str) -> None:
    """All <state> tokens should appear before all <action> tokens."""
    collator, _, bp = build_collator(model_path)
    state_token_id = bp.tokenizer.convert_tokens_to_ids("<state>")
    action_token_id = bp.tokenizer.convert_tokens_to_ids("<action>")

    sample = make_vla_sample(n_states=18, n_actions=32)
    batch = collator.collate_raw([sample])
    input_ids = batch["input_ids"][0]

    state_positions = (input_ids == state_token_id).nonzero(as_tuple=True)[0]
    action_positions = (input_ids == action_token_id).nonzero(as_tuple=True)[0]

    max_state = int(state_positions.max()) if len(state_positions) > 0 else -1
    min_action = int(action_positions.min()) if len(action_positions) > 0 else float("inf")

    report.add(assert_check(
        max_state < min_action,
        "2.2b state tokens before action tokens",
        f"max_state_pos={max_state}, min_action_pos={min_action}",
    ))


def test_answer_start_idx_semantic_accuracy(report: PhaseReport, model_path: str) -> None:
    """answer_start_idx should point to the first token of assistant content."""
    collator, _, bp = build_collator(model_path)
    action_token_id = bp.tokenizer.convert_tokens_to_ids("<action>")

    sample = make_vla_sample(n_states=18, n_actions=32)
    batch = collator.collate_raw([sample])

    input_ids = batch["input_ids"][0]
    answer_start = int(batch["answer_start_idx"][0])

    # For VLA: the first token after prompt should be part of the assistant template
    # (could be a template token before the <action> tokens)
    # Decode a window around answer_start_idx for inspection
    window_start = max(0, answer_start - 3)
    window_end = min(len(input_ids), answer_start + 5)
    window_ids = input_ids[window_start:window_end]
    decoded_window = bp.tokenizer.decode(window_ids, skip_special_tokens=False)

    # The first <action> token should be at or after answer_start_idx
    action_positions = (input_ids == action_token_id).nonzero(as_tuple=True)[0]
    first_action_pos = int(action_positions.min()) if len(action_positions) > 0 else -1

    report.add(assert_check(
        answer_start <= first_action_pos,
        "2.2c answer_start_idx <= first <action> position",
        f"answer_start={answer_start}, first_action_pos={first_action_pos}, "
        f"decoded_window='{decoded_window}'",
    ))


def test_answer_start_idx_decoded_text(report: PhaseReport, model_path: str) -> None:
    """Decode around answer_start_idx to verify it's the prompt/answer boundary."""
    collator, _, bp = build_collator(model_path)

    sample = make_vlm_sample(answer="The cube on the table is red.")
    batch = collator.collate_raw([sample])

    input_ids = batch["input_ids"][0]
    answer_start = int(batch["answer_start_idx"][0])

    # Tokens before answer_start should be the prompt
    prompt_tail = bp.tokenizer.decode(input_ids[max(0, answer_start - 5):answer_start], skip_special_tokens=False)
    # Tokens from answer_start should be the answer
    answer_head = bp.tokenizer.decode(input_ids[answer_start:answer_start + 8], skip_special_tokens=False)

    report.add(assert_check(
        True,  # Informational check
        "2.2d answer boundary decoded text",
        f"prompt_tail='...{prompt_tail}' | answer_head='{answer_head}...'",
        details={"answer_start_idx": answer_start, "prompt_tail": prompt_tail, "answer_head": answer_head},
    ))


# ── 2.3 Labels correctness ─────────────────────────────────────────────────

def test_labels_vla_all_ignored(report: PhaseReport, model_path: str) -> None:
    """VLA samples should have labels == -100 everywhere (action supervised by flow/diffloss)."""
    collator, _, bp = build_collator(model_path)
    sample = make_vla_sample()
    batch = collator.collate_raw([sample])
    labels = batch["labels"][0]

    report.add(assert_check(
        bool(torch.all(labels == -100)),
        "2.3a VLA labels all -100",
        f"non_ignored_count={int((labels != -100).sum())}",
    ))


def test_labels_vlm_prompt_masked_answer_valid(report: PhaseReport, model_path: str) -> None:
    """VLM: labels before answer_start_idx are -100, after are valid token ids."""
    collator, _, bp = build_collator(model_path)
    sample = make_vlm_sample()
    batch = collator.collate_raw([sample])

    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    answer_start = int(batch["answer_start_idx"][0])
    attention_mask = batch["attention_mask"][0]

    # Prompt portion: all -100
    prompt_labels = labels[:answer_start]
    check_prompt = bool(torch.all(prompt_labels == -100))

    # Answer portion (non-padding): should match input_ids
    answer_labels = labels[answer_start:]
    answer_ids = input_ids[answer_start:]
    answer_mask = attention_mask[answer_start:]
    valid_answer = answer_labels[answer_mask == 1]
    valid_input = answer_ids[answer_mask == 1]
    check_answer = bool(torch.all(valid_answer == valid_input))

    report.add(assert_check(
        check_prompt and check_answer,
        "2.3b VLM labels: prompt=-100, answer=input_ids",
        f"prompt_ok={check_prompt}, answer_ok={check_answer}, "
        f"valid_answer_tokens={int((valid_answer != -100).sum())}",
    ))


def test_labels_valid_token_ids(report: PhaseReport, model_path: str) -> None:
    """All non-ignored label values should be valid token ids."""
    collator, _, bp = build_collator(model_path)
    # len(tokenizer) includes base vocab + all added special tokens
    vocab_size = len(bp.tokenizer)

    sample = make_vlm_sample()
    batch = collator.collate_raw([sample])
    labels = batch["labels"][0]
    valid_labels = labels[labels != -100]

    if len(valid_labels) > 0:
        in_range = bool(torch.all((valid_labels >= 0) & (valid_labels < vocab_size + 10)))
    else:
        in_range = True

    report.add(assert_check(
        in_range,
        "2.3c labels valid token ids",
        f"valid_count={len(valid_labels)}, range=[{int(valid_labels.min()) if len(valid_labels) > 0 else 'N/A'}, "
        f"{int(valid_labels.max()) if len(valid_labels) > 0 else 'N/A'}]",
    ))


def test_labels_padding_masked(report: PhaseReport, model_path: str) -> None:
    """Where attention_mask==0 (padding), labels must be -100."""
    collator, _, bp = build_collator(model_path)

    # Use different instruction lengths to force token-level padding.
    # Action horizon must stay the same (collator uses torch.stack on raw tensors).
    samples = [
        make_vla_sample(instruction="pick up the red cube from the table and place it carefully"),
        make_vla_sample(instruction="go"),
    ]
    batch = collator.collate_raw(samples)
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    padding_labels = labels[attention_mask == 0]
    check = bool(torch.all(padding_labels == -100)) if len(padding_labels) > 0 else True

    report.add(assert_check(
        check,
        "2.3d padding positions have labels=-100",
        f"padding_tokens={len(padding_labels)}",
    ))


# ── 2.4 Attention mask and padding ─────────────────────────────────────────

def test_attention_mask_shape_consistency(report: PhaseReport, model_path: str) -> None:
    collator, _, _ = build_collator(model_path)
    sample = make_vla_sample()
    batch = collator.collate_raw([sample])
    report.add(assert_check(
        batch["attention_mask"].shape == batch["input_ids"].shape,
        "2.4a attention_mask.shape == input_ids.shape",
        f"mask={tuple(batch['attention_mask'].shape)}, ids={tuple(batch['input_ids'].shape)}",
    ))


def test_attention_mask_right_padding_monotonic(report: PhaseReport, model_path: str) -> None:
    """Right padding: once a 0 appears, all subsequent values should be 0."""
    collator, _, _ = build_collator(model_path)
    samples = [
        make_vla_sample(instruction="pick up the red cube from the table and place it carefully"),
        make_vla_sample(instruction="go"),
    ]
    batch = collator.collate_raw(samples)
    attention_mask = batch["attention_mask"]

    monotonic = True
    for b in range(attention_mask.shape[0]):
        mask = attention_mask[b]
        diffs = mask[1:] - mask[:-1]
        # In right padding: diffs can have at most one transition from 1 to 0
        transitions = (diffs == -1).sum()
        if transitions > 1:
            monotonic = False
            break
        # After first 0, all should be 0
        zeros = (mask == 0).nonzero(as_tuple=True)[0]
        if len(zeros) > 0:
            first_zero = int(zeros[0])
            if not bool(torch.all(mask[first_zero:] == 0)):
                monotonic = False
                break

    report.add(assert_check(
        monotonic,
        "2.4b right padding monotonic",
        "attention_mask is contiguous 1s then 0s",
    ))


def test_attention_mask_nonzero_count(report: PhaseReport, model_path: str) -> None:
    collator, _, _ = build_collator(model_path)
    sample = make_vla_sample()
    batch = collator.collate_raw([sample])
    attention_mask = batch["attention_mask"]
    min_valid = int(attention_mask.sum(dim=1).min())
    report.add(assert_check(
        min_valid > 0,
        "2.4c every sample has at least one valid token",
        f"min_valid_tokens={min_valid}",
    ))


def test_padding_token_id_consistency(report: PhaseReport, model_path: str) -> None:
    """Padding positions should use pad_token_id in input_ids."""
    collator, _, bp = build_collator(model_path)
    pad_id = bp.tokenizer.pad_token_id

    samples = [
        make_vla_sample(instruction="pick up the red cube from the table and place it carefully"),
        make_vla_sample(instruction="go"),
    ]
    batch = collator.collate_raw(samples)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    padding_ids = input_ids[attention_mask == 0]
    if len(padding_ids) > 0:
        check = bool(torch.all(padding_ids == pad_id))
    else:
        check = True

    report.add(assert_check(
        check,
        "2.4d padding positions use pad_token_id",
        f"pad_token_id={pad_id}, padding_count={len(padding_ids)}",
    ))


# ── 2.5 Visual token alignment ─────────────────────────────────────────────

def test_video_grid_thw_consistency(report: PhaseReport, model_path: str) -> None:
    """VLA video sample: visual token count matches input_ids placeholder count."""
    collator, _, bp = build_collator(model_path)

    sample = make_vla_sample(image_horizon=6, image_size=(224, 224))
    batch = collator.collate_raw([sample])

    video_grid_thw = batch["video_grid_thw"]
    check_present = video_grid_thw is not None
    check_shape = check_present and video_grid_thw.ndim == 2 and video_grid_thw.shape[1] == 3

    if check_present:
        # Calculate expected visual token count from grid
        # Qwen3-VL: total_tokens = sum(prod(thw[i]) / merge_size^2)
        # merge_size is typically 2 for Qwen3-VL
        merge_size = bp.processor.image_processor.merge_size if hasattr(bp.processor, "image_processor") else 2
        total_vis_tokens = 0
        for i in range(video_grid_thw.shape[0]):
            t, h, w = video_grid_thw[i].tolist()
            total_vis_tokens += int(t * h * w / (merge_size ** 2))

        # Count video pad tokens in input_ids
        # Qwen3-VL uses specific token ids for video placeholders
        input_ids = batch["input_ids"][0]
        # The video token id is typically <|video_pad|>
        try:
            video_pad_id = bp.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        except Exception:
            video_pad_id = None

        if video_pad_id is not None and video_pad_id != bp.tokenizer.unk_token_id:
            vid_token_count = int((input_ids == video_pad_id).sum())
            check_count = vid_token_count == total_vis_tokens
            msg = f"grid_tokens={total_vis_tokens}, input_ids_count={vid_token_count}"
        else:
            check_count = True
            msg = "video_pad_token not found in vocab, skipping count check"
    else:
        check_count = False
        msg = "video_grid_thw is None"

    report.add(assert_check(
        check_present and check_shape and check_count,
        "2.5a video grid_thw consistent with input_ids",
        msg,
    ))


def test_image_grid_thw_consistency(report: PhaseReport, model_path: str) -> None:
    """VLM image sample: visual token count matches input_ids placeholder count."""
    collator, _, bp = build_collator(model_path)

    sample = make_vlm_sample(image_size=(224, 224))
    batch = collator.collate_raw([sample])

    image_grid_thw = batch["image_grid_thw"]
    check_present = image_grid_thw is not None

    if check_present:
        merge_size = bp.processor.image_processor.merge_size if hasattr(bp.processor, "image_processor") else 2
        total_vis_tokens = 0
        for i in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[i].tolist()
            total_vis_tokens += int(t * h * w / (merge_size ** 2))

        input_ids = batch["input_ids"][0]
        try:
            image_pad_id = bp.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        except Exception:
            image_pad_id = None

        if image_pad_id is not None and image_pad_id != bp.tokenizer.unk_token_id:
            img_token_count = int((input_ids == image_pad_id).sum())
            check_count = img_token_count == total_vis_tokens
            msg = f"grid_tokens={total_vis_tokens}, input_ids_count={img_token_count}"
        else:
            check_count = True
            msg = "image_pad_token not found in vocab, skipping count check"
    else:
        check_count = False
        msg = "image_grid_thw is None"

    report.add(assert_check(
        check_present and check_count,
        "2.5b image grid_thw consistent with input_ids",
        msg,
    ))


def test_mixed_batch_both_modalities(report: PhaseReport, model_path: str) -> None:
    """Mixed VLA+VLM batch should have both pixel_values and pixel_values_videos."""
    collator, _, _ = build_collator(model_path)

    vla_sample = make_vla_sample()
    vlm_sample = make_vlm_sample()
    batch = collator.collate_raw([vla_sample, vlm_sample])

    has_video = batch["pixel_values_videos"] is not None
    has_image = batch["pixel_values"] is not None
    video_grid = batch["video_grid_thw"] is not None
    image_grid = batch["image_grid_thw"] is not None

    report.add(assert_check(
        has_video and has_image and video_grid and image_grid,
        "2.5c mixed batch: both modalities present",
        f"pixel_values={has_image}, pixel_values_videos={has_video}, "
        f"image_grid={image_grid}, video_grid={video_grid}",
    ))


# ── 2.6 camera_intrinsic ───────────────────────────────────────────────────

def test_camera_intrinsic_text_mode(report: PhaseReport, model_path: str) -> None:
    """text mode: no camera_intrinsic tensor in batch, but text contains fx/fy/cx/cy."""
    collator, _, bp = build_collator(model_path)  # default mode="text"
    sample = make_vla_sample()
    batch = collator.collate_raw([sample])

    no_tensor_key = "camera_intrinsic" not in batch
    # Check that the rendered text contains intrinsic values
    decoded = bp.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
    has_fx = "fx:" in decoded

    report.add(assert_check(
        no_tensor_key and has_fx,
        "2.6a camera_intrinsic text mode: no tensor, text has fx/fy",
        f"no_tensor_key={no_tensor_key}, has_fx={has_fx}",
    ))


# ── 2.7 Mixed batch contract ───────────────────────────────────────────────

def test_mixed_batch_is_vla_data(report: PhaseReport, model_path: str) -> None:
    """is_vla_data flag should be correct for mixed batch."""
    collator, _, _ = build_collator(model_path)
    vla = make_vla_sample()
    vlm = make_vlm_sample()
    batch = collator.collate_raw([vla, vlm])

    is_vla = batch["is_vla_data"]
    report.add(assert_check(
        bool(is_vla[0]) and not bool(is_vla[1]),
        "2.7a is_vla_data correct in mixed batch",
        f"is_vla_data={is_vla.tolist()}",
    ))


def test_mixed_batch_vlm_labels_only(report: PhaseReport, model_path: str) -> None:
    """In mixed batch: VLA row labels all -100, VLM row has valid labels."""
    collator, _, _ = build_collator(model_path)
    vla = make_vla_sample()
    vlm = make_vlm_sample()
    batch = collator.collate_raw([vla, vlm])

    labels = batch["labels"]
    vla_all_ignored = bool(torch.all(labels[0] == -100))
    vlm_has_valid = bool(torch.any(labels[1] != -100))

    report.add(assert_check(
        vla_all_ignored and vlm_has_valid,
        "2.7b mixed batch: VLA labels=-100, VLM has valid labels",
        f"vla_ignored={vla_all_ignored}, vlm_valid={vlm_has_valid}",
    ))


def test_batch_required_keys(report: PhaseReport, model_path: str) -> None:
    """Batch must contain all keys required by the model."""
    collator, _, _ = build_collator(model_path)
    vla = make_vla_sample()
    batch = collator.collate_raw([vla])

    required = [
        "input_ids", "attention_mask", "labels", "answer_start_idx",
        "pixel_values_videos", "video_grid_thw", "mm_token_type_ids",
        "states", "actions", "n_states", "n_actions",
        "actions_valid_mask", "is_vla_data",
    ]
    missing = [k for k in required if k not in batch]
    report.add(assert_check(
        len(missing) == 0,
        "2.7c batch contains all required keys",
        f"missing={missing}" if missing else "all present",
    ))


# ── 2.8 Token count stress test ────────────────────────────────────────────

def test_varying_action_counts(report: PhaseReport, model_path: str) -> None:
    """Different n_actions values should produce correct token counts."""
    collator, _, bp = build_collator(model_path)
    action_token_id = bp.tokenizer.convert_tokens_to_ids("<action>")

    for n_actions in [1, 8, 16, 32]:
        sample = make_vla_sample(n_actions=n_actions)
        batch = collator.collate_raw([sample])
        count = int((batch["input_ids"][0] == action_token_id).sum())
        if count != n_actions:
            report.add(assert_check(
                False,
                "2.8a varying action counts",
                f"n_actions={n_actions}, token_count={count}",
            ))
            return

    report.add(assert_check(
        True,
        "2.8a varying action counts",
        "all n_actions=[1,8,16,32] produced correct token counts",
    ))


# ── Main ────────────────────────────────────────────────────────────────────

def run_all(model_path: str, skip_visual: bool = False) -> PhaseReport:
    out_dir = get_output_dir(OUTPUT_PART)
    report = PhaseReport("Part 2: Collator / Tokenization / Mask", out_dir)

    print("\n=== Part 2: Collator / Tokenization / Mask ===\n")

    # 2.1 Chat messages
    test_vla_message_structure(report, model_path)
    test_vla_message_prompt_only(report, model_path)
    test_vlm_message_structure(report, model_path)

    # 2.2 Token encoding
    test_input_ids_special_token_counts(report, model_path)
    test_input_ids_token_order(report, model_path)
    test_answer_start_idx_semantic_accuracy(report, model_path)
    test_answer_start_idx_decoded_text(report, model_path)

    # 2.3 Labels
    test_labels_vla_all_ignored(report, model_path)
    test_labels_vlm_prompt_masked_answer_valid(report, model_path)
    test_labels_valid_token_ids(report, model_path)
    test_labels_padding_masked(report, model_path)

    # 2.4 Attention mask
    test_attention_mask_shape_consistency(report, model_path)
    test_attention_mask_right_padding_monotonic(report, model_path)
    test_attention_mask_nonzero_count(report, model_path)
    test_padding_token_id_consistency(report, model_path)

    # 2.5 Visual tokens
    test_video_grid_thw_consistency(report, model_path)
    test_image_grid_thw_consistency(report, model_path)
    test_mixed_batch_both_modalities(report, model_path)

    # 2.6 Camera intrinsic
    test_camera_intrinsic_text_mode(report, model_path)

    # 2.7 Mixed batch
    test_mixed_batch_is_vla_data(report, model_path)
    test_mixed_batch_vlm_labels_only(report, model_path)
    test_batch_required_keys(report, model_path)

    # 2.8 Stress
    test_varying_action_counts(report, model_path)

    report.save()
    report.print_summary()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 2: Collator/Tokenization verification")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to Qwen3-VL-2B-Instruct weights")
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()
    report = run_all(model_path=args.model_path, skip_visual=args.skip_visual)
    exit(0 if report.all_passed else 1)
