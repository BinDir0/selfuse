from __future__ import annotations

import torch

from src.tests.codex.common import (
    build_dummy_collator,
    build_wrapper_stub,
    make_expected_vla_sample,
    make_obs,
)


def test_prepare_process_repeat_matches_training_side_contract():
    collator = build_dummy_collator()
    wrapper = build_wrapper_stub(history_pad_mode="repeat", data_collator=collator)
    obs = make_obs([[1.0, 1.0], [2.0, 2.0]])

    prepared = wrapper.prepare_process(obs)
    expected_sample = make_expected_vla_sample(
        states=torch.tensor(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
            dtype=torch.float32,
        ),
        n_states=4,
    )
    expected_batch = collator([expected_sample])

    for key in [
        "input_ids",
        "attention_mask",
        "states",
        "n_states",
        "n_actions",
        "answer_start_idx",
        "is_vla_data",
    ]:
        assert torch.equal(prepared[key], expected_batch[key])


def test_prepare_process_truncate_matches_training_side_contract():
    collator = build_dummy_collator()
    wrapper = build_wrapper_stub(history_pad_mode="truncate", data_collator=collator)
    obs = make_obs([[1.0, 1.0], [2.0, 2.0]])

    prepared = wrapper.prepare_process(obs)
    expected_sample = make_expected_vla_sample(
        states=torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=torch.float32,
        ),
        n_states=2,
    )
    expected_batch = collator([expected_sample])

    for key in [
        "input_ids",
        "attention_mask",
        "states",
        "n_states",
        "n_actions",
        "answer_start_idx",
        "is_vla_data",
    ]:
        assert torch.equal(prepared[key], expected_batch[key])


def test_prepare_process_uses_default_instruction_when_obs_instruction_missing():
    wrapper = build_wrapper_stub(history_pad_mode="truncate")
    prepared = wrapper.prepare_process(make_obs([[1.0, 1.0]], instruction=None))
    assert prepared["n_states"].item() == 1


def test_configure_batch_processor_text_kwargs_updates_nested_text_kwargs():
    wrapper = build_wrapper_stub(history_pad_mode="truncate")
    wrapper.configure_batch_processor_text_kwargs(tokenizer_padding="left", max_length=128)

    text_kwargs = wrapper.data_collator.batch_processor.processor_call_kwargs["text_kwargs"]
    assert text_kwargs["padding"] == "left"
    assert text_kwargs["max_length"] == 128


def test_prepare_process_accepts_chest_fields_from_env_wrapper_path():
    class _SpyCollator:
        def __init__(self):
            self.last_samples = None

        def __call__(self, samples):
            self.last_samples = samples
            return {"dummy_float": torch.tensor([1.0], dtype=torch.float32)}

    spy_collator = _SpyCollator()
    wrapper = build_wrapper_stub(history_pad_mode="truncate", data_collator=spy_collator)
    obs = make_obs([[1.0, 1.0]])
    obs["chest_image"] = torch.zeros(1, 8, 8, 3, dtype=torch.uint8).numpy()
    obs["chest_intrinsic"] = torch.tensor([2.0, 2.0, 0.5, 0.5], dtype=torch.float32).numpy()

    prepared = wrapper.prepare_process(obs)
    assert "breast_images" in spy_collator.last_samples[0]
    assert "breast_intrinsic" in spy_collator.last_samples[0]
    assert torch.is_tensor(prepared["dummy_float"])

