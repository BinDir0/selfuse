import pytest
import torch

from src.utils.batch_invariants import validate_attention_batch


def test_right_pad_and_labels_ok():
    pad = 0
    inp = torch.tensor([[1, 2, 3, pad, pad]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 10, -100, -100]], dtype=torch.long)
    validate_attention_batch(
        inp,
        mask,
        labels=labels,
        ignore_index=-100,
        pad_token_id=pad,
        padding_side="right",
        answer_start_idx=torch.tensor([2], dtype=torch.long),
    )


def test_right_pad_hole_fails():
    pad = 0
    inp = torch.tensor([[1, pad, 3, pad, pad]], dtype=torch.long)
    mask = torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.long)
    with pytest.raises(AssertionError, match="right-padded"):
        validate_attention_batch(
            inp,
            mask,
            labels=None,
            ignore_index=-100,
            pad_token_id=pad,
            padding_side="right",
        )


def test_pad_token_mismatch_fails():
    pad = 0
    inp = torch.tensor([[1, 2, 3, 99, 99]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
    with pytest.raises(AssertionError, match="pad_token_id"):
        validate_attention_batch(
            inp,
            mask,
            labels=None,
            ignore_index=-100,
            pad_token_id=pad,
            padding_side="right",
        )


def test_label_on_pad_fails():
    pad = 0
    inp = torch.tensor([[1, 2, 3, pad, pad]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
    labels = torch.tensor([[1, 2, 3, 9, -100]], dtype=torch.long)
    with pytest.raises(AssertionError, match="ignore_index"):
        validate_attention_batch(
            inp,
            mask,
            labels=labels,
            ignore_index=-100,
            pad_token_id=pad,
            padding_side="right",
        )


def test_left_pad_ok():
    pad = 0
    inp = torch.tensor([[pad, pad, 1, 2, 3]], dtype=torch.long)
    mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)
    validate_attention_batch(
        inp,
        mask,
        labels=torch.full_like(inp, -100),
        ignore_index=-100,
        pad_token_id=pad,
        padding_side="left",
        answer_start_idx=torch.tensor([2], dtype=torch.long),
    )
