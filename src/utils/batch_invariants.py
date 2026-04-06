"""Optional runtime checks for collated multimodal batches (padding mask, labels)."""

from __future__ import annotations

from typing import Literal

import torch


def validate_attention_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    labels: torch.Tensor | None,
    ignore_index: int,
    pad_token_id: int | None,
    padding_side: Literal["left", "right"] = "right",
    answer_start_idx: torch.Tensor | None = None,
) -> None:
    """Assert padding mask / labels invariants for one collated batch.

    Checks:
      - attention_mask entries are 0 or 1
      - padding layout is contiguous (all ones then all zeros for right pad; mirrored for left)
      - where attention_mask == 0, input_ids equal pad_token_id (when pad_token_id is set)
      - where attention_mask == 0, labels equal ignore_index (when labels provided)

    Enable from training via UnifiedVLACollator.validate_batch_invariants or env EGOVLA_VALIDATE_BATCH=1.
    """
    if input_ids.shape != attention_mask.shape:
        raise AssertionError(
            f"input_ids {tuple(input_ids.shape)} and attention_mask {tuple(attention_mask.shape)} must match"
        )
    if not torch.is_floating_point(attention_mask):
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise AssertionError("attention_mask must be 0/1 (or cast to bool/long with only 0/1).")

    mask_bool = attention_mask.bool()
    batch_size, seq_len = input_ids.shape

    for b in range(batch_size):
        m = mask_bool[b]
        if padding_side == "right":
            # No 'valid' token after a padded position.
            if torch.any((~m).cumsum(dim=-1) * m):
                raise AssertionError(
                    f"Row {b}: attention_mask is not right-padded (valid tokens after padding)."
                )
        elif padding_side == "left":
            # No 'valid' token before a padded position (padding only on the left).
            if torch.any(m.cumsum(dim=-1) * ~m):
                raise AssertionError(
                    f"Row {b}: attention_mask is not left-padded (valid tokens before padding)."
                )
        else:
            raise ValueError(f"Unknown padding_side: {padding_side}")

        if pad_token_id is not None:
            pad_pos = ~m
            if torch.any(pad_pos):
                ids_at_pad = input_ids[b][pad_pos]
                if not torch.all(ids_at_pad == int(pad_token_id)):
                    raise AssertionError(
                        f"Row {b}: input_ids at attention_mask==0 must equal pad_token_id={pad_token_id}, "
                        f"got unique {torch.unique(ids_at_pad).tolist()}"
                    )

        if labels is not None:
            if labels.shape != input_ids.shape:
                raise AssertionError(
                    f"labels {tuple(labels.shape)} must match input_ids {tuple(input_ids.shape)}"
                )
            lab = labels[b]
            if torch.any(lab[~m] != int(ignore_index)):
                bad = lab[~m][lab[~m] != int(ignore_index)]
                raise AssertionError(
                    f"Row {b}: labels must be ignore_index={ignore_index} where attention_mask==0, "
                    f"found values like {bad[:8].tolist()}"
                )

    if answer_start_idx is not None:
        if answer_start_idx.shape != (batch_size,):
            raise AssertionError(
                f"answer_start_idx shape {tuple(answer_start_idx.shape)} expected ({batch_size},)"
            )
        if torch.any(answer_start_idx < 0) or torch.any(answer_start_idx > seq_len):
            raise AssertionError(
                f"answer_start_idx must be in [0, {seq_len}], "
                f"got min={answer_start_idx.min().item()} max={answer_start_idx.max().item()}"
            )
        # First supervised / generation token should lie in a non-padded position when strictly inside the row.
        for b in range(batch_size):
            s = int(answer_start_idx[b].item())
            if s < seq_len and not bool(m[s].item()):
                raise AssertionError(
                    f"Row {b}: answer_start_idx={s} points to a padded position (attention_mask==0)."
                )


def should_validate_batch(*, flag: bool) -> bool:
    """Return True if collator flag or EGOVLA_VALIDATE_BATCH env requests validation."""
    if flag:
        return True
    import os

    v = os.environ.get("EGOVLA_VALIDATE_BATCH", "").strip().lower()
    return v in ("1", "true", "yes", "on")
