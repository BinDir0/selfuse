"""Unit tests for world-model view masking."""

from types import SimpleNamespace

import torch

from src.policy.legendvla_loss import compute_wm_loss


class FakeWorldModelLossModel:
    def __init__(self, pred, target, view_mask, *, mask_loss_by_view_mask=True):
        self.use_world_model = True
        self.world_model_config = SimpleNamespace(mask_loss_by_view_mask=mask_loss_by_view_mask)
        self._pred = pred
        self._target = target
        self._view_mask = view_mask

    def forward_world_model_stream(self, batch, _backbone_output):
        return {
            "pred": self._pred,
            "target": self._target,
            "n_future_frames": batch["n_future_frames"],
            "view_mask": self._view_mask,
        }


def _backbone_output(batch_size):
    return SimpleNamespace(last_hidden_states=torch.zeros(batch_size, 1, 1))


def test_wm_loss_masks_inactive_chest_view():
    pred = torch.zeros(1, 2, 2, 1, 1)
    target = torch.tensor([[[[[1.0]], [[1.0]]], [[[100.0]], [[100.0]]]]])
    view_mask = torch.tensor([[True, False]])
    batch = {
        "future_frames": torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8),
        "n_future_frames": torch.tensor([2]),
    }
    model = FakeWorldModelLossModel(pred, target, view_mask)

    loss = compute_wm_loss(model, batch, _backbone_output(batch_size=1))

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_wm_loss_can_ignore_view_mask_and_supervise_all_views():
    pred = torch.zeros(1, 2, 2, 1, 1)
    target = torch.tensor([[[[[1.0]], [[1.0]]], [[[3.0]], [[3.0]]]]])
    view_mask = torch.tensor([[True, False]])
    batch = {
        "future_frames": torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8),
        "n_future_frames": torch.tensor([2]),
    }
    model = FakeWorldModelLossModel(
        pred,
        target,
        view_mask,
        mask_loss_by_view_mask=False,
    )

    loss = compute_wm_loss(model, batch, _backbone_output(batch_size=1))

    torch.testing.assert_close(loss, torch.tensor(5.0))


def test_wm_loss_masks_inactive_head_view():
    pred = torch.zeros(1, 2, 2, 1, 1)
    target = torch.tensor([[[[[100.0]], [[100.0]]], [[[2.0]], [[2.0]]]]])
    view_mask = torch.tensor([[False, True]])
    batch = {
        "future_frames": torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8),
        "n_future_frames": torch.tensor([2]),
    }
    model = FakeWorldModelLossModel(pred, target, view_mask)

    loss = compute_wm_loss(model, batch, _backbone_output(batch_size=1))

    torch.testing.assert_close(loss, torch.tensor(4.0))


def test_wm_loss_denominator_uses_active_views_and_valid_future_frames():
    pred = torch.zeros(3, 2, 2, 1, 1)
    target = torch.full_like(pred, 100.0)
    target[0, 0, :, :, :] = 1.0
    target[1, 1, :, :, :] = 1.0
    target[2, :, :, :, :] = 1.0
    view_mask = torch.tensor([
        [True, False],
        [False, True],
        [True, True],
    ])
    batch = {
        "future_frames": torch.zeros(3, 2, 4, 4, 3, dtype=torch.uint8),
        "n_future_frames": torch.tensor([2, 1, 0]),
    }
    model = FakeWorldModelLossModel(pred, target, view_mask)

    loss = compute_wm_loss(model, batch, _backbone_output(batch_size=3))

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_wm_loss_zero_when_no_future_frames_are_valid():
    pred = torch.zeros(1, 2, 2, 1, 1)
    target = torch.ones_like(pred)
    view_mask = torch.tensor([[True, True]])
    batch = {
        "future_frames": torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8),
        "n_future_frames": torch.tensor([0]),
    }
    model = FakeWorldModelLossModel(pred, target, view_mask)

    loss = compute_wm_loss(model, batch, _backbone_output(batch_size=1))

    torch.testing.assert_close(loss, torch.tensor(0.0))
