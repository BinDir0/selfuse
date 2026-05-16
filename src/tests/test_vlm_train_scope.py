from __future__ import annotations

import torch
from torch import nn
from omegaconf import OmegaConf

from src.policy.legendvla import LegendVLA
from src.workspace.train_legendvla_workspace import _resolve_vlm_train_scope


class DummyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Linear(2, 2)
        self.model.language_model = nn.Linear(2, 2)


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = DummyBaseModel()
        self.lm_head = nn.Linear(2, 4)
        self.hidden_size = 2
        self.vocab_size = 4
        self.pad_token_id = 0
        self.image_token_id = 1
        self.state_token_id = 2
        self.action_token_id = 3


def make_model() -> LegendVLA:
    return LegendVLA(
        backbone=DummyBackbone(),
        state_encoder=nn.Linear(48, 2),
        action_encoder=nn.Linear(48, 2),
        time_embedding=nn.Linear(1, 2),
        flow_expert=nn.Linear(2, 2),
        action_decoder=nn.Linear(2, 48),
        shape_meta={
            "action": {"shape": [48], "horizon": 32},
            "obs": {"state": {"shape": [48], "horizon": 6}},
        },
        action_hidden_size=2,
    )


def test_freeze_text_weights_keeps_vision_trainable():
    model = make_model()
    visual_params = list(model.backbone.base_model.model.visual.parameters())
    text_params = list(model.backbone.base_model.model.language_model.parameters())
    lm_head_params = list(model.backbone.lm_head.parameters())

    model.freeze_text_weights_in_vlm()

    assert all(param.requires_grad for param in visual_params)
    assert all(not param.requires_grad for param in text_params)
    assert all(not param.requires_grad for param in lm_head_params)
    assert set(map(id, model.trainable_vlm_parameters)) == set(map(id, visual_params))


def test_resolve_vlm_train_scope_honors_legacy_train_vlm_false():
    cfg = OmegaConf.create({"train_vlm": False, "vlm_train_scope": "vision"})
    assert _resolve_vlm_train_scope(cfg) == "none"


def test_resolve_vlm_train_scope_rejects_invalid_values():
    cfg = OmegaConf.create({"train_vlm": True, "vlm_train_scope": "text"})
    try:
        _resolve_vlm_train_scope(cfg)
    except ValueError as exc:
        assert "vlm_train_scope" in str(exc)
    else:
        raise AssertionError("invalid vlm_train_scope should raise ValueError")
