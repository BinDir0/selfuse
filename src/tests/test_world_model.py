"""World model integration tests for future-frame latent supervision."""

from __future__ import annotations

import torch
from torch import nn

from src.model.action.action_head import FourierActionEncoder, MLPProjector
from src.model.common.modules import TimeEmbedding
from src.policy.legendvla import ARActionTrainConfig, FlowConfig, LegendVLA, LossConfig, RTCConfig
from src.model.vlm.prefix_cache import BackboneStreamOutput
from src.tests.dummy_flow_expert import DummyFlowExpert


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 32, vocab_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.image_token_id = 100
        self.state_token_id = 101
        self.action_token_id = 102
        self.future_frame_token_id = 103
        self.num_heads = 2
        self.num_kv_heads = 2
        self.head_dim = 8
        self.tokenizer = type("T", (), {"eos_token_id": 2})()
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.visual = nn.Module()
        self.base_model.model.visual.spatial_merge_size = 1
        self.base_model.model.visual.temporal_patch_size = 1
        self.base_model.model.visual.temporal_patch_size = 1
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.hidden_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        pixel_values_videos,
        video_grid_thw,
        mm_token_type_ids,
        state_slot_embeds,
        action_slot_embeds,
        future_frame_slot_embeds=None,
        state_token_id=None,
        action_token_id=None,
        use_cache=True,
        output_hidden_states=True,
        past_key_values=None,
    ):
        del pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, mm_token_type_ids
        del state_token_id, action_token_id, use_cache, output_hidden_states, past_key_values

        embeds = self.embed(input_ids)
        # State/action slot embeds are 3D (B, T, H) — uniform per sample.
        for token_id, slot_embeds_cur in (
            (self.state_token_id, state_slot_embeds),
            (self.action_token_id, action_slot_embeds),
        ):
            if slot_embeds_cur is None:
                continue
            mask = input_ids == token_id
            slot = mask.long().cumsum(dim=1) - 1
            gather_index = slot.clamp(min=0, max=slot_embeds_cur.shape[1] - 1).unsqueeze(-1).expand(-1, -1, embeds.shape[-1])
            slot_values = torch.gather(slot_embeds_cur, dim=1, index=gather_index)
            embeds = torch.where(mask.unsqueeze(-1), slot_values, embeds)
        # Future frame slot embeds are 2D (total_tokens, H) — variable per
        # sample. Use masked_scatter to match real backbone behavior.
        if future_frame_slot_embeds is not None:
            ff_mask = (input_ids == self.future_frame_token_id).unsqueeze(-1).expand_as(embeds)
            embeds = embeds.masked_scatter(ff_mask, future_frame_slot_embeds.to(embeds.dtype))

        hidden = self.hidden_proj(embeds)
        batch_size, seq_len, _ = hidden.shape
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        key = hidden[..., : self.num_kv_heads * self.head_dim].view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return BackboneStreamOutput(
            last_hidden_states=hidden,
            position_ids=position_ids,
            past_key_values_hf=[(key, key.clone())],
        )


class DummyTargetEncoder(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.encoder_type = "self_vit"
        self.feature_dim = feature_dim
        self.init_called = False
        self.update_calls = 0
        self.register_buffer("basis", torch.linspace(0.1, 1.0, feature_dim), persistent=False)

    def init_ema(self, source_visual: nn.Module, momentum: float = 0.996) -> None:
        del source_visual, momentum
        self.init_called = True

    def update_ema(self, source_visual: nn.Module) -> None:
        del source_visual
        self.update_calls += 1

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = pixel_values.to(dtype=torch.float32).mean(dim=-1, keepdim=True)
        return base + self.basis.unsqueeze(0), grid_thw


class DummyWMDiffLoss(nn.Module):
    def __init__(self, target_channels: int, z_channels: int):
        super().__init__()
        self.proj = nn.Linear(z_channels, target_channels)
        self.last_mask: torch.Tensor | None = None

    def forward(self, target, latent_condition_embeds, mask=None):
        pred = self.proj(latent_condition_embeds)
        loss = (pred - target).pow(2)
        if mask is not None:
            self.last_mask = mask.detach().clone()
            loss = loss * mask.unsqueeze(-1).to(dtype=loss.dtype)
            return loss.sum() / mask.sum().clamp(min=1)
        self.last_mask = None
        return loss.mean()


def build_world_model() -> LegendVLA:
    hidden_size = 32
    action_dim = 12
    state_dim = 12
    action_hidden_size = 16
    backbone = DummyBackbone(hidden_size=hidden_size)
    target_encoder = DummyTargetEncoder(feature_dim=hidden_size)
    return LegendVLA(
        backbone=backbone,
        state_encoder=FourierActionEncoder(
            action_dim=state_dim,
            width=hidden_size,
            time_cond=False,
            enable_fourier_embed=False,
            mlp_depth=2,
            final_layer_norm=False,
            use_mlp_layer_norm=False,
        ),
        ar_action_encoder=FourierActionEncoder(
            action_dim=action_dim,
            width=hidden_size,
            time_cond=False,
            enable_fourier_embed=False,
            mlp_depth=2,
            final_layer_norm=False,
            use_mlp_layer_norm=False,
        ),
        action_encoder=FourierActionEncoder(
            action_dim=action_dim,
            width=action_hidden_size,
            time_cond=False,
            enable_fourier_embed=False,
            mlp_depth=2,
            final_layer_norm=False,
            use_mlp_layer_norm=False,
        ),
        time_embedding=TimeEmbedding(action_hidden_size),
        flow_expert=DummyFlowExpert(hidden_size=action_hidden_size, time_hidden_size=action_hidden_size),
        action_decoder=MLPProjector(
            input_dim=action_hidden_size,
            output_dim=action_dim,
            width=action_hidden_size,
            depth=2,
            final_layer_norm=False,
            use_mlp_layer_norm=False,
        ),
        latent_condition_projector=MLPProjector(
            input_dim=hidden_size,
            output_dim=action_hidden_size,
            width=hidden_size,
            depth=2,
            final_layer_norm=False,
            use_mlp_layer_norm=False,
        ),
        shape_meta={
            "obs": {"state": {"shape": [state_dim], "horizon": 2}},
            "action": {"shape": [action_dim], "horizon": 2},
            "future_frame": {"horizon": 3, "stride": 1},
        },
        diffloss=None,
        action_hidden_size=action_hidden_size,
        flow_config=FlowConfig(num_parallel_t=1, num_inference_steps=2),
        rtc_config=RTCConfig(enabled=False),
        ar_action_train_config=ARActionTrainConfig(chunk_size=2),
        loss_config=LossConfig(ce_loss_weight=0.0, diffusion_loss_weight=0.0, flow_loss_weight=0.0, reg_loss_weight=0.0, wm_loss_weight=1.0),
        target_encoder=target_encoder,
        wm_condition_projector=MLPProjector(
            input_dim=hidden_size,
            output_dim=action_hidden_size,
            width=hidden_size,
            depth=2,
            final_layer_norm=False,
            use_mlp_layer_norm=False,
        ),
        wm_diffloss=DummyWMDiffLoss(target_channels=hidden_size, z_channels=action_hidden_size),
        world_model_cfg={"ff_noise_std": 0.0, "ema_momentum": 0.9},
    )


def build_world_model_batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[100, 10, 101, 101, 102, 102, 103, 103, 103, 0]], dtype=torch.long)
    attention_mask = (input_ids != 0).long()
    action_dim = 12
    # Obs video: 2 temporal patches of 1x1 spatial, patch_dim=4
    obs_pv = torch.tensor([
        [10.0, 10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0, 20.0],
    ], dtype=torch.float32)
    # Future frames: 3 temporal patches of 1x1 spatial, patch_dim=4
    ff_pv = torch.tensor([
        [100.0, 100.0, 100.0, 100.0],
        [200.0, 200.0, 200.0, 200.0],
        [300.0, 300.0, 300.0, 300.0],
    ], dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.full_like(input_ids, -100),
        "pixel_values": torch.randn(1, 3, 16, 16),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        "pixel_values_videos": obs_pv,
        "video_grid_thw": torch.tensor([[2, 1, 1]], dtype=torch.long),
        "mm_token_type_ids": torch.zeros_like(input_ids),
        "states": torch.randn(1, 2, action_dim),
        "actions": torch.randn(1, 2, action_dim),
        "actions_valid_mask": torch.ones(1, 2, action_dim, dtype=torch.bool),
        "answer_start_idx": torch.tensor([4], dtype=torch.long),
        "is_vla_data": torch.tensor([True], dtype=torch.bool),
        "n_states": torch.tensor([2], dtype=torch.long),
        "n_actions": torch.tensor([2], dtype=torch.long),
        "ff_pixel_values": ff_pv,
        "ff_grid_thw": torch.tensor([[3, 1, 1]], dtype=torch.long),
        "ff_video_indices": torch.tensor([0], dtype=torch.long),
        "n_future_frames": torch.tensor([2], dtype=torch.long),
    }


def test_world_model_build_slot_embeddings_exposes_future_frame_slots():
    model = build_world_model()
    batch = build_world_model_batch()

    slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)

    assert model.target_encoder.init_called
    assert slot_embeds["future_frame"] is not None
    # future_frame slot embeds are 2D flat (total_tokens, hidden) to support
    # variable per-sample token counts with masked_scatter in the backbone.
    assert slot_embeds["future_frame"].shape == (3, model.vlm_hidden_size)
    assert batch["_wm_target_features"].shape == (3, model.vlm_hidden_size)


def test_world_model_train_forward_backward_updates_world_model_parameters():
    model = build_world_model()
    batch = build_world_model_batch()

    output = model("train", batch)
    assert output["wm_loss"].item() > 0
    torch.testing.assert_close(output["total_loss"], output["wm_loss"])

    output["total_loss"].backward()

    assert model.wm_diffloss.last_mask is not None
    assert model.wm_diffloss.last_mask.tolist() == [1.0, 1.0, 0.0]
    assert any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in model.world_model_parameters
    )
    assert all(param.grad is None for param in model.target_encoder.parameters())


def test_world_model_non_train_modes_keep_wm_loss_zero():
    model = build_world_model()
    batch = build_world_model_batch()

    ar_output = model("train_ar", batch)
    flow_output = model("train_flow", batch)

    assert ar_output["wm_loss"].item() == 0.0
    assert flow_output["wm_loss"].item() == 0.0


def test_world_model_update_ema_delegates_to_target_encoder():
    model = build_world_model()

    model.update_ema()

    assert model.target_encoder.update_calls == 1


# ======================================================================
# encode_world_model_targets: concat ordering and split
# ======================================================================


def test_encode_world_model_targets_concat_ordering():
    """Verify obs+future concat ordering and that only future features are returned."""
    model = build_world_model()
    batch = build_world_model_batch()

    target_features, ff_only_grid = model.encode_world_model_targets(batch)

    # 3 ff patches of 1x1 spatial with sms=1 → 3 merged tokens
    assert target_features.shape == (3, model.vlm_hidden_size)
    assert ff_only_grid.tolist() == [[3, 1, 1]]

    # DummyTargetEncoder: output[i] = mean(input[i]) + basis
    # Combined input = [obs(10), obs(20), ff(100), ff(200), ff(300)]
    # After split, target_features should contain only ff-derived features.
    basis = model.target_encoder.basis
    torch.testing.assert_close(target_features[0], 100.0 + basis)
    torch.testing.assert_close(target_features[1], 200.0 + basis)
    torch.testing.assert_close(target_features[2], 300.0 + basis)


def test_encode_world_model_targets_multi_sample():
    """Verify per-sample concat and split with multiple obs/ff entries."""
    model = build_world_model()

    # 2 obs videos with different temporal lengths
    obs_pv = torch.tensor([
        [10.0, 10.0, 10.0, 10.0],   # obs video 0, patch 0
        [20.0, 20.0, 20.0, 20.0],   # obs video 0, patch 1
        [30.0, 30.0, 30.0, 30.0],   # obs video 1, patch 0
    ], dtype=torch.float32)

    # 2 ff entries mapping to different videos
    ff_pv = torch.tensor([
        [100.0, 100.0, 100.0, 100.0],  # ff entry 0 (→ video 0), 1 patch
        [200.0, 200.0, 200.0, 200.0],  # ff entry 1 (→ video 1), patch 0
        [300.0, 300.0, 300.0, 300.0],  # ff entry 1 (→ video 1), patch 1
    ], dtype=torch.float32)

    batch = {
        "pixel_values_videos": obs_pv,
        "video_grid_thw": torch.tensor([[2, 1, 1], [1, 1, 1]], dtype=torch.long),
        "ff_pixel_values": ff_pv,
        "ff_grid_thw": torch.tensor([[1, 1, 1], [2, 1, 1]], dtype=torch.long),
        "ff_video_indices": torch.tensor([0, 1], dtype=torch.long),
    }

    target_features, ff_only_grid = model.encode_world_model_targets(batch)

    # Entry 0: combined [obs0_p0, obs0_p1, ff0_p0] → 3 patches, obs=2, ff=1
    # Entry 1: combined [obs1_p0, ff1_p0, ff1_p1] → 3 patches, obs=1, ff=2
    # Total ff tokens: 1 + 2 = 3
    assert target_features.shape == (3, model.vlm_hidden_size)
    assert ff_only_grid.tolist() == [[1, 1, 1], [2, 1, 1]]

    # Verify feature values match ff inputs (not obs inputs)
    basis = model.target_encoder.basis
    torch.testing.assert_close(target_features[0], 100.0 + basis)  # from ff entry 0
    torch.testing.assert_close(target_features[1], 200.0 + basis)  # from ff entry 1, patch 0
    torch.testing.assert_close(target_features[2], 300.0 + basis)  # from ff entry 1, patch 1


def test_encode_world_model_targets_non_contiguous_video_indices():
    """ff_video_indices can skip video entries (e.g., when only some VLA samples have future frames)."""
    model = build_world_model()

    # 3 obs videos, but only videos 0 and 2 have future frames
    obs_pv = torch.tensor([
        [10.0, 10.0, 10.0, 10.0],   # obs video 0
        [20.0, 20.0, 20.0, 20.0],   # obs video 1
        [30.0, 30.0, 30.0, 30.0],   # obs video 2
    ], dtype=torch.float32)

    ff_pv = torch.tensor([
        [100.0, 100.0, 100.0, 100.0],  # ff entry 0 → video 0
        [300.0, 300.0, 300.0, 300.0],  # ff entry 1 → video 2
    ], dtype=torch.float32)

    batch = {
        "pixel_values_videos": obs_pv,
        "video_grid_thw": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.long),
        "ff_pixel_values": ff_pv,
        "ff_grid_thw": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
        "ff_video_indices": torch.tensor([0, 2], dtype=torch.long),
    }

    target_features, ff_only_grid = model.encode_world_model_targets(batch)

    assert target_features.shape == (2, model.vlm_hidden_size)
    assert ff_only_grid.tolist() == [[1, 1, 1], [1, 1, 1]]

    # Entry 0 concat: [obs_video0(10), ff0(100)] → ff feature from 100
    # Entry 1 concat: [obs_video2(30), ff1(300)] → ff feature from 300
    # Obs video 1 (20) is never used because no ff maps to it.
    basis = model.target_encoder.basis
    torch.testing.assert_close(target_features[0], 100.0 + basis)
    torch.testing.assert_close(target_features[1], 300.0 + basis)
