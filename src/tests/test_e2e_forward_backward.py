"""
End-to-end forward and backward verification for LegendVLA.

Tests the full training loop: forward → loss → backward → gradient check,
covering all three training modes and all inference modes.
Does NOT require downloading the real Qwen3-VL model.
"""

import torch
from torch import nn

from src.policy.legendvla import (
    LegendVLA, FlowConfig, RTCConfig, LossConfig, ARActionTrainConfig,
    WorldModelConfig,
)
from src.policy.legendvla_loss import build_flow_inputs
from src.model.action.action_head import FourierActionEncoder, MLPProjector
from src.model.common.modules import TimeEmbedding
from src.model.vlm.prefix_cache import BackboneStreamOutput
from src.tests.dummy_flow_expert import DummyFlowExpert


# ======================================================================
# DummyBackbone and DummyDiffLoss (matching real interface)
# ======================================================================


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size=64, vocab_size=128, num_layers=8,
                 num_heads=4, num_kv_heads=4, head_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.image_token_id = 100
        self.state_token_id = 101
        self.action_token_id = 102
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.tokenizer = type("T", (), {"eos_token_id": 2})()
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.visual = nn.Module()
        self.base_model.model.visual.spatial_merge_size = 1
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
        camera_slot_embeds=None,
        output_attentions=False,
        state_token_id=None,
        action_token_id=None,
        use_cache=True,
        output_hidden_states=True,
        past_key_values=None,
        is_vla_mask=None,
    ):
        del pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, mm_token_type_ids
        del camera_slot_embeds, output_attentions
        del state_token_id, action_token_id, use_cache, output_hidden_states, past_key_values
        del is_vla_mask

        embeds = self.embed(input_ids)
        if state_slot_embeds is not None:
            mask = input_ids == self.state_token_id
            slot = mask.long().cumsum(dim=1) - 1
            idx = slot.clamp(min=0, max=state_slot_embeds.shape[1] - 1).unsqueeze(-1).expand(-1, -1, embeds.shape[-1])
            embeds = torch.where(mask.unsqueeze(-1), torch.gather(state_slot_embeds, 1, idx), embeds)
        if action_slot_embeds is not None:
            mask = input_ids == self.action_token_id
            slot = mask.long().cumsum(dim=1) - 1
            idx = slot.clamp(min=0, max=action_slot_embeds.shape[1] - 1).unsqueeze(-1).expand(-1, -1, embeds.shape[-1])
            embeds = torch.where(mask.unsqueeze(-1), torch.gather(action_slot_embeds, 1, idx), embeds)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        hidden = self.hidden_proj(embeds)
        B, L, _ = hidden.shape
        cache = []
        for _ in range(self.num_layers):
            k = hidden[..., :self.num_kv_heads * self.head_dim].view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
            cache.append((k, k.clone()))
        return BackboneStreamOutput(
            last_hidden_states=hidden,
            position_ids=position_ids,
            past_key_values_hf=cache,
        )


class DummyDiffLoss(nn.Module):
    def __init__(self, target_channels, z_channels):
        super().__init__()
        self.proj = nn.Linear(z_channels, target_channels)
        self.num_inference_steps = 5

    def forward(self, target, latent_condition_embeds, mask=None):
        pred = self.proj(latent_condition_embeds)
        loss = (pred - target).pow(2)
        if mask is not None:
            loss = loss * mask.unsqueeze(-1).to(dtype=loss.dtype)
        return loss.mean()

    def sample(self, latent_condition, temperature=1.0, cfg=1.0):
        del temperature, cfg
        return self.proj(latent_condition)


# ======================================================================
# Helpers
# ======================================================================

H = 64          # vlm hidden
AH = 32         # action expert hidden
TH = 32         # time hidden
NH = 4          # num heads
NKV = 4         # num kv heads
HD = 16         # head dim (must satisfy: NH * HD can be produced by q_proj)
AD = 48         # action dim
SD = 48         # state dim
LAYERS = 4


def build_model(with_diffloss=False, knowledge_insulation=True, num_parallel_t=1, rtc_config=None):
    backbone = DummyBackbone(hidden_size=H, num_layers=LAYERS,
                             num_heads=NH, num_kv_heads=NKV, head_dim=HD)
    diffloss = DummyDiffLoss(target_channels=AD * 2, z_channels=AH) if with_diffloss else None
    if rtc_config is None:
        rtc_config = RTCConfig()

    return LegendVLA(
        backbone=backbone,
        state_encoder=FourierActionEncoder(action_dim=SD, width=H, time_cond=False,
                                           enable_fourier_embed=False, mlp_depth=2,
                                           final_layer_norm=False, use_mlp_layer_norm=False),
        ar_action_encoder=FourierActionEncoder(action_dim=AD, width=H, time_cond=False,
                                               enable_fourier_embed=False, mlp_depth=2,
                                               final_layer_norm=False, use_mlp_layer_norm=False),
        action_encoder=FourierActionEncoder(action_dim=AD, width=AH, time_cond=False,
                                            enable_fourier_embed=False, mlp_depth=2,
                                            final_layer_norm=False, use_mlp_layer_norm=False),
        time_embedding=TimeEmbedding(TH),
        flow_expert=DummyFlowExpert(hidden_size=AH, time_hidden_size=TH),
        action_decoder=MLPProjector(input_dim=AH, output_dim=AD, width=AH, depth=2,
                                    final_layer_norm=False, use_mlp_layer_norm=False),
        latent_condition_projector=MLPProjector(input_dim=H, output_dim=AH, width=H, depth=2,
                                                final_layer_norm=False, use_mlp_layer_norm=False),
        shape_meta={"obs": {"state": {"shape": [SD], "horizon": 2}}, "action": {"shape": [AD], "horizon": 4}},
        diffloss=diffloss,
        action_hidden_size=AH,
        flow_config=FlowConfig(num_parallel_t=num_parallel_t, num_inference_steps=3),
        ar_action_train_config=ARActionTrainConfig(chunk_size=2),
        rtc_config=rtc_config,
        loss_config=LossConfig(),
        knowledge_insulation=knowledge_insulation,
    )


def build_batch(batch_size=2):
    # Sample 0: VLA, Sample 1: VLM-only
    input_ids = torch.tensor([
        [100, 10, 11, 101, 101, 102, 102, 102, 102, 0],
        [100, 12, 13, 14, 15, 16, 0, 0, 0, 0],
    ])[:batch_size]
    attention_mask = (input_ids != 0).long()
    labels = torch.full_like(input_ids, -100)
    if batch_size > 1:
        labels[1, 1:6] = input_ids[1, 1:6]
    labels[attention_mask == 0] = -100

    actions_valid_mask = torch.zeros(batch_size, 4, AD, dtype=torch.bool)
    if batch_size > 0:
        actions_valid_mask[0] = True

    states = torch.randn(batch_size, 2, SD)
    actions = torch.randn(batch_size, 4, AD)
    if batch_size > 1:
        states[1] = 0
        actions[1] = 0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": torch.randn(batch_size, 3, 16, 16),
        "image_grid_thw": torch.ones(batch_size, 3, dtype=torch.long),
        "pixel_values_videos": None,
        "video_grid_thw": None,
        "mm_token_type_ids": torch.zeros(batch_size, input_ids.shape[1], dtype=torch.long),
        "states": states,
        "actions": actions,
        "actions_valid_mask": actions_valid_mask,
        "answer_start_idx": torch.tensor([5, 6], dtype=torch.long)[:batch_size],
        "is_vla_data": torch.tensor([True, False], dtype=torch.bool)[:batch_size],
        "n_states": torch.tensor([2, 0], dtype=torch.long)[:batch_size],
        "n_actions": torch.tensor([4, 0], dtype=torch.long)[:batch_size],
    }


# ======================================================================
# Tests
# ======================================================================


class TestEndToEndForwardBackward:
    """Full forward + backward through every training mode."""

    def test_train_mode_forward_backward(self):
        """mode='train': CE + DiffLoss + Flow, backward produces gradients."""
        model = build_model(with_diffloss=True)
        batch = build_batch()
        output = model("train", batch)

        assert "total_loss" in output
        assert "ce_loss" in output
        assert "diffusion_loss" in output
        assert "reg_loss" in output
        assert "flow_loss" in output
        assert output["total_loss"].requires_grad
        assert output["ce_loss"].item() > 0
        assert output["diffusion_loss"].item() > 0
        assert output["flow_loss"].item() > 0

        output["total_loss"].backward()

        # Verify gradients propagated to all parameter groups
        has_backbone_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.trainable_vlm_parameters)
        has_expert_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                              for p in model.action_expert_parameters)
        has_diffloss_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.ar_action_heads_parameters)
        assert has_backbone_grad, "No gradient reached backbone parameters"
        assert has_expert_grad, "No gradient reached action expert parameters"
        assert has_diffloss_grad, "No gradient reached diffloss parameters"

    def test_train_mode_without_diffloss(self):
        """mode='train' without DiffLoss should keep diffusion loss at zero
        and not instantiate the diffloss-specific head modules.

        (state_encoder is part of ar_action_heads_parameters but always gets
        gradients through flow_loss / ce_loss via the backbone state slot, so
        it is NOT a signal for diffloss activity.)
        """
        model = build_model(with_diffloss=False)
        batch = build_batch()
        output = model("train", batch)

        assert output["total_loss"].requires_grad
        assert output["diffusion_loss"].item() == 0.0
        output["total_loss"].backward()

        # use_diffloss gates the whole AR-action-via-diffloss path; without
        # the diffloss module the trio is effectively disabled even if
        # ar_action_encoder is still instantiated for other heads.
        assert not model.use_diffloss
        assert model.diffloss is None

    def test_train_ar_mode(self):
        """mode='train_ar': only CE + DiffLoss, no flow."""
        model = build_model(with_diffloss=True)
        batch = build_batch()
        output = model("train_ar", batch)

        assert output["total_loss"].requires_grad
        assert output["flow_loss"].item() == 0.0
        output["total_loss"].backward()

    def test_train_flow_mode(self):
        """mode='train_flow': only flow loss, no CE."""
        model = build_model(with_diffloss=False)
        batch = build_batch()
        output = model("train_flow", batch)

        assert output["total_loss"].requires_grad
        assert output["ce_loss"].item() == 0.0
        output["total_loss"].backward()

        # Flow expert should have gradients
        expert_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                              for p in model.flow_expert.parameters())
        assert expert_has_grad, "No gradient reached flow expert in train_flow mode"

    def test_train_flow_mode_parallel_timesteps(self):
        """Parallel flow denoising should keep the flow-only path finite and differentiable."""
        model = build_model(with_diffloss=False, num_parallel_t=4)
        batch = build_batch(batch_size=1)
        output = model("train_flow", batch)

        assert output["total_loss"].requires_grad
        assert torch.isfinite(output["flow_loss"])
        output["total_loss"].backward()

        expert_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                              for p in model.flow_expert.parameters())
        assert expert_has_grad, "No gradient reached flow expert with parallel flow timesteps enabled"

    def test_build_flow_inputs_normalizes_single_t_shape_without_rtc(self):
        model = build_model(with_diffloss=False, rtc_config=RTCConfig(enabled=False))
        batch = build_batch(batch_size=1)
        sampled_t = torch.tensor([0.37], dtype=batch["actions"].dtype)
        sampled_t_2d = sampled_t.unsqueeze(1)

        torch.manual_seed(123)
        single_inputs = build_flow_inputs(model, batch, num_parallel_t=1, sampled_t=sampled_t)
        torch.manual_seed(123)
        same_inputs = build_flow_inputs(model, batch, num_parallel_t=1, sampled_t=sampled_t_2d)

        torch.testing.assert_close(single_inputs["noise"], same_inputs["noise"])
        torch.testing.assert_close(single_inputs["noisy_actions"], same_inputs["noisy_actions"])
        torch.testing.assert_close(single_inputs["time_for_model"], same_inputs["time_for_model"])
        torch.testing.assert_close(single_inputs["loss_mask"], same_inputs["loss_mask"])

    def test_build_flow_inputs_normalizes_single_t_shape_with_rtc(self):
        model = build_model(with_diffloss=False, rtc_config=RTCConfig(enabled=True, delay_strategy="uniform", max_delay=4))
        batch = build_batch(batch_size=1)
        sampled_t = torch.tensor([0.37], dtype=batch["actions"].dtype)
        sampled_t_2d = sampled_t.unsqueeze(1)

        torch.manual_seed(123)
        single_inputs = build_flow_inputs(model, batch, num_parallel_t=1, sampled_t=sampled_t)
        torch.manual_seed(123)
        same_inputs = build_flow_inputs(model, batch, num_parallel_t=1, sampled_t=sampled_t_2d)

        torch.testing.assert_close(single_inputs["noise"], same_inputs["noise"])
        torch.testing.assert_close(single_inputs["noisy_actions"], same_inputs["noisy_actions"])
        torch.testing.assert_close(single_inputs["time_for_model"], same_inputs["time_for_model"])
        torch.testing.assert_close(single_inputs["loss_mask"], same_inputs["loss_mask"])

    def test_parallel_flow_repeats_action_position_ids_per_chunk(self):
        model = build_model(with_diffloss=False, num_parallel_t=4)
        batch = build_batch(batch_size=1)
        slot_embeds = model.build_slot_embeddings(batch)
        backbone_output = model.forward_backbone_stream(batch, slot_embeds)
        flow_inputs = build_flow_inputs(model, batch, num_parallel_t=4)

        model.forward_flow_stream(
            batch=batch,
            backbone_output=backbone_output,
            flow_inputs=flow_inputs,
            num_parallel_chunks=4,
        )

        base_position_ids = model.build_action_position_ids(batch, batch["actions"])
        expected_position_ids = base_position_ids.repeat(1, 4)
        torch.testing.assert_close(model.flow_expert.last_action_position_ids, expected_position_ids)
        assert model.flow_expert.last_num_parallel_chunks == 4

    def test_parallel_flow_matches_separate_single_t_forward_values(self):
        model = build_model(
            with_diffloss=False,
            num_parallel_t=4,
            rtc_config=RTCConfig(enabled=False),
        )
        model.eval()
        batch = build_batch(batch_size=1)
        slot_embeds = model.build_slot_embeddings(batch)
        backbone_output = model.forward_backbone_stream(batch, slot_embeds)
        sampled_t = torch.tensor([[0.15, 0.35, 0.55, 0.75]], dtype=batch["actions"].dtype)
        packed_inputs = build_flow_inputs(model, batch, num_parallel_t=4, sampled_t=sampled_t)
        packed_output = model.forward_flow_stream(
            batch=batch,
            backbone_output=backbone_output,
            flow_inputs=packed_inputs,
            num_parallel_chunks=4,
        )

        horizon = batch["actions"].shape[1]
        for chunk_idx in range(4):
            chunk_start = chunk_idx * horizon
            chunk_end = chunk_start + horizon
            single_inputs = {
                "actions": packed_inputs["actions"][:, chunk_start:chunk_end],
                "noise": packed_inputs["noise"][:, chunk_start:chunk_end],
                "noisy_actions": packed_inputs["noisy_actions"][:, chunk_start:chunk_end],
                "time_for_model": packed_inputs["time_for_model"][:, chunk_start:chunk_end],
                "loss_mask": packed_inputs["loss_mask"][:, chunk_start:chunk_end],
            }
            single_output = model.forward_flow_stream(
                batch=batch,
                backbone_output=backbone_output,
                flow_inputs=single_inputs,
                num_parallel_chunks=1,
            )
            torch.testing.assert_close(
                packed_output["action_hidden_states"][:, chunk_start:chunk_end],
                single_output["action_hidden_states"],
            )
            torch.testing.assert_close(
                packed_output["pred_v"][:, chunk_start:chunk_end],
                single_output["pred_v"],
            )

    def test_parallel_flow_matches_separate_single_t_forward_values_with_rtc(self):
        model = build_model(
            with_diffloss=False,
            num_parallel_t=3,
            rtc_config=RTCConfig(enabled=True, delay_strategy="uniform", max_delay=4),
        )
        model.eval()
        batch = build_batch(batch_size=1)
        slot_embeds = model.build_slot_embeddings(batch)
        backbone_output = model.forward_backbone_stream(batch, slot_embeds)
        sampled_t = torch.tensor([[0.2, 0.5, 0.8]], dtype=batch["actions"].dtype)
        torch.manual_seed(123)
        packed_inputs = build_flow_inputs(model, batch, num_parallel_t=3, sampled_t=sampled_t)
        packed_output = model.forward_flow_stream(
            batch=batch,
            backbone_output=backbone_output,
            flow_inputs=packed_inputs,
            num_parallel_chunks=3,
        )

        horizon = batch["actions"].shape[1]
        for chunk_idx in range(3):
            chunk_start = chunk_idx * horizon
            chunk_end = chunk_start + horizon
            single_inputs = {
                "actions": packed_inputs["actions"][:, chunk_start:chunk_end],
                "noise": packed_inputs["noise"][:, chunk_start:chunk_end],
                "noisy_actions": packed_inputs["noisy_actions"][:, chunk_start:chunk_end],
                "time_for_model": packed_inputs["time_for_model"][:, chunk_start:chunk_end],
                "loss_mask": packed_inputs["loss_mask"][:, chunk_start:chunk_end],
            }
            single_output = model.forward_flow_stream(
                batch=batch,
                backbone_output=backbone_output,
                flow_inputs=single_inputs,
                num_parallel_chunks=1,
            )
            torch.testing.assert_close(
                packed_output["action_hidden_states"][:, chunk_start:chunk_end],
                single_output["action_hidden_states"],
            )
            torch.testing.assert_close(
                packed_output["pred_v"][:, chunk_start:chunk_end],
                single_output["pred_v"],
            )

    def test_train_flow_mode_knowledge_insulation_blocks_backbone_gradients(self):
        """Knowledge insulation should stop flow gradients from reaching the backbone through prefix KV."""
        batch = build_batch(batch_size=1)

        model_without_insulation = build_model(with_diffloss=False, knowledge_insulation=False)
        output_without_insulation = model_without_insulation("train_flow", batch)
        output_without_insulation["total_loss"].backward()
        has_backbone_grad_without_insulation = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model_without_insulation.trainable_vlm_parameters
        )
        assert has_backbone_grad_without_insulation, (
            "Expected train_flow gradients to reach backbone without knowledge insulation"
        )

        insulated_batch = build_batch(batch_size=1)
        model_with_insulation = build_model(with_diffloss=False, knowledge_insulation=True)
        output_with_insulation = model_with_insulation("train_flow", insulated_batch)
        output_with_insulation["total_loss"].backward()
        has_backbone_grad_with_insulation = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model_with_insulation.trainable_vlm_parameters
        )
        assert not has_backbone_grad_with_insulation, (
            "Knowledge insulation should block train_flow gradients from reaching backbone"
        )

    def test_gradient_isolation_between_groups(self):
        """Verify parameter group isolation: freezing one group doesn't kill gradients in others."""
        model = build_model(with_diffloss=False)
        model.freeze_non_lora_weights_in_vlm()
        batch = build_batch()
        output = model("train", batch)
        output["total_loss"].backward()

        # Backbone should be frozen
        assert all(p.grad is None or p.grad.abs().sum() == 0
                   for p in model.backbone.parameters()), "Frozen backbone should have no gradients"
        # Expert should still have gradients
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.action_expert_parameters), "Expert should have gradients when backbone frozen"

    def test_two_consecutive_steps(self):
        """Simulate two training steps to verify state doesn't corrupt."""
        model = build_model(with_diffloss=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for step in range(2):
            batch = build_batch()
            optimizer.zero_grad()
            output = model("train", batch)
            output["total_loss"].backward()
            optimizer.step()
            assert torch.isfinite(output["total_loss"]), f"Non-finite loss at step {step}"


class TestEndToEndInference:
    """All inference modes produce correct shapes without error."""

    def test_flow_inference(self):
        model = build_model()
        model.eval()
        batch = build_batch(batch_size=1)
        with torch.no_grad():
            actions = model("infer_action", batch)
        assert actions.shape == (1, 4, AD)
        assert torch.isfinite(actions).all()

    def test_flow_inference_with_rtc(self):
        """Flow inference with RTC: prev_action_chunk should pin prefix steps."""
        model = build_model()
        model.eval()
        batch = build_batch(batch_size=1)
        prev_chunk = torch.randn(1, 4, AD)
        with torch.no_grad():
            actions = model("infer_action", batch,
                            prev_action_chunk=prev_chunk, inference_delay=2)
        assert actions.shape == (1, 4, AD)
        # First 2 steps should be pinned to prev_chunk
        torch.testing.assert_close(actions[:, :2], prev_chunk[:, :2])

    def test_ar_inference(self):
        model = build_model(with_diffloss=True)
        model.eval()
        batch = build_batch(batch_size=1)
        with torch.no_grad():
            result = model("infer_vla", batch, max_new_tokens=4)
        assert result["generated_actions"].shape == (1, 4, AD)
        assert result["generated_hidden_states"] is not None

    def test_vlm_inference(self):
        model = build_model()
        model.eval()
        batch = {
            "input_ids": torch.tensor([[100, 10, 11, 12]], dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.tensor([[100, 10, 11, 12]], dtype=torch.long),
            "pixel_values": torch.randn(1, 3, 16, 16),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            "pixel_values_videos": None,
            "video_grid_thw": None,
            "mm_token_type_ids": torch.zeros(1, 4, dtype=torch.long),
            "answer_start_idx": torch.tensor([4], dtype=torch.long),
            "is_vla_data": torch.tensor([False]),
            "n_states": torch.tensor([0], dtype=torch.long),
            "n_actions": torch.tensor([0], dtype=torch.long),
        }
        with torch.no_grad():
            result = model("infer_vlm", batch, max_new_tokens=3)
        assert result["generated_ids"].shape == (1, 3)


class TestParameterGroups:
    """Verify parameter group properties used by the optimizer."""

    def test_parameter_groups_are_disjoint(self):
        """No parameter should appear in more than one group."""
        model = build_model(with_diffloss=True)
        vlm_ids = {id(p) for p in model.trainable_vlm_parameters}
        expert_ids = {id(p) for p in model.action_expert_parameters}
        diffloss_ids = {id(p) for p in model.ar_action_heads_parameters}

        assert vlm_ids.isdisjoint(expert_ids), "VLM and expert params overlap"
        assert vlm_ids.isdisjoint(diffloss_ids), "VLM and diffloss params overlap"
        assert expert_ids.isdisjoint(diffloss_ids), "Expert and diffloss params overlap"

    def test_all_trainable_params_covered(self):
        """Every trainable parameter should be in exactly one group."""
        model = build_model(with_diffloss=True)
        all_trainable = {id(p) for p in model.parameters() if p.requires_grad}
        grouped = set()
        grouped.update(id(p) for p in model.trainable_vlm_parameters)
        grouped.update(id(p) for p in model.action_expert_parameters)
        grouped.update(id(p) for p in model.ar_action_heads_parameters)

        uncovered = all_trainable - grouped
        assert len(uncovered) == 0, (
            f"{len(uncovered)} trainable params not covered by any optimizer group"
        )

    def test_freeze_vlm_only_affects_backbone(self):
        model = build_model()
        before_expert = sum(p.requires_grad for p in model.action_expert_parameters)
        model.freeze_non_lora_weights_in_vlm()
        after_expert = sum(p.requires_grad for p in model.action_expert_parameters)
        assert before_expert == after_expert
        assert all(not p.requires_grad for p in model.backbone.parameters())

    def test_freeze_ae_only_affects_action_modules(self):
        """freeze_non_lora_weights_in_ae freezes the flow action-expert stack
        (state_encoder, action_encoder, time_embedding, flow_expert,
        action_decoder). ar_action_encoder belongs to the AR/diffloss head
        attached to the backbone and is intentionally NOT in this scope.
        """
        model = build_model()
        before_vlm = sum(p.requires_grad for p in model.trainable_vlm_parameters)
        model.freeze_non_lora_weights_in_ae()
        after_vlm = sum(p.requires_grad for p in model.trainable_vlm_parameters)
        assert before_vlm == after_vlm
        assert all(not p.requires_grad for p in model.state_encoder.parameters())
        assert all(not p.requires_grad for p in model.action_encoder.parameters())
        assert all(not p.requires_grad for p in model.flow_expert.parameters())


class TestLossValues:
    """Sanity checks on loss magnitudes and structure."""

    def test_loss_dict_keys(self):
        model = build_model()
        batch = build_batch()
        output = model("train", batch)
        assert set(output.keys()) == {"total_loss", "ce_loss", "diffusion_loss", "reg_loss", "flow_loss", "wm_loss"}

    def test_total_loss_is_weighted_sum(self):
        """total_loss should equal weighted combination of sub-losses."""
        model = build_model()
        batch = build_batch()
        output = model("train", batch)
        w = model.loss_config
        expected = (w.ce_loss_weight * output["ce_loss"]
                    + w.diffusion_loss_weight * output["diffusion_loss"]
                    + w.reg_loss_weight * output["reg_loss"]
                    + w.flow_loss_weight * output["flow_loss"])
        torch.testing.assert_close(output["total_loss"], expected, rtol=1e-4, atol=1e-6)

    def test_all_losses_finite(self):
        model = build_model(with_diffloss=True)
        batch = build_batch()
        output = model("train", batch)
        for key, value in output.items():
            assert torch.isfinite(value), f"{key} is not finite: {value}"

    def test_vlm_only_batch_produces_ce_loss_only(self):
        """A batch with is_vla_data=False should produce CE loss but zero flow/diffusion."""
        model = build_model()
        batch = build_batch()
        batch["is_vla_data"] = torch.tensor([False, False], dtype=torch.bool)
        output = model("train", batch)
        assert output["flow_loss"].item() == 0.0
        assert output["diffusion_loss"].item() == 0.0
        assert output["wm_loss"].item() == 0.0


# ======================================================================
# World model: dedicated encoders + motion conditioning
# ======================================================================


def build_wm_enabled_model(action_conditioning=False, motion_conditioning=False):
    """Minimal LegendVLA with world model + optional conditioning for config-wiring tests."""
    from src.model.action.action_head import ActionEncoder, MLPProjector

    backbone = DummyBackbone(hidden_size=H, num_layers=LAYERS,
                             num_heads=NH, num_kv_heads=NKV, head_dim=HD)
    wm_expert = DummyFlowExpert(hidden_size=AH, time_hidden_size=TH)
    teacher = nn.Linear(AD, 32)  # stand-in; not invoked in these tests
    wm_action_encoder = (
        ActionEncoder(action_dim=AD, width=AH, time_cond=False)
        if action_conditioning else None
    )
    wm_motion_encoder = (
        MLPProjector(input_dim=16, output_dim=AH, width=AH, depth=2,
                     final_layer_norm=False, use_mlp_layer_norm=False)
        if motion_conditioning else None
    )
    return LegendVLA(
        backbone=backbone,
        state_encoder=FourierActionEncoder(action_dim=SD, width=H, time_cond=False,
                                           enable_fourier_embed=False, mlp_depth=2,
                                           final_layer_norm=False, use_mlp_layer_norm=False),
        action_encoder=FourierActionEncoder(action_dim=AD, width=AH, time_cond=False,
                                            enable_fourier_embed=False, mlp_depth=2,
                                            final_layer_norm=False, use_mlp_layer_norm=False),
        time_embedding=TimeEmbedding(TH),
        flow_expert=DummyFlowExpert(hidden_size=AH, time_hidden_size=TH),
        action_decoder=MLPProjector(input_dim=AH, output_dim=AD, width=AH, depth=2,
                                    final_layer_norm=False, use_mlp_layer_norm=False),
        shape_meta={"obs": {"state": {"shape": [SD], "horizon": 2}}, "action": {"shape": [AD], "horizon": 4}},
        action_hidden_size=AH,
        flow_config=FlowConfig(num_parallel_t=1, num_inference_steps=3),
        ar_action_train_config=ARActionTrainConfig(chunk_size=2),
        rtc_config=RTCConfig(),
        loss_config=LossConfig(),
        knowledge_insulation=True,
        world_model_expert=wm_expert,
        frozen_teacher=teacher,
        wm_action_encoder=wm_action_encoder,
        wm_motion_encoder=wm_motion_encoder,
        world_model_config=WorldModelConfig(
            num_future_frames=2, target_image_size=(32, 32),
            teacher_patch_size=16, upsample_factor=1,
            action_conditioning=action_conditioning,
            motion_conditioning=motion_conditioning,
        ),
    )


class TestWorldModelDedicatedEncoders:
    def test_wm_action_encoder_is_separate_module_from_flow_action_encoder(self):
        model = build_wm_enabled_model(action_conditioning=True)
        assert model.wm_action_encoder is not None
        assert model.wm_action_encoder is not model.action_encoder

    def test_action_conditioning_without_encoder_raises(self):
        from src.model.action.action_head import ActionEncoder
        import pytest
        with pytest.raises(ValueError, match="wm_action_encoder"):
            LegendVLA(
                backbone=DummyBackbone(hidden_size=H, num_layers=LAYERS,
                                       num_heads=NH, num_kv_heads=NKV, head_dim=HD),
                state_encoder=FourierActionEncoder(action_dim=SD, width=H, time_cond=False,
                                                   enable_fourier_embed=False, mlp_depth=2,
                                                   final_layer_norm=False, use_mlp_layer_norm=False),
                action_encoder=ActionEncoder(action_dim=AD, width=AH, time_cond=False),
                time_embedding=TimeEmbedding(TH),
                flow_expert=DummyFlowExpert(hidden_size=AH, time_hidden_size=TH),
                action_decoder=MLPProjector(input_dim=AH, output_dim=AD, width=AH, depth=2,
                                            final_layer_norm=False, use_mlp_layer_norm=False),
                shape_meta={"obs": {"state": {"shape": [SD], "horizon": 2}}, "action": {"shape": [AD], "horizon": 4}},
                action_hidden_size=AH,
                world_model_expert=DummyFlowExpert(hidden_size=AH, time_hidden_size=TH),
                frozen_teacher=nn.Linear(AD, 32),
                wm_action_encoder=None,
                world_model_config=WorldModelConfig(
                    num_future_frames=2, target_image_size=(32, 32),
                    teacher_patch_size=16, upsample_factor=1,
                    action_conditioning=True,
                ),
            )

    def test_motion_conditioning_without_encoder_raises(self):
        from src.model.action.action_head import ActionEncoder
        import pytest
        with pytest.raises(ValueError, match="wm_motion_encoder"):
            LegendVLA(
                backbone=DummyBackbone(hidden_size=H, num_layers=LAYERS,
                                       num_heads=NH, num_kv_heads=NKV, head_dim=HD),
                state_encoder=FourierActionEncoder(action_dim=SD, width=H, time_cond=False,
                                                   enable_fourier_embed=False, mlp_depth=2,
                                                   final_layer_norm=False, use_mlp_layer_norm=False),
                action_encoder=ActionEncoder(action_dim=AD, width=AH, time_cond=False),
                time_embedding=TimeEmbedding(TH),
                flow_expert=DummyFlowExpert(hidden_size=AH, time_hidden_size=TH),
                action_decoder=MLPProjector(input_dim=AH, output_dim=AD, width=AH, depth=2,
                                            final_layer_norm=False, use_mlp_layer_norm=False),
                shape_meta={"obs": {"state": {"shape": [SD], "horizon": 2}}, "action": {"shape": [AD], "horizon": 4}},
                action_hidden_size=AH,
                world_model_expert=DummyFlowExpert(hidden_size=AH, time_hidden_size=TH),
                frozen_teacher=nn.Linear(AD, 32),
                wm_motion_encoder=None,
                world_model_config=WorldModelConfig(
                    num_future_frames=2, target_image_size=(32, 32),
                    teacher_patch_size=16, upsample_factor=1,
                    motion_conditioning=True,
                ),
            )

    def test_world_model_parameters_include_wm_encoders(self):
        model = build_wm_enabled_model(action_conditioning=True, motion_conditioning=True)
        wm_param_ids = {id(p) for p in model.world_model_parameters}
        for p in model.wm_action_encoder.parameters():
            assert id(p) in wm_param_ids
        for p in model.wm_motion_encoder.parameters():
            assert id(p) in wm_param_ids


# ======================================================================
# World model: single-forward bidirectional pass over both views
# ======================================================================


class FakeFrozenTeacher(nn.Module):
    """Stand-in teacher: returns zeros with shape matching WM pred contract."""

    def __init__(self, K: int, spatial: int, D: int):
        super().__init__()
        self.K = K
        self.spatial = spatial
        self.D = D
        self.call_count = 0
        self.last_input_batch = None

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        self.last_input_batch = int(frames.shape[0])
        return torch.zeros(frames.shape[0], self.K, self.spatial, self.D,
                           dtype=torch.float32, device=frames.device)


def build_wm_forward_fixture(has_breast: bool):
    """Build a minimal WM-enabled LegendVLA whose forward_world_model_stream can run
    end-to-end, plus a minimal batch + backbone output that satisfy the contract."""
    from src.model.action.action_head import ActionEncoder, MLPProjector
    from src.model.vlm.prefix_cache import BackboneStreamOutput, PrefixKVCache

    B, K = 2, 2
    target_hw = 32
    teacher_patch, upsample = 16, 1
    grid = target_hw // (teacher_patch * upsample)  # 2
    spatial = (grid * upsample) ** 2                # 4
    D = AH

    wm_expert = DummyFlowExpert(hidden_size=D, time_hidden_size=TH)
    teacher = FakeFrozenTeacher(K=K, spatial=spatial, D=D)
    wm_action_encoder = ActionEncoder(action_dim=AD, width=D, time_cond=False)
    wm_motion_encoder = MLPProjector(
        input_dim=16, output_dim=D, width=D, depth=2,
        final_layer_norm=False, use_mlp_layer_norm=False,
    )

    model = LegendVLA(
        backbone=DummyBackbone(hidden_size=H, num_layers=LAYERS,
                               num_heads=NH, num_kv_heads=NKV, head_dim=HD),
        state_encoder=FourierActionEncoder(action_dim=SD, width=H, time_cond=False,
                                           enable_fourier_embed=False, mlp_depth=2,
                                           final_layer_norm=False, use_mlp_layer_norm=False),
        action_encoder=FourierActionEncoder(action_dim=AD, width=D, time_cond=False,
                                            enable_fourier_embed=False, mlp_depth=2,
                                            final_layer_norm=False, use_mlp_layer_norm=False),
        time_embedding=TimeEmbedding(TH),
        flow_expert=DummyFlowExpert(hidden_size=D, time_hidden_size=TH),
        action_decoder=MLPProjector(input_dim=D, output_dim=AD, width=D, depth=2,
                                    final_layer_norm=False, use_mlp_layer_norm=False),
        shape_meta={"obs": {"state": {"shape": [SD], "horizon": 2}},
                    "action": {"shape": [AD], "horizon": 4}},
        action_hidden_size=D,
        flow_config=FlowConfig(num_parallel_t=1, num_inference_steps=3),
        ar_action_train_config=ARActionTrainConfig(chunk_size=2),
        rtc_config=RTCConfig(),
        loss_config=LossConfig(),
        knowledge_insulation=True,
        world_model_expert=wm_expert,
        frozen_teacher=teacher,
        wm_action_encoder=wm_action_encoder,
        wm_motion_encoder=wm_motion_encoder,
        world_model_config=WorldModelConfig(
            num_future_frames=K, target_image_size=(target_hw, target_hw),
            teacher_patch_size=teacher_patch, upsample_factor=upsample,
            action_conditioning=True, motion_conditioning=True,
        ),
    )

    # Wrap expert.forward to count calls while keeping real behavior.
    orig_expert_forward = wm_expert.forward
    call_counter = {"expert": 0}
    def counting_expert_forward(*args, **kwargs):
        call_counter["expert"] += 1
        return orig_expert_forward(*args, **kwargs)
    wm_expert.forward = counting_expert_forward

    prefix_len = 3
    prefix_cache = PrefixKVCache(
        keys=torch.randn(LAYERS, B, NKV, prefix_len, HD),
        values=torch.randn(LAYERS, B, NKV, prefix_len, HD),
        mask=torch.ones(B, prefix_len, dtype=torch.bool),
        lengths=torch.full((B,), prefix_len, dtype=torch.long),
    )
    backbone_output = BackboneStreamOutput(
        last_hidden_states=torch.zeros(B, prefix_len, H),
        position_ids=torch.arange(prefix_len).unsqueeze(0).expand(B, -1),
        past_key_values_hf=None,
        prefix_cache=prefix_cache,
    )

    batch = {
        "input_ids": torch.zeros(B, prefix_len, dtype=torch.long),
        "answer_start_idx": torch.full((B,), prefix_len, dtype=torch.long),
        "actions": torch.zeros(B, 4, AD),
        "actions_valid_mask": torch.ones(B, 4, AD, dtype=torch.bool),
        "n_future_frames": torch.full((B,), K, dtype=torch.long),
        "future_frames": torch.zeros(B, K, target_hw, target_hw, 3, dtype=torch.uint8),
        "future_head_motion": torch.zeros(B, K, 16),
    }
    if has_breast:
        batch["breast_future_frames"] = torch.zeros(B, K, target_hw, target_hw, 3, dtype=torch.uint8)
        batch["future_breast_motion"] = torch.zeros(B, K, 16)

    return model, batch, backbone_output, teacher, call_counter, (B, K, spatial, D)


class TestWorldModelSingleForward:
    def test_single_view_calls_expert_and_teacher_once(self):
        model, batch, bbo, teacher, counter, (B, K, spatial, D) = build_wm_forward_fixture(has_breast=False)
        out = model.forward_world_model_stream(batch, bbo)
        assert counter["expert"] == 1
        assert teacher.call_count == 1
        assert teacher.last_input_batch == B
        assert out["pred"].shape == (B, 1, K, spatial, D)
        assert out["target"].shape == (B, 1, K, spatial, D)

    def test_dual_view_calls_expert_and_teacher_once(self):
        model, batch, bbo, teacher, counter, (B, K, spatial, D) = build_wm_forward_fixture(has_breast=True)
        out = model.forward_world_model_stream(batch, bbo)
        # Key assertion: still one expert + one teacher call, regardless of V.
        assert counter["expert"] == 1
        assert teacher.call_count == 1
        # Teacher saw head+breast stacked along batch dim → 2B.
        assert teacher.last_input_batch == 2 * B
        assert out["pred"].shape == (B, 2, K, spatial, D)
        assert out["target"].shape == (B, 2, K, spatial, D)
