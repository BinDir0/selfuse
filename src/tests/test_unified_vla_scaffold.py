import torch
from torch import nn

from src.policy.legendvla import LegendVLA
from src.model.vlm.prefix_cache import BackboneStreamOutput, gather_action_position_ids


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size=32, vocab_size=128, num_layers=8, num_heads=4, num_kv_heads=4, head_dim=8):
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
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.hidden_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def build_inputs_embeds(
        self,
        input_ids,
        pixel_values,
        image_grid_thw,
        mm_token_type_ids,
        state_slot_embeds,
        action_slot_embeds,
    ):
        del pixel_values, image_grid_thw, mm_token_type_ids
        embeds = self.embed(input_ids)
        if state_slot_embeds is not None:
            state_mask = input_ids == self.state_token_id
            state_slot = state_mask.long().cumsum(dim=1) - 1
            gather_index = state_slot.clamp(min=0, max=state_slot_embeds.shape[1] - 1).unsqueeze(-1).expand(-1, -1, embeds.shape[-1])
            state_values = torch.gather(state_slot_embeds, dim=1, index=gather_index)
            embeds = torch.where(state_mask.unsqueeze(-1), state_values, embeds)
        if action_slot_embeds is not None:
            action_mask = input_ids == self.action_token_id
            action_slot = action_mask.long().cumsum(dim=1) - 1
            gather_index = action_slot.clamp(min=0, max=action_slot_embeds.shape[1] - 1).unsqueeze(-1).expand(-1, -1, embeds.shape[-1])
            action_values = torch.gather(action_slot_embeds, dim=1, index=gather_index)
            embeds = torch.where(action_mask.unsqueeze(-1), action_values, embeds)
        return type("BackboneEmbedOutput", (), {
            "inputs_embeds": embeds,
            "visual_pos_masks": None,
            "deepstack_visual_embeds": None,
        })

    def compute_position_ids(self, input_ids, inputs_embeds, attention_mask, image_grid_thw, mm_token_type_ids, past_key_values=None):
        del input_ids, inputs_embeds, image_grid_thw, mm_token_type_ids, past_key_values
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return position_ids.unsqueeze(0).expand(3, -1, -1)

    def forward_language_model(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        use_cache,
        output_hidden_states,
        past_key_values=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
    ):
        del attention_mask, use_cache, output_hidden_states, past_key_values, visual_pos_masks, deepstack_visual_embeds
        hidden = self.hidden_proj(inputs_embeds)
        batch_size, seq_len, _ = hidden.shape
        cache = []
        for _ in range(self.num_layers):
            key = hidden[..., : self.num_kv_heads * self.head_dim].view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value = key.clone()
            cache.append((key, value))
        return BackboneStreamOutput(
            last_hidden_states=hidden,
            all_hidden_states=(hidden,),
            position_ids=position_ids,
            past_key_values_hf=cache,
        )


class DummyDiffLoss(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, target, latent_condition_embeds, mask=None):
        del latent_condition_embeds
        loss = target.pow(2)
        if mask is not None:
            loss = loss * mask.unsqueeze(-1).to(dtype=loss.dtype)
        return loss.mean()

    def sample(self, latent_condition, temperature: float = 1.0, cfg: float = 1.0):
        del temperature, cfg
        return latent_condition.new_zeros(latent_condition.shape[0], self.action_dim)


def make_vla_batch(batch_size=2, action_horizon=4, action_dim=48):
    full_input_ids = torch.tensor([
        [100, 10, 101, 101, 102, 102, 102, 102, 0, 0],
        [100, 11, 12, 13, 14, 0, 0, 0, 0, 0],
    ], dtype=torch.long)
    input_ids = full_input_ids[:batch_size]
    attention_mask = (input_ids != 0).long()
    labels = input_ids.clone()
    labels[:, :4] = -100
    states = torch.randn(batch_size, 18, action_dim)
    actions = torch.randn(batch_size, action_horizon, action_dim)
    actions_valid_mask = torch.zeros(batch_size, action_horizon, action_dim, dtype=torch.bool)
    actions_valid_mask[0] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": torch.randn(batch_size, 3, 16, 16),
        "image_grid_thw": torch.ones(batch_size, 3, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(batch_size, input_ids.shape[1], dtype=torch.long),
        "states": states,
        "actions": actions,
        "actions_valid_mask": actions_valid_mask,
        "answer_start_idx": torch.tensor([4, 5], dtype=torch.long)[:batch_size],
        "is_vla_data": torch.tensor([True, False])[:batch_size],
        "n_states": torch.tensor([2, 0], dtype=torch.long)[:batch_size],
        "n_actions": torch.tensor([4, 0], dtype=torch.long)[:batch_size],
        "t": torch.rand(batch_size),
    }


def make_vlm_batch():
    input_ids = torch.tensor([[100, 10, 11, 12]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
        "pixel_values": torch.randn(1, 3, 16, 16),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        "mm_token_type_ids": torch.zeros_like(input_ids),
        "answer_start_idx": torch.tensor([4], dtype=torch.long),
        "is_vla_data": torch.tensor([False]),
        "n_states": torch.tensor([0], dtype=torch.long),
        "n_actions": torch.tensor([0], dtype=torch.long),
    }


def make_model(diffloss=None):
    cfg = {
        "ignore_index": -100,
        "flow_sig_min": 0.001,
        "num_inference_steps": 5,
        "time_hidden_size": 32,
        "time_min_period": 0.004,
        "time_max_period": 4.0,
        "ar_action_noise_std": 0.02,
        "ar_action_chunk_size": 2,
        "diffloss_micro_batch_size": 1,
        "use_rtc": True,
        "rtc_delay_strategy": "uniform",
        "rtc_max_delay": 2,
        "expert": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_layers": 4,
            "rope_theta": 10000.0,
        },
        "diffloss": {
            "enabled": diffloss is not None,
            "z_channels": 32,
            "target_channels": 48,
            "num_inference_steps": 5,
        },
        "loss_weights": {
            "ce_loss_weight": 0.1,
            "diffusion_loss_weight": 1.0,
            "flow_loss_weight": 1.0,
        },
    }
    shape_meta = {
        "obs": {"state": {"shape": [48], "horizon": 18}},
        "action": {"shape": [48], "horizon": 4},
    }
    return LegendVLA(cfg=cfg, shape_meta=shape_meta, backbone=DummyBackbone(), diffloss=diffloss)


def test_gather_action_position_ids_uses_backbone_positions():
    input_ids = torch.tensor([[1, 102, 102, 0]], dtype=torch.long)
    position_ids = torch.tensor([[[0, 5, 6, 0]], [[0, 5, 6, 0]], [[0, 5, 6, 0]]], dtype=torch.long)
    gathered = gather_action_position_ids(input_ids, 102, position_ids, torch.tensor([2]))
    assert gathered.tolist() == [[5, 6]]


def test_legendvla_scaffold_forward():
    model = make_model(diffloss=None)
    batch = make_vla_batch()
    output = model("train", batch)
    assert set(output.keys()) == {"total_loss", "ce_loss", "diffusion_loss", "flow_loss"}
    assert output["total_loss"].ndim == 0


def test_legendvla_flow_inference_smoke():
    model = make_model(diffloss=None)
    batch = make_vla_batch(batch_size=1)
    actions = model("infer_action", batch)
    assert actions.shape == (1, 4, 48)


def test_legendvla_vla_inference_smoke():
    model = make_model(diffloss=DummyDiffLoss(action_dim=48))
    batch = make_vla_batch(batch_size=1)
    result = model("infer_vla", batch, max_new_tokens=4)
    assert result["generated_actions"].shape == (1, 4, 48)
    assert result["generated_hidden_states"].shape[:2] == (1, 4)


def test_legendvla_vlm_inference_smoke():
    model = make_model(diffloss=None)
    batch = make_vlm_batch()
    result = model("infer_vlm", batch, max_new_tokens=3, temperature=0.0)
    assert result["generated_ids"].shape == (1, 3)
    assert result["full_ids"].shape == (1, 7)
