import torch
from torch import nn

from src.policy.legendvla import LegendVLA, FlowConfig, RTCConfig, LossConfig, ARActionTrainConfig
from src.model.action.action_head import FourierActionEncoder, MLPProjector
from src.model.common.modules import TimeEmbedding
from src.model.vlm.prefix_cache import BackboneStreamOutput
from src.tests.dummy_flow_expert import DummyFlowExpert


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
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.visual = nn.Module()
        self.base_model.model.visual.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, hidden_size)]
        )
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
        state_token_id=None,
        action_token_id=None,
        use_cache=True,
        output_hidden_states=True,
        past_key_values=None,
    ):
        del pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, mm_token_type_ids
        del state_token_id, action_token_id, use_cache, output_hidden_states, past_key_values

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

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        hidden = embeds
        for block in self.base_model.model.visual.blocks:
            hidden = block(hidden)
        hidden = self.hidden_proj(hidden)
        for layer in self.language_model.layers:
            hidden = layer(hidden)
        batch_size, seq_len, _ = hidden.shape
        cache = []
        for _ in range(self.num_layers):
            key = hidden[..., : self.num_kv_heads * self.head_dim].view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value = key.clone()
            cache.append((key, value))
        return BackboneStreamOutput(
            last_hidden_states=hidden,
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
        "pixel_values_videos": None,
        "video_grid_thw": None,
        "mm_token_type_ids": torch.zeros(batch_size, input_ids.shape[1], dtype=torch.long),
        "states": states,
        "actions": actions,
        "actions_valid_mask": actions_valid_mask,
        "answer_start_idx": torch.tensor([4, 5], dtype=torch.long)[:batch_size],
        "is_vla_data": torch.tensor([True, False])[:batch_size],
        "n_states": torch.tensor([2, 0], dtype=torch.long)[:batch_size],
        "n_actions": torch.tensor([4, 0], dtype=torch.long)[:batch_size],
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
        "pixel_values_videos": None,
        "video_grid_thw": None,
        "mm_token_type_ids": torch.zeros_like(input_ids),
        "answer_start_idx": torch.tensor([4], dtype=torch.long),
        "is_vla_data": torch.tensor([False]),
        "n_states": torch.tensor([0], dtype=torch.long),
        "n_actions": torch.tensor([0], dtype=torch.long),
    }


def make_model(diffloss=None):
    hidden_size = 32
    action_dim = 48
    state_dim = 48
    action_hidden_size = 32
    time_hidden_size = 32
    num_heads = 4
    num_kv_heads = 4
    head_dim = 8

    backbone = DummyBackbone(
        hidden_size=hidden_size, vocab_size=128,
        num_layers=8, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    state_encoder = FourierActionEncoder(
        action_dim=state_dim, width=hidden_size,
        time_cond=False, enable_fourier_embed=False, mlp_depth=2,
        final_layer_norm=False, use_mlp_layer_norm=False,
    )
    ar_action_encoder = FourierActionEncoder(
        action_dim=action_dim, width=hidden_size,
        time_cond=False, enable_fourier_embed=False, mlp_depth=2,
        final_layer_norm=False, use_mlp_layer_norm=False,
    )
    action_encoder = FourierActionEncoder(
        action_dim=action_dim, width=action_hidden_size,
        time_cond=False, enable_fourier_embed=False, mlp_depth=2,
        final_layer_norm=False, use_mlp_layer_norm=False,
    )
    time_embedding = TimeEmbedding(time_hidden_size)
    flow_expert = DummyFlowExpert(hidden_size=action_hidden_size, time_hidden_size=time_hidden_size)
    action_decoder = MLPProjector(
        input_dim=action_hidden_size, output_dim=action_dim,
        width=action_hidden_size, depth=2,
        final_layer_norm=False, use_mlp_layer_norm=False,
    )
    latent_condition_projector = MLPProjector(
        input_dim=hidden_size, output_dim=32,
        width=hidden_size, depth=2,
        final_layer_norm=False, use_mlp_layer_norm=False,
    )

    shape_meta = {
        "obs": {"state": {"shape": [48], "horizon": 18}},
        "action": {"shape": [48], "horizon": 4},
    }
    return LegendVLA(
        backbone=backbone,
        state_encoder=state_encoder,
        ar_action_encoder=ar_action_encoder,
        action_encoder=action_encoder,
        time_embedding=time_embedding,
        flow_expert=flow_expert,
        action_decoder=action_decoder,
        latent_condition_projector=latent_condition_projector,
        shape_meta=shape_meta,
        diffloss=diffloss,
        action_hidden_size=action_hidden_size,
        flow_config=FlowConfig(num_inference_steps=5),
        ar_action_train_config=ARActionTrainConfig(chunk_size=2),
        rtc_config=RTCConfig(delay_strategy="uniform", max_delay=2),
        loss_config=LossConfig(),
    )


def test_legendvla_scaffold_forward():
    model = make_model(diffloss=None)
    batch = make_vla_batch()
    output = model("train", batch)
    assert set(output.keys()) == {"total_loss", "ce_loss", "diffusion_loss", "reg_loss", "flow_loss"}
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


def test_legendvla_compile_blocks_smoke():
    model = make_model(diffloss=None)
    model.compile_blocks({"backend": "eager"})

    batch = make_vla_batch(batch_size=1)
    output = model("train", batch)

    assert set(output.keys()) == {"total_loss", "ce_loss", "diffusion_loss", "reg_loss", "flow_loss"}
    assert output["total_loss"].ndim == 0


def test_legendvla_compile_blocks_respects_module_flags():
    model = make_model(diffloss=None)
    model.compile_blocks({"backend": "eager", "vision": False, "text": True, "flow": True})

    assert not hasattr(model.backbone.base_model.model.visual.blocks[0], "_orig_mod")
    assert hasattr(model.backbone.language_model.layers[0], "_orig_mod")
    assert hasattr(model.flow_expert.layers[0], "_orig_mod")


def test_legendvla_compile_blocks_defaults_to_all_enabled():
    model = make_model(diffloss=None)

    model.compile_blocks({"backend": "eager"})

    assert hasattr(model.backbone.base_model.model.visual.blocks[0], "_orig_mod")
    assert hasattr(model.backbone.language_model.layers[0], "_orig_mod")
    assert hasattr(model.flow_expert.layers[0], "_orig_mod")
