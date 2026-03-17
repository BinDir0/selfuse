from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.model.vlm.prefix_cache import BackboneStreamOutput


def get_cfg_value(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def align_features_by_slot(features: torch.Tensor, slot: torch.LongTensor) -> torch.Tensor:
    batch_size, seq_len = slot.shape
    hidden_size = features.shape[-1]
    if features.size(1) == 0:
        return features.new_zeros((batch_size, seq_len, hidden_size))
    safe_slot = slot.clamp(min=0, max=features.size(1) - 1)
    gather_index = safe_slot.unsqueeze(-1).expand(-1, -1, hidden_size)
    return torch.gather(features, dim=1, index=gather_index)


@dataclass
class BackboneEmbedOutput:
    inputs_embeds: torch.Tensor
    visual_pos_masks: torch.Tensor | None
    deepstack_visual_embeds: list[torch.Tensor] | None


class Qwen3VLBackboneWrapper(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.model_name_or_path = get_cfg_value(cfg, "model_name_or_path")
        self.trust_remote_code = bool(get_cfg_value(cfg, "trust_remote_code", False))
        self.freeze_backbone = bool(get_cfg_value(cfg, "freeze_backbone", False))
        self.state_token = get_cfg_value(cfg, "state_token", "<state>")
        self.action_token = get_cfg_value(cfg, "action_token", "<action>")
        self.torch_dtype_name = get_cfg_value(cfg, "torch_dtype", None)

        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL backbone requires a recent transformers installation. "
                "Please install transformers with Qwen3-VL support before instantiating this model."
            ) from exc

        processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        tokenizer = processor.tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": [self.state_token, self.action_token]})

        torch_dtype = None
        if self.torch_dtype_name is not None:
            torch_dtype = getattr(torch, str(self.torch_dtype_name))

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.model.resize_token_embeddings(len(tokenizer))

        self.tokenizer = tokenizer
        self.processor = processor
        self.language_model = self.model.model.language_model
        self.hidden_size = self.model.config.text_config.hidden_size
        self.vocab_size = len(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.image_token = getattr(processor, "image_token", "<|image_pad|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.state_token_id = tokenizer.convert_tokens_to_ids(self.state_token)
        self.action_token_id = tokenizer.convert_tokens_to_ids(self.action_token)
        self.num_heads = self.model.config.text_config.num_attention_heads
        self.num_kv_heads = self.model.config.text_config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.lm_head = self.model.lm_head

        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def resize_token_embeddings(self, vocab_size: int) -> None:
        self.model.resize_token_embeddings(vocab_size)
        self.vocab_size = vocab_size

    def build_base_text_embeds(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def encode_image_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        if pixel_values is None:
            return inputs_embeds, None, None

        image_outputs = self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.model.model.get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = image_outputs.deepstack_features
        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

    def replace_slot_embeddings(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        state_slot_embeds: torch.Tensor | None,
        action_slot_embeds: torch.Tensor | None,
        state_token_id: int | None = None,
        action_token_id: int | None = None,
    ) -> torch.Tensor:
        state_token_id = self.state_token_id if state_token_id is None else state_token_id
        action_token_id = self.action_token_id if action_token_id is None else action_token_id

        if state_slot_embeds is not None:
            state_mask = input_ids == state_token_id
            state_slot = state_mask.long().cumsum(dim=1) - 1
            aligned_state_embeds = align_features_by_slot(state_slot_embeds.to(inputs_embeds.dtype), state_slot)
            inputs_embeds = torch.where(state_mask.unsqueeze(-1), aligned_state_embeds, inputs_embeds)

        if action_slot_embeds is not None:
            action_mask = input_ids == action_token_id
            action_slot = action_mask.long().cumsum(dim=1) - 1
            aligned_action_embeds = align_features_by_slot(action_slot_embeds.to(inputs_embeds.dtype), action_slot)
            inputs_embeds = torch.where(action_mask.unsqueeze(-1), aligned_action_embeds, inputs_embeds)

        return inputs_embeds

    def build_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
        state_slot_embeds: torch.Tensor | None,
        action_slot_embeds: torch.Tensor | None,
        state_token_id: int | None = None,
        action_token_id: int | None = None,
    ) -> BackboneEmbedOutput:
        del mm_token_type_ids
        inputs_embeds = self.build_base_text_embeds(input_ids)
        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self.encode_image_features(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        inputs_embeds = self.replace_slot_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            state_slot_embeds=state_slot_embeds,
            action_slot_embeds=action_slot_embeds,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )
        return BackboneEmbedOutput(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def compute_position_ids(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
        past_key_values: Any = None,
    ) -> torch.Tensor | None:
        return self.model.model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

    def forward_language_model(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        use_cache: bool,
        output_hidden_states: bool,
        past_key_values: Any = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> BackboneStreamOutput:
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        return BackboneStreamOutput(
            last_hidden_states=outputs.last_hidden_state,
            all_hidden_states=outputs.hidden_states,
            position_ids=position_ids,
            past_key_values_hf=outputs.past_key_values,
        )

