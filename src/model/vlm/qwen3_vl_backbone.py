from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    create_causal_mask,
    eager_attention_forward,
)

from src.model.vlm.prefix_cache import BackboneStreamOutput


@dataclass
class BackboneEmbedOutput:
    inputs_embeds: torch.Tensor
    visual_pos_masks: torch.Tensor | None
    deepstack_visual_embeds: list[torch.Tensor] | None


class Qwen3VLTextAttentionWithKV(Qwen3VLTextAttention):
    """Wrap the upstream attention block and also expose the full-sequence key/value tensors."""

    def __init__(self, base_attention: Qwen3VLTextAttention):
        nn.Module.__init__(self)
        # Keep hidden ref for non-module attributes (config, head_dim, scaling, etc.)
        object.__setattr__(self, "base_attention", base_attention)
        # Register parameter-bearing submodules so FSDP2 can manage them.
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        base_attention = self.base_attention
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, base_attention.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = base_attention.rotary_fn(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                base_attention.layer_idx,
                cache_kwargs,
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            base_attention.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            base_attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not base_attention.training else base_attention.attention_dropout,
            scaling=base_attention.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, key_states, value_states


class Qwen3VLTextDecoderLayerWithKV(Qwen3VLTextDecoderLayer):
    """Wrap the upstream decoder layer and return the layer key/value tensors explicitly."""

    def __init__(self, base_layer: Qwen3VLTextDecoderLayer):
        nn.Module.__init__(self)
        self.self_attn = Qwen3VLTextAttentionWithKV(base_layer.self_attn)
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        self.mlp = base_layer.mlp
        self.hidden_size = base_layer.hidden_size
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, key_states, value_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, key_states, value_states


class Qwen3VLTextModelWithKV(Qwen3VLTextModel):
    """Run the upstream decoder stack while returning full-sequence layer KV for prefix reuse.

    This wrapper keeps the original forward inputs intact and only changes the outputs:
    `past_key_values` is reused to carry one `(key, value)` tuple per decoder layer.
    """

    def __init__(self, base_model: Qwen3VLTextModel):
        nn.Module.__init__(self)
        object.__setattr__(self, "upstream_text_model", base_model)
        self.config = base_model.config
        self.padding_idx = base_model.padding_idx
        self.vocab_size = base_model.vocab_size
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayerWithKV(layer) for layer in base_model.layers]
        )
        # Clear original layers so FSDP2 doesn't find the same parameters
        # through both the original and wrapper module paths.
        base_model.layers = nn.ModuleList()
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        return_dict = bool(kwargs.pop("return_dict", True))
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("output_attentions", None)

        base_model = self.upstream_text_model
        if inputs_embeds is None:
            inputs_embeds = base_model.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_seen_tokens = int(past_key_values.get_seq_length())
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = base_model.rotary_emb(hidden_states, rope_position_ids)
        layer_kv: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states, key_states, value_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            layer_kv.append((key_states, value_states))

            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                hidden_states = base_model._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = base_model.norm(hidden_states)
        full_layer_kv = tuple(layer_kv)
        if not return_dict:
            return hidden_states, full_layer_kv
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=full_layer_kv,
        )


class Qwen3VLBackboneWrapper(nn.Module):
    """Wrap the HF Qwen3-VL backbone and expose the project-specific embedding pipeline."""

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        freeze_backbone: bool = False,
        torch_dtype: str | None = None,
        state_token: str = "<state>",
        action_token: str = "<action>",
        use_lora: bool = False,
        lora: dict[str, Any] | None = None,
        use_quantization: bool = False,
        quantization: dict[str, Any] | None = None,
        device_map: Any = None,
        low_cpu_mem_usage: bool = True,
        attn_implementation: str | None = None,
    ):
        super().__init__()
        try:
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL backbone requires a recent transformers installation."
            ) from exc

        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = processor.tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": [state_token, action_token]})

        resolved_dtype = self._resolve_torch_dtype(torch_dtype)
        quantization_config = None
        if use_quantization:
            qkwargs = dict(quantization or {})
            compute_dtype_name = qkwargs.get("bnb_4bit_compute_dtype")
            qkwargs["bnb_4bit_compute_dtype"] = (
                self._resolve_torch_dtype(compute_dtype_name)
                if compute_dtype_name is not None
                else (resolved_dtype or torch.bfloat16)
            )
            quantization_config = BitsAndBytesConfig(**qkwargs)

        if attn_implementation is not None and not isinstance(attn_implementation, str):
            raise TypeError("attn_implementation must be a string or None.")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=resolved_dtype,
            quantization_config=quantization_config,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation,
        )
        self.model.resize_token_embeddings(len(tokenizer))

        self.use_lora = use_lora
        if use_lora:
            self._apply_lora(lora or {}, use_quantization)

        text_config = self.model.config.text_config
        self.base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        self.hf_language_model = self.base_model.model.language_model
        self.language_model = Qwen3VLTextModelWithKV(self.hf_language_model)
        self.lm_head = self.model.lm_head

        self.tokenizer = tokenizer
        self.processor = processor
        self.hidden_size = text_config.hidden_size
        self.vocab_size = len(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.image_token = getattr(processor, "image_token", "<|image_pad|>")
        self.video_token = getattr(processor, "video_token", "<|video_pad|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.state_token_id = tokenizer.convert_tokens_to_ids(state_token)
        self.action_token_id = tokenizer.convert_tokens_to_ids(action_token)
        self.num_heads = text_config.num_attention_heads
        self.num_kv_heads = text_config.num_key_value_heads
        self.head_dim = int(getattr(text_config, "head_dim", self.hidden_size // self.num_heads))

        if freeze_backbone:
            self.freeze_non_lora_parameters()

    def _apply_lora(self, lora_cfg: dict, use_quantization: bool) -> None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model_for_peft = self.model
        if use_quantization:
            model_for_peft = prepare_model_for_kbit_training(model_for_peft, use_gradient_checkpointing=False)

        kwargs = dict(lora_cfg)
        for key in ("target_modules", "modules_to_save"):
            if isinstance(kwargs.get(key), tuple):
                kwargs[key] = list(kwargs[key])

        self.model = get_peft_model(model_for_peft, LoraConfig(task_type="CAUSAL_LM", **kwargs))

    def freeze_non_lora_parameters(self) -> None:
        if self.use_lora:
            base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
            for param in base.parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str | torch.dtype | None) -> torch.dtype | None:
        if dtype_name is None:
            return None
        if isinstance(dtype_name, torch.dtype):
            return dtype_name
        return getattr(torch, str(dtype_name))

    def enable_gradient_checkpointing(self) -> None:
        """Enable checkpointing on the vision tower and the training text wrapper."""
        checkpoint_func = partial(checkpoint, use_reentrant=False)
        self.language_model.gradient_checkpointing = True
        self.language_model._gradient_checkpointing_func = checkpoint_func
        for layer in self.language_model.layers:
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = checkpoint_func

        visual_model = self.base_model.model.visual
        enable_method = getattr(visual_model, "gradient_checkpointing_enable", None)
        if callable(enable_method):
            enable_method()

    def disable_gradient_checkpointing(self) -> None:
        """Disable checkpointing on the vision tower and the training text wrapper."""
        self.language_model.gradient_checkpointing = False
        for layer in self.language_model.layers:
            layer.gradient_checkpointing = False

        visual_model = self.base_model.model.visual
        disable_method = getattr(visual_model, "gradient_checkpointing_disable", None)
        if callable(disable_method):
            disable_method()

    def encode_visual_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        base_model = self.base_model
        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        # This block follows the upstream Qwen3-VL embedding replacement flow.
        if pixel_values is not None:
            image_outputs = base_model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = base_model.model.get_placeholder_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            deepstack_image_embeds = image_outputs.deepstack_features

        if pixel_values_videos is not None:
            video_outputs = base_model.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                return_dict=True,
            )
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = base_model.model.get_placeholder_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            deepstack_video_embeds = video_outputs.deepstack_features

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for image_embed, video_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                visual_embed = image_embed.new_zeros(visual_pos_masks.sum(), image_embed.shape[-1]).to(image_embed.device)
                visual_embed[image_mask_joint, :] = image_embed
                visual_embed[video_mask_joint, :] = video_embed
                deepstack_visual_embeds.append(visual_embed)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

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
            state_mask = (input_ids == state_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(state_mask, state_slot_embeds.to(inputs_embeds.dtype))

        if action_slot_embeds is not None:
            action_mask = (input_ids == action_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(action_mask, action_slot_embeds.to(inputs_embeds.dtype))

        return inputs_embeds

    def build_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
        state_slot_embeds: torch.Tensor | None,
        action_slot_embeds: torch.Tensor | None,
        state_token_id: int | None = None,
        action_token_id: int | None = None,
    ) -> BackboneEmbedOutput:
        """Construct final language-model embeddings from project batch tensors."""
        del mm_token_type_ids
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self.encode_visual_features(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
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

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
        state_slot_embeds: torch.Tensor | None,
        action_slot_embeds: torch.Tensor | None,
        state_token_id: int | None = None,
        action_token_id: int | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = True,
        past_key_values: Any = None,
    ) -> BackboneStreamOutput:
        del output_hidden_states
        embed_output = self.build_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            state_slot_embeds=state_slot_embeds,
            action_slot_embeds=action_slot_embeds,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )
        position_ids = self.base_model.model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=embed_output.inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=embed_output.inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            visual_pos_masks=embed_output.visual_pos_masks,
            deepstack_visual_embeds=embed_output.deepstack_visual_embeds,
            use_cache=use_cache,
            return_dict=True,
        )

        # In training, `past_key_values` carries one full-sequence `(key, value)` pair per text layer.
        return BackboneStreamOutput(
            last_hidden_states=outputs.last_hidden_state,
            position_ids=position_ids,
            past_key_values_hf=outputs.past_key_values,
        )
