from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.model.vlm.prefix_cache import BackboneStreamOutput


@dataclass
class BackboneEmbedOutput:
    inputs_embeds: torch.Tensor
    visual_pos_masks: torch.Tensor | None
    deepstack_visual_embeds: list[torch.Tensor] | None


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

        # ── Tokenizer & processor ──
        processor = AutoProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code,
        )
        tokenizer = processor.tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": [state_token, action_token]})

        # ── Quantization config ──
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

        # ── Load model ──
        if attn_implementation is not None and not isinstance(attn_implementation, str):
            raise TypeError("attn_implementation must be a string or None.")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=resolved_dtype,
            quantization_config=quantization_config,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation={"text_config": attn_implementation} if attn_implementation is not None else None,
        )
        self.model.resize_token_embeddings(len(tokenizer))

        # ── Extract references BEFORE LoRA wrapping ──
        # Qwen3VLForConditionalGeneration → .model (Qwen3VLModel) → .language_model (Qwen3VLTextModel)
        text_config = self.model.config.text_config
        self.base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        self.language_model = self.base_model.model.language_model
        self.lm_head = self.model.lm_head

        # ── LoRA (via PEFT) ──
        self.use_lora = use_lora
        if use_lora:
            self._apply_lora(lora or {}, use_quantization)

        # ── Public attributes ──
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

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    def _apply_lora(self, lora_cfg: dict, use_quantization: bool) -> None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model_for_peft = self.model
        if use_quantization:
            model_for_peft = prepare_model_for_kbit_training(model_for_peft, use_gradient_checkpointing=False)

        kwargs = dict(lora_cfg)
        # Ensure list type for PEFT
        for key in ("target_modules", "modules_to_save"):
            if isinstance(kwargs.get(key), tuple):
                kwargs[key] = list(kwargs[key])

        # get_peft_model automatically freezes non-LoRA parameters
        self.model = get_peft_model(model_for_peft, LoraConfig(task_type="CAUSAL_LM", **kwargs))

    def freeze_non_lora_parameters(self) -> None:
        if self.use_lora:
            # PEFT already handles requires_grad; re-freeze any that were unfrozen
            base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
            for param in base.parameters():
                param.requires_grad = False
            # Re-enable LoRA params
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_torch_dtype(dtype_name: str | torch.dtype | None) -> torch.dtype | None:
        if dtype_name is None:
            return None
        if isinstance(dtype_name, torch.dtype):
            return dtype_name
        return getattr(torch, str(dtype_name))

    # ------------------------------------------------------------------
    # Embedding construction
    # ------------------------------------------------------------------
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

        # Upstream-aligned block copied/adapted from transformers Qwen3-VL.
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
        """Construct final language-model embeddings from project batch tensors.

        The embedding pipeline is intentionally non-standard and happens in three stages:
        - Start from token embeddings of `input_ids`.
        - Replace Qwen image/video placeholder token positions with visual features from the HF vision tower.
        - Replace project `<state>` and `<action>` placeholder token positions with learned slot embeddings.

        `mm_token_type_ids` is kept for position-id computation later in the pipeline, but it does not directly
        control embedding replacement in this function.
        """
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
        use_cache: bool = True,
        output_hidden_states: bool = True,
        past_key_values: Any = None,
    ) -> BackboneStreamOutput:
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
        # Upstream-aligned block copied/adapted from transformers Qwen3-VL.
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=embed_output.inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            visual_pos_masks=embed_output.visual_pos_masks,
            deepstack_visual_embeds=embed_output.deepstack_visual_embeds,
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
