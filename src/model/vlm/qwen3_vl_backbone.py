from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.model.vlm.prefix_cache import BackboneStreamOutput


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
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.freeze_backbone = freeze_backbone
        self.state_token = state_token
        self.action_token = action_token
        self.torch_dtype_name = torch_dtype
        self.use_lora = use_lora
        self.use_quantization = use_quantization
        self.lora_cfg = dict(lora or {})
        self.quantization_cfg = dict(quantization or {})
        self.device_map = device_map
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self._lora_param_ids: set[int] = set()

        try:
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
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

        torch_dtype = self.resolve_torch_dtype(self.torch_dtype_name)
        quantization_config = None
        if self.use_quantization:
            quantization_kwargs = dict(self.quantization_cfg)
            compute_dtype_name = quantization_kwargs.get("bnb_4bit_compute_dtype")
            if compute_dtype_name is None:
                quantization_kwargs["bnb_4bit_compute_dtype"] = torch_dtype or torch.bfloat16
            else:
                quantization_kwargs["bnb_4bit_compute_dtype"] = self.resolve_torch_dtype(compute_dtype_name)
            quantization_config = BitsAndBytesConfig(**quantization_kwargs)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map=self.device_map,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        )
        self.model.resize_token_embeddings(len(tokenizer))

        if self.use_lora:
            self.apply_transformers_lora()

        self.base_model = self.resolve_base_model(self.model)
        self.tokenizer = tokenizer
        self.processor = processor
        self.language_model = self.resolve_language_model(self.base_model)
        self.hidden_size = self.base_model.config.text_config.hidden_size
        self.vocab_size = len(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.image_token = getattr(processor, "image_token", "<|image_pad|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.state_token_id = tokenizer.convert_tokens_to_ids(self.state_token)
        self.action_token_id = tokenizer.convert_tokens_to_ids(self.action_token)
        self.num_heads = self.base_model.config.text_config.num_attention_heads
        self.num_kv_heads = self.base_model.config.text_config.num_key_value_heads
        self.head_dim = int(getattr(self.base_model.config.text_config, "head_dim", self.hidden_size // self.num_heads))
        self.lm_head = self.base_model.lm_head

        if self.freeze_backbone:
            self.freeze_non_lora_parameters()

    @staticmethod
    def resolve_torch_dtype(dtype_name: str | torch.dtype | None) -> torch.dtype | None:
        if dtype_name is None:
            return None
        if isinstance(dtype_name, torch.dtype):
            return dtype_name
        return getattr(torch, str(dtype_name))

    def resolve_base_model(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "get_base_model"):
            return model.get_base_model()
        return model

    def resolve_language_model(self, model: nn.Module) -> nn.Module:
        if hasattr(model, "language_model"):
            return model.language_model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        raise AttributeError("Unable to locate the Qwen3-VL language model module.")

    def apply_transformers_lora(self) -> None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError(
                "LoRA training now relies on the Hugging Face PEFT integration. "
                "Please install `peft` to enable `use_lora=True`."
            ) from exc

        model_for_peft = self.model
        if self.use_quantization:
            model_for_peft = prepare_model_for_kbit_training(model_for_peft, use_gradient_checkpointing=False)

        lora_kwargs = dict(self.lora_cfg)
        task_type_value = str(lora_kwargs.pop("task_type", "CAUSAL_LM"))
        target_modules = lora_kwargs.pop(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        modules_to_save = lora_kwargs.pop("modules_to_save", None)
        if isinstance(target_modules, tuple):
            target_modules = list(target_modules)
        if isinstance(modules_to_save, tuple):
            modules_to_save = list(modules_to_save)

        task_type = getattr(TaskType, task_type_value)
        peft_config = LoraConfig(
            task_type=task_type,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            **lora_kwargs,
        )
        self.model = get_peft_model(model_for_peft, peft_config)
        self._lora_param_ids = {
            id(param)
            for _, param in self.model.named_parameters()
            if param.requires_grad
        }

    def freeze_non_lora_parameters(self) -> None:
        if not self.use_lora:
            for param in self.model.parameters():
                param.requires_grad = False
            return

        for param in self.model.parameters():
            param.requires_grad = id(param) in self._lora_param_ids

    def resize_token_embeddings(self, vocab_size: int) -> None:
        self.base_model.resize_token_embeddings(vocab_size)
        self.vocab_size = vocab_size

    def build_base_text_embeds(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.base_model.get_input_embeddings()(input_ids)

    def encode_image_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        if pixel_values is None:
            return inputs_embeds, None, None

        image_outputs = self.base_model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.base_model.model.get_placeholder_mask(
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
        return self.base_model.model.compute_3d_position_ids(
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
