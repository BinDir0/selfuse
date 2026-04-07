from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.utils.compile_utils import compile_module_list
from src.model.vlm.prefix_cache import (
    BackboneStreamOutput,
    slice_prefix_cache_from_full_kv,
)


@dataclass(frozen=True)
class FlowConfig:
    sig_min: float = 0.001
    num_parallel_t: int = 1
    sampling: str = "beta"
    alpha: float = 1.5
    beta: float = 1.0
    num_inference_steps: int = 10


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = True
    delay_strategy: str = "exp"
    max_delay: int = 16


@dataclass(frozen=True)
class LossConfig:
    ce_loss_weight: float = 0.1
    diffusion_loss_weight: float = 1.0
    flow_loss_weight: float = 1.0
    reg_loss_weight: float = 0.0


@dataclass(frozen=True)
class ARActionTrainConfig:
    noise_std: float = 0.02
    chunk_size: int = 4
    diffloss_repeat: int = 1


class LegendVLA(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        state_encoder: nn.Module,
        ar_action_encoder: nn.Module,
        action_encoder: nn.Module,
        time_embedding: nn.Module,
        flow_expert: nn.Module,
        action_decoder: nn.Module,
        latent_condition_projector: nn.Module,
        shape_meta: dict,
        diffloss: nn.Module | None = None,
        reg_action_head: nn.Module | None = None,
        ignore_index: int = -100,
        action_hidden_size: int = 1024,
        flow_config: FlowConfig = FlowConfig(),
        rtc_config: RTCConfig = RTCConfig(),
        loss_config: LossConfig = LossConfig(),
        ar_action_train_config: ARActionTrainConfig = ARActionTrainConfig(),
        knowledge_insulation: bool | int = True,
        # Camera intrinsic as token embedding (optional)
        camera_intrinsic_mode: str = "text",
        camera_encoder: nn.Module | None = None,        
    ):
        super().__init__()
        self.shape_meta = shape_meta

        # Camera intrinsic encoding
        self.camera_intrinsic_mode = camera_intrinsic_mode
        self.camera_encoder = camera_encoder

        # Backbone and derived attributes
        self.backbone = backbone
        self.vlm_hidden_size = int(getattr(self.backbone, "hidden_size"))
        self.vocab_size = int(getattr(self.backbone, "vocab_size"))
        self.pad_token_id = int(getattr(self.backbone, "pad_token_id"))
        self.image_token_index = int(getattr(self.backbone, "image_token_id"))
        self.state_token_index = int(getattr(self.backbone, "state_token_id"))
        self.action_token_index = int(getattr(self.backbone, "action_token_id"))
        self.lm_head = getattr(self.backbone, "lm_head")
        self.eos_token_id = getattr(getattr(self.backbone, "tokenizer", None), "eos_token_id", None)

        # Grouped config (frozen dataclasses)
        self.ignore_index = ignore_index
        self.CELoss = nn.CrossEntropyLoss(reduction="sum", ignore_index=ignore_index)
        self.flow_config = flow_config
        self.rtc_config = rtc_config
        self.loss_config = loss_config
        self.ar_action_train_config = ar_action_train_config

        if self.flow_config.num_parallel_t < 1:
            raise ValueError(f"num_parallel_t must be >= 1, got {self.flow_config.num_parallel_t}.")
        if self.flow_config.sampling not in ("beta", "uniform"):
            raise ValueError(f"Unsupported flow sampling strategy: {self.flow_config.sampling}")
        # Mutable: overridden at inference time by legendvla_inference_wrapper.
        self.num_inference_steps = self.flow_config.num_inference_steps
        self.knowledge_insulation = knowledge_insulation

        # Shape meta derived
        self.action_dim = int(shape_meta["action"]["shape"][0])
        self.state_dim = int(shape_meta["obs"]["state"]["shape"][0])
        self.horizon_steps = int(shape_meta["action"]["horizon"])
        self.num_state_tokens = int(shape_meta["obs"]["state"]["horizon"])
        self.num_action_tokens = int(shape_meta["action"]["horizon"])
        self.action_horizon = self.num_action_tokens

        # Expert hidden size (explicit parameter, must match action_encoder output width)
        self.action_hidden_size = action_hidden_size

        # Submodules (all pre-instantiated by Hydra)
        self.state_encoder = state_encoder
        self.ar_action_encoder = ar_action_encoder
        self.action_encoder = action_encoder
        self.time_embedding = time_embedding
        self.flow_expert = flow_expert
        self.action_decoder = action_decoder
        self.diffloss = diffloss
        self.reg_action_head = reg_action_head
        self.latent_condition_projector = latent_condition_projector

    def compile_blocks(
        self,
        compile_kwargs: dict[str, Any],
    ) -> None:
        compile_flags = self.resolve_compile_block_flags(compile_kwargs)
        block_compile_kwargs = {
            key: value
            for key, value in compile_kwargs.items()
            if key not in {"vision", "text", "flow", "diffloss"}
        }

        if compile_flags["vision"]:
            compile_module_list(self.backbone.base_model.model.visual.blocks, block_compile_kwargs)
        if compile_flags["text"]:
            compile_module_list(self.backbone.language_model.layers, block_compile_kwargs)
        if compile_flags["flow"]:
            compile_module_list(self.flow_expert.layers, block_compile_kwargs)
        if compile_flags["diffloss"] and self.diffloss is not None:
            self.diffloss.net = torch.compile(self.diffloss.net, **block_compile_kwargs)

    def resolve_compile_block_flags(
        self,
        compile_kwargs: dict[str, Any],
    ) -> dict[str, bool]:
        return {
            "vision": bool(compile_kwargs.get("vision", True)),
            "text": bool(compile_kwargs.get("text", True)),
            "flow": bool(compile_kwargs.get("flow", True)),
            "diffloss": bool(compile_kwargs.get("diffloss", True)),
        }

    def enable_gradient_checkpointing(self, config: dict | None = None) -> None:
        """Enable checkpointing on modules that support it.

        Args:
            config: per-component checkpointing config. Keys:
                text:           {enabled: bool, every_n: int}
                vision:         {enabled: bool, every_n: int}
                action_expert:  {enabled: bool, every_n: int}
                When *config* is None every component is fully checkpointed
                (backward compatible with the old parameterless call).
        """
        if config is None:
            config = {}
        text_cfg = config.get("text", {})
        vision_cfg = config.get("vision", {})
        expert_cfg = config.get("action_expert", {})

        # Defaults: enabled=True, every_n=1 (same as the old behaviour).
        # every_n=0 disables checkpointing for that component.
        text_enabled = text_cfg.get("enabled", True)
        text_every_n = text_cfg.get("every_n", 1) if text_enabled else 0
        vision_enabled = vision_cfg.get("enabled", True)
        vision_every_n = vision_cfg.get("every_n", 1) if vision_enabled else 0
        expert_enabled = expert_cfg.get("enabled", True)
        expert_every_n = expert_cfg.get("every_n", 1) if expert_enabled else 0

        self.backbone.enable_gradient_checkpointing(
            text_every_n=text_every_n,
            vision_every_n=vision_every_n,
        )
        if expert_every_n > 0:
            self.flow_expert.enable_gradient_checkpointing(every_n=expert_every_n)

    def disable_gradient_checkpointing(self) -> None:
        """Disable checkpointing on modules that support it."""
        for module in (self.backbone, self.flow_expert):
            disable_method = getattr(module, "disable_gradient_checkpointing", None)
            if callable(disable_method):
                disable_method()

    @property
    def trainable_vlm_parameters(self):
        return [param for param in self.backbone.parameters() if param.requires_grad]

    @property
    def lora_trainable_vlm_parameters(self):
        return self.trainable_vlm_parameters

    @property
    def action_expert_parameters(self):
        modules = [
            self.action_encoder,
            self.time_embedding,
            self.flow_expert,
            self.action_decoder,
        ]
        camera_encoder = getattr(self, "camera_encoder", None)
        if camera_encoder is not None:
            modules.append(camera_encoder)
        return [param for module in modules for param in module.parameters() if param.requires_grad]

    @property
    def diffloss_parameters(self):
        modules = [
            self.state_encoder,
            self.ar_action_encoder,
            self.latent_condition_projector,
        ]
        if self.diffloss is not None:
            modules.append(self.diffloss)
        if self.reg_action_head is not None:
            modules.append(self.reg_action_head)
        return [param for module in modules for param in module.parameters() if param.requires_grad]

    def build_prefix_lengths(self, batch: dict) -> torch.Tensor:
        answer_start_idx = batch.get("answer_start_idx")
        assert answer_start_idx is not None, (
            "answer_start_idx is required in batch. "
            "Ensure the collator provides it."
        )
        return answer_start_idx.to(device=batch["input_ids"].device, dtype=torch.long)

    def build_action_position_ids(self, batch: dict, action_ref: torch.Tensor) -> torch.Tensor:
        """Build absolute position ids for action tokens.

        Args:
            batch: Collated batch (used by build_prefix_lengths for prefix info).
            action_ref: Any tensor with shape [B, action_len, ...] to derive dimensions from
                (e.g. action_embeds in flow stream, or raw actions in training).
        """
        action_len = action_ref.shape[1]
        device = action_ref.device
        base = self.build_prefix_lengths(batch).to(device=device, dtype=torch.long).unsqueeze(1)
        return base + torch.arange(action_len, device=device).unsqueeze(0)

    def build_slot_embeddings(self, batch: dict, add_action_noise: bool = True) -> dict[str, torch.Tensor | None]:
        slot_embeds: dict[str, torch.Tensor | None] = {"state": None, "action": None, "camera": None}
        
        if self.camera_intrinsic_mode == "token" and self.camera_encoder is not None and "camera_intrinsic" in batch:
            camera_embeds = self.camera_encoder(batch["camera_intrinsic"])
            slot_embeds["camera"] = camera_embeds
        if "states" in batch:
            state_embeds = self.state_encoder(batch["states"])
            slot_embeds["state"] = state_embeds
        if "actions" in batch:
            action_input = batch["actions"]
            if add_action_noise:
                action_input = action_input + torch.randn_like(batch["actions"]) * self.ar_action_train_config.noise_std
            action_embeds = self.ar_action_encoder(action_input)
            slot_embeds["action"] = action_embeds
        return slot_embeds

    def forward_backbone_stream(
        self, batch: dict, slot_embeds: dict, output_attentions: bool = False,
    ) -> BackboneStreamOutput:
        output = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            pixel_values_videos=batch["pixel_values_videos"],
            video_grid_thw=batch["video_grid_thw"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            state_slot_embeds=slot_embeds.get("state"),
            action_slot_embeds=slot_embeds.get("action"),
            camera_slot_embeds=slot_embeds.get("camera"),
            output_attentions=output_attentions,
        )
        output.prefix_cache = slice_prefix_cache_from_full_kv(
            output.past_key_values_hf,
            self.build_prefix_lengths(batch),
        )
        if output.prefix_cache is not None:
            if self.knowledge_insulation is True:
                output.prefix_cache = output.prefix_cache.detach()
            elif isinstance(self.knowledge_insulation, int) and self.knowledge_insulation > 0:
                output.prefix_cache = output.prefix_cache.partial_detach(
                    self.knowledge_insulation
                )
        output.past_key_values_hf = None
        return output

    def forward_flow_stream(
        self,
        batch: dict,
        backbone_output: BackboneStreamOutput,
        flow_inputs: dict,
        num_parallel_chunks: int,
        output_attentions: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        time_for_model = flow_inputs["time_for_model"]
        if time_for_model.ndim != 2:
            raise ValueError(
                f"Expected packed flow time tensor with shape [B, T*H], got {time_for_model.shape}."
            )
        if num_parallel_chunks < 1:
            raise ValueError(f"num_parallel_chunks must be >= 1, got {num_parallel_chunks}.")

        batch_size, seq_len = time_for_model.shape
        time_cond = self.time_embedding(time_for_model.reshape(-1)).reshape(
            batch_size, seq_len, -1,
        )
        action_embeds = self.action_encoder(flow_inputs["noisy_actions"])
        action_mask = batch["actions_valid_mask"].any(dim=-1).to(dtype=torch.bool)
        # actions_valid_mask is always single-chunk [B, H, D]; position ids are
        # built per-chunk then repeated, matching the expanded action_embeds.
        action_position_ids = self.build_action_position_ids(batch, batch["actions_valid_mask"])
        action_mask = action_mask.repeat(1, num_parallel_chunks)
        action_position_ids = action_position_ids.repeat(1, num_parallel_chunks)
        if action_embeds.shape[1] != action_mask.shape[1]:
            raise ValueError(
                f"Action mask length {action_mask.shape[1]} does not match action embeddings length {action_embeds.shape[1]}."
            )
        if action_embeds.shape[1] != action_position_ids.shape[1]:
            raise ValueError(
                "Action position ids length "
                f"{action_position_ids.shape[1]} does not match action embeddings length {action_embeds.shape[1]}."
            )
        prefix_cache = backbone_output.prefix_cache
        expert_output = self.flow_expert(
            action_embeds=action_embeds,
            prefix_cache=prefix_cache,
            action_position_ids=action_position_ids,
            time_cond=time_cond,
            action_mask=action_mask,
            num_parallel_chunks=num_parallel_chunks,
            output_attentions=output_attentions,
        )
        if output_attentions:
            expert_hidden, expert_attn_weights = expert_output
        else:
            expert_hidden = expert_output
            expert_attn_weights = None
        pred_v = self.action_decoder(expert_hidden)
        return {
            "time_cond": time_cond,
            "action_hidden_states": expert_hidden,
            "pred_v": pred_v,
            "expert_attention_weights": expert_attn_weights,
        }

    def compute_loss(self, batch: dict, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        from src.policy.legendvla_loss import compute_total_loss

        return compute_total_loss(self, batch)

    def compute_ar_loss(self, batch: dict, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        from src.policy.legendvla_loss import compute_ar_only_loss

        return compute_ar_only_loss(self, batch)

    def compute_flow_loss(self, batch: dict, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        from src.policy.legendvla_loss import compute_flow_only_loss

        return compute_flow_only_loss(self, batch)

    def freeze_non_lora_weights_in_vlm(self):
        freeze_method = getattr(self.backbone, "freeze_non_lora_parameters", None)
        if callable(freeze_method):
            freeze_method()
            return
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_non_lora_weights_in_ae(self):
        modules = [
            self.state_encoder,
            self.ar_action_encoder,
            self.action_encoder,
            self.time_embedding,
            self.flow_expert,
            self.action_decoder,
        ]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def freeze_weights_in_depth(self):
        return

    def freeze_all_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def infer_action(self, input: dict, **kwargs):
        from src.policy.legendvla_inference import infer_flow_action

        return infer_flow_action(self, input, **kwargs)

    def infer_vla(self, input: dict, **kwargs):
        from src.policy.legendvla_inference import infer_ar_action

        return infer_ar_action(self, input, **kwargs)

    def infer_vlm(self, input: dict, **kwargs):
        from src.policy.legendvla_inference import infer_vlm_generation

        return infer_vlm_generation(self, input, **kwargs)

    def forward(self, mode: str, batch: dict, **kwargs) -> dict[str, torch.Tensor]:
        """Dispatch the top-level LegendVLA execution modes.

        Mode contract:
        - `train`: full multitask loss on one collated batch.
        - `train_ar`: autoregressive language/action loss only.
        - `train_flow`: flow/diffusion action loss only.
        - `infer_action`: continuous action inference from a prepared VLA batch.
        - `infer_vla`: autoregressive action-token decoding from a prepared VLA batch.
        - `infer_vlm`: general VLM text generation from a prepared multimodal batch.

        All modes expect the batch schema produced by the collator and backbone wrappers in this repository rather
        than raw Hugging Face model inputs alone.
        """
        if mode == "train":
            return self.compute_loss(batch, **kwargs)
        if mode == "train_ar":
            return self.compute_ar_loss(batch, **kwargs)
        if mode == "train_flow":
            return self.compute_flow_loss(batch, **kwargs)
        if mode == "infer_action":
            return self.infer_action(batch, **kwargs)
        if mode == "infer_vla":
            return self.infer_vla(batch, **kwargs)
        if mode == "infer_vlm":
            return self.infer_vlm(batch, **kwargs)
        raise ValueError(f"Invalid mode: {mode}")
