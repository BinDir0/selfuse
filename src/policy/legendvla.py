from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from src.utils.compile_utils import compile_module_list
from src.model.vlm.prefix_cache import (
    BackboneStreamOutput,
    gather_action_position_ids,
    slice_prefix_cache_from_full_kv,
)


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
        ignore_index: int = -100,
        action_hidden_size: int = 1024,
        flow_sig_min: float = 0.001,
        num_inference_steps: int = 10,
        ar_action_noise_std: float = 0.02,
        ar_action_chunk_size: int = 4,
        diffloss_micro_batch_size: int = 16,
        use_rtc: bool = True,
        rtc_delay_strategy: str = "exp",
        rtc_max_delay: int = 16,
        ce_loss_weight: float = 0.1,
        diffusion_loss_weight: float = 1.0,
        flow_loss_weight: float = 1.0,
        knowledge_insulation: bool | int = True,
    ):
        super().__init__()
        self.shape_meta = shape_meta

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

        # Scalar config
        self.ignore_index = ignore_index
        self.CELoss = nn.CrossEntropyLoss(reduction="sum", ignore_index=ignore_index)
        self.flow_sig_min = flow_sig_min
        self.num_inference_steps = num_inference_steps
        self.ar_action_noise_std = ar_action_noise_std
        self.ar_action_chunk_size = ar_action_chunk_size
        self.diffloss_micro_batch_size = diffloss_micro_batch_size
        self.use_rtc = use_rtc
        self.rtc_delay_strategy = rtc_delay_strategy
        self.rtc_max_delay = rtc_max_delay
        self.knowledge_insulation = knowledge_insulation
        self.loss_weights = SimpleNamespace(
            ce_loss_weight=ce_loss_weight,
            diffusion_loss_weight=diffusion_loss_weight,
            flow_loss_weight=flow_loss_weight,
        )

        # Shape meta derived
        self.action_dim = int(shape_meta["action"]["shape"][0])
        self.state_dim = int(shape_meta["obs"]["state"]["shape"][0])
        self.horizon_steps = int(shape_meta["action"]["horizon"])
        self.num_state_tokens = int(shape_meta["obs"]["state"]["horizon"])
        self.num_action_tokens = int(shape_meta["action"]["horizon"])

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
            "vision": bool(compile_kwargs.get("vision", False)),
            "text": bool(compile_kwargs.get("text", False)),
            "flow": bool(compile_kwargs.get("flow", False)),
            "diffloss": bool(compile_kwargs.get("diffloss", False)),
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
            self.state_encoder,
            self.ar_action_encoder,
            self.action_encoder,
            self.time_embedding,
            self.flow_expert,
            self.action_decoder,
        ]
        return [param for module in modules for param in module.parameters() if param.requires_grad]

    @property
    def diffloss_parameters(self):
        modules = [self.latent_condition_projector]
        if self.diffloss is not None:
            modules.append(self.diffloss)
        return [param for module in modules for param in module.parameters() if param.requires_grad]

    def build_prefix_lengths(self, batch: dict) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        answer_start_idx = batch.get("answer_start_idx")
        action_mask = input_ids == self.action_token_index
        has_action_tokens = action_mask.any(dim=1)

        if answer_start_idx is not None:
            prefix_lengths = answer_start_idx.to(device=input_ids.device, dtype=torch.long)
        elif attention_mask is not None:
            prefix_lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
        else:
            prefix_lengths = torch.full(
                (input_ids.shape[0],),
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )

        if torch.any(has_action_tokens):
            action_start_idx = action_mask.to(dtype=torch.long).argmax(dim=1)
            prefix_lengths = torch.where(has_action_tokens, action_start_idx, prefix_lengths)
        return prefix_lengths

    def build_action_position_ids(
        self,
        batch: dict,
        backbone_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if "actions" not in batch:
            raise ValueError("Action position ids require an `actions` tensor in the batch.")

        n_actions = batch.get("n_actions")
        if n_actions is None:
            n_actions = torch.full(
                (batch["actions"].shape[0],),
                batch["actions"].shape[1],
                dtype=torch.long,
                device=batch["actions"].device,
            )

        gathered = gather_action_position_ids(
            input_ids=batch["input_ids"],
            action_token_id=self.action_token_index,
            position_ids=backbone_position_ids,
            n_actions=n_actions,
            action_len=batch["actions"].shape[1],
        )
        if gathered.numel() > 0:
            return gathered

        batch_size = batch["actions"].shape[0]
        action_len = batch["actions"].shape[1]
        device = batch["actions"].device
        base = self.build_prefix_lengths(batch).to(device=device, dtype=torch.long).unsqueeze(1)
        positions = base + torch.arange(action_len, device=device).unsqueeze(0)
        return positions.expand(batch_size, -1)

    def build_slot_embeddings(self, batch: dict, add_action_noise: bool = True) -> dict[str, torch.Tensor | None]:
        slot_embeds: dict[str, torch.Tensor | None] = {"state": None, "action": None}
        if "states" in batch:
            state_embeds = self.state_encoder(batch["states"])
            slot_embeds["state"] = state_embeds
        if "actions" in batch:
            action_input = batch["actions"]
            if add_action_noise:
                action_input = action_input + torch.randn_like(batch["actions"]) * self.ar_action_noise_std
            action_embeds = self.ar_action_encoder(action_input)
            slot_embeds["action"] = action_embeds
        return slot_embeds

    def forward_backbone_stream(self, batch: dict, slot_embeds: dict) -> BackboneStreamOutput:
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
    ) -> dict[str, torch.Tensor | None]:
        time_for_model = flow_inputs["time_for_model"]
        if time_for_model.ndim == 2 and self.use_rtc:
            time_cond = self.time_embedding(time_for_model.reshape(-1)).reshape(
                time_for_model.shape[0],
                time_for_model.shape[1],
                -1,
            )
        else:
            time_cond = self.time_embedding(time_for_model)
        action_embeds = self.action_encoder(flow_inputs["noisy_actions"])
        action_mask = batch["actions_valid_mask"].any(dim=-1).to(dtype=torch.bool)
        action_position_ids = self.build_action_position_ids(batch, backbone_output.position_ids)
        prefix_cache = backbone_output.prefix_cache
        expert_hidden = self.flow_expert(
            action_embeds=action_embeds,
            prefix_cache=prefix_cache,
            action_position_ids=action_position_ids,
            time_cond=time_cond,
            action_mask=action_mask,
            mode="flow",
        )
        pred_v = self.action_decoder(expert_hidden)
        return {
            "time_cond": time_cond,
            "action_hidden_states": expert_hidden,
            "pred_v": pred_v,
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
