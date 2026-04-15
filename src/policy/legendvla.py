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
from src.utils.sample_utils import generate_multi_span_mask


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
    wm_loss_weight: float = 0.0


@dataclass(frozen=True)
class SpanMaskConfig:
    mask_ratio: float = 0.4
    mean_span_len: float = 4.0
    start_bias_alpha: float = 1.5
    p_no_mask: float = 0.1
    keep_last: bool = False
    prefer_early: bool = True


class WorldModelHead(nn.Module):
    """Query tokens + output projection for world model prediction."""

    def __init__(self, n_queries: int, hidden_size: int, upsample_factor: int):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(n_queries, hidden_size) * 0.02)
        self.output_proj = nn.Linear(hidden_size, hidden_size * upsample_factor ** 2)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)


@dataclass(frozen=True)
class WorldModelConfig:
    num_future_frames: int = 0
    target_image_size: tuple[int, int] | None = None
    teacher_patch_size: int = 16
    upsample_factor: int = 2
    action_conditioning: bool = False
    detach_action_cond: bool = False


@dataclass(frozen=True)
class ARActionTrainConfig:
    noise_std: float = 0.02
    chunk_size: int = 4
    diffloss_repeat: int = 1
    input_mask_enabled: bool = False
    action_mask: SpanMaskConfig = SpanMaskConfig()
    state_mask: SpanMaskConfig = SpanMaskConfig(
        mask_ratio=0.65, mean_span_len=5.0, start_bias_alpha=2.0,
        p_no_mask=0.1, keep_last=True, prefer_early=False,
    )


class InputMaskEmbeddings(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.action = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.state = nn.Parameter(torch.randn(hidden_size) * 0.02)


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
        # World model (optional)
        world_model_expert: nn.Module | None = None,
        frozen_teacher: nn.Module | None = None,
        world_model_config: WorldModelConfig = WorldModelConfig(),
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

        # Mask embeddings only needed when input_mask_enabled is True.
        if self.ar_action_train_config.input_mask_enabled:
            self.input_mask_embeddings = InputMaskEmbeddings(self.vlm_hidden_size)

        # World model components
        self.world_model_expert = world_model_expert
        self.frozen_teacher = frozen_teacher
        self.world_model_config = world_model_config
        self.use_world_model = world_model_expert is not None and frozen_teacher is not None
        if self.use_world_model:
            self._init_world_model(world_model_expert, world_model_config)

    def _init_world_model(self, expert: nn.Module, config: WorldModelConfig) -> None:
        tH, tW = config.target_image_size
        stride = config.teacher_patch_size * config.upsample_factor
        assert tH % stride == 0 and tW % stride == 0, (
            f"target ({tH},{tW}) must be divisible by patch*upsample={stride}"
        )
        self.wm_grid_h = tH // stride
        self.wm_grid_w = tW // stride
        self.wm_upsample_factor = config.upsample_factor
        self.wm_num_future_frames = config.num_future_frames

        D = expert.hidden_size
        n_queries = config.num_future_frames * self.wm_grid_h * self.wm_grid_w
        self.wm_head = WorldModelHead(n_queries, D, config.upsample_factor)

    def compile_blocks(
        self,
        compile_kwargs: dict[str, Any],
    ) -> None:
        compile_flags = self.resolve_compile_block_flags(compile_kwargs)
        block_compile_kwargs = {
            key: value
            for key, value in compile_kwargs.items()
            if key not in {"vision", "text", "flow", "diffloss", "world_model"}
        }

        if compile_flags["vision"]:
            compile_module_list(self.backbone.base_model.model.visual.blocks, block_compile_kwargs)
        if compile_flags["text"]:
            compile_module_list(self.backbone.language_model.layers, block_compile_kwargs)
        if compile_flags["flow"]:
            compile_module_list(self.flow_expert.layers, block_compile_kwargs)
        if compile_flags["diffloss"] and self.diffloss is not None:
            self.diffloss.net = torch.compile(self.diffloss.net, **block_compile_kwargs)
        if compile_flags["world_model"] and self.use_world_model:
            compile_module_list(self.world_model_expert.layers, block_compile_kwargs)
            # DINOv3 ViT layer list lives at `model.model.layer`.
            compile_module_list(self.frozen_teacher.model.model.layer, block_compile_kwargs)

    def resolve_compile_block_flags(
        self,
        compile_kwargs: dict[str, Any],
    ) -> dict[str, bool]:
        return {
            "vision": bool(compile_kwargs.get("vision", True)),
            "text": bool(compile_kwargs.get("text", True)),
            "flow": bool(compile_kwargs.get("flow", True)),
            "diffloss": bool(compile_kwargs.get("diffloss", True)),
            "world_model": bool(compile_kwargs.get("world_model", True)),
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
    def trainable_vision_parameters(self):
        visual = getattr(self.backbone.base_model.model, "visual", None)
        if visual is None:
            return []
        return [param for param in visual.parameters() if param.requires_grad]

    @property
    def trainable_text_parameters(self):
        # Language model + lm_head + embedding; exclude vision tower
        vision_ids = {id(p) for p in (getattr(self.backbone.base_model.model, "visual", None) or nn.Module()).parameters()}
        return [param for param in self.backbone.parameters() if param.requires_grad and id(param) not in vision_ids]

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
        if hasattr(self, "input_mask_embeddings"):
            modules.append(self.input_mask_embeddings)
        return [param for module in modules for param in module.parameters() if param.requires_grad]

    @property
    def world_model_parameters(self):
        params = []
        if self.use_world_model:
            params.extend(p for p in self.world_model_expert.parameters() if p.requires_grad)
            params.extend(p for p in self.wm_head.parameters() if p.requires_grad)
        return params

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
        slot_embeds: dict[str, torch.Tensor | None] = {
            "state": None, "action": None, "camera": None,
        }

        if self.camera_intrinsic_mode == "token" and self.camera_encoder is not None and "camera_intrinsic" in batch:
            camera_embeds = self.camera_encoder(batch["camera_intrinsic"])
            slot_embeds["camera"] = camera_embeds

        mask_cfg = self.ar_action_train_config
        do_mask = self.training and mask_cfg.input_mask_enabled

        if "states" in batch:
            state_embeds = self.state_encoder(batch["states"])  # [B, H_s, vlm_hidden_size]
            if do_mask:
                state_mask = generate_multi_span_mask(
                    state_embeds.shape[0], self.num_state_tokens,
                    mask_cfg.state_mask, device=state_embeds.device,
                )
                state_embeds = torch.where(
                    state_mask.unsqueeze(-1),
                    self.input_mask_embeddings.state,
                    state_embeds,
                )
            slot_embeds["state"] = state_embeds

        if "actions" in batch:
            action_input = batch["actions"]
            if add_action_noise:
                action_input = action_input + torch.randn_like(batch["actions"]) * mask_cfg.noise_std
            action_embeds = self.ar_action_encoder(action_input)  # [B, H_a, vlm_hidden_size]
            if do_mask:
                action_mask = generate_multi_span_mask(
                    action_embeds.shape[0], self.num_action_tokens,
                    mask_cfg.action_mask, device=action_embeds.device,
                )
                action_embeds = torch.where(
                    action_mask.unsqueeze(-1),
                    self.input_mask_embeddings.action,
                    action_embeds,
                )
            slot_embeds["action"] = action_embeds

        return slot_embeds

    def forward_backbone_stream(
        self, batch: dict, slot_embeds: dict, output_attentions: bool = False,
    ) -> BackboneStreamOutput:
        # is_vla_mask is consumed by the MEM temporal attention path when
        # mask_non_vla=True; backbone ignores it otherwise.
        is_vla_data = batch.get("is_vla_data")
        is_vla_mask = is_vla_data.to(dtype=torch.bool) if is_vla_data is not None else None
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
            is_vla_mask=is_vla_mask,
        )
        output.prefix_cache = slice_prefix_cache_from_full_kv(
            output.past_key_values_hf,
            self.build_prefix_lengths(batch),
        )
        # KV cache is never detached at the backbone level.
        # Each expert controls its own detach via detach_prefix_kv.
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
            suffix_embeds=action_embeds,
            prefix_cache=prefix_cache,
            suffix_position_ids=action_position_ids,
            cond=time_cond,
            suffix_mask=action_mask,
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

    def forward_world_model_stream(
        self,
        batch: dict,
        backbone_output: BackboneStreamOutput,
        action_cond_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run world model expert and frozen teacher on future frames.

        Args:
            action_cond_embeds: Optional [B, H, D] action embeddings from
                action_encoder(clean_actions). Prepended to WM queries so the
                expert can attend to action tokens when predicting future frames.

        Returns dict with ``pred`` and ``target`` feature maps, plus ``n_future_frames``.
        """
        B = backbone_output.last_hidden_states.shape[0]
        queries = self.wm_head.query_embed.unsqueeze(0).expand(B, -1, -1)

        # Action conditioning: prepend encoded clean actions before WM queries
        if action_cond_embeds is not None:
            if self.world_model_config.detach_action_cond:
                action_cond_embeds = action_cond_embeds.detach()
            suffix = torch.cat([action_cond_embeds, queries], dim=1)
            action_len = action_cond_embeds.shape[1]
        else:
            suffix = queries
            action_len = 0

        # Position IDs: sequential from prefix_len, action tokens then queries
        prefix_lengths = self.build_prefix_lengths(batch)
        base = prefix_lengths.unsqueeze(1).to(device=suffix.device, dtype=torch.long)
        position_ids = base + torch.arange(suffix.shape[1], device=suffix.device).unsqueeze(0)

        # Build suffix_mask from real validity
        # Action portion: [B, H] — valid where any action dim is non-zero
        if action_cond_embeds is not None:
            action_mask = batch["actions_valid_mask"].any(dim=-1).to(dtype=torch.bool)
        else:
            action_mask = None

        # Query portion: [B, K*gh*gw] — valid for frames k < n_future_frames[b]
        K = self.wm_num_future_frames
        queries_per_frame = self.wm_grid_h * self.wm_grid_w
        n_future = batch["n_future_frames"].to(device=suffix.device)
        frame_valid = torch.arange(K, device=suffix.device).unsqueeze(0) < n_future.unsqueeze(1)
        query_mask = frame_valid.unsqueeze(-1).expand(-1, -1, queries_per_frame).reshape(B, -1)

        if action_mask is not None:
            suffix_mask = torch.cat([action_mask, query_mask], dim=1)
        else:
            suffix_mask = query_mask

        wm_hidden = self.world_model_expert(
            suffix_embeds=suffix,
            prefix_cache=backbone_output.prefix_cache,
            suffix_position_ids=position_ids,
            suffix_mask=suffix_mask,
        )

        # Extract only query portion (discard action token outputs)
        wm_hidden = wm_hidden[:, action_len:, :]

        # Depth-to-space upsample (inverse of Qwen3-VL spatial merge).
        # Source: transformers Qwen3VLVisionModel.fast_pos_embed_interpolate
        K = self.wm_num_future_frames
        gh, gw, uf = self.wm_grid_h, self.wm_grid_w, self.wm_upsample_factor
        D = self.world_model_expert.hidden_size
        x = self.wm_head.output_proj(wm_hidden)
        x = x.reshape(B * K, gh, gw, uf, uf, D)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * K, gh * uf, gw * uf, D)
        pred = x.flatten(1, 2).reshape(B, K, -1, D)

        target = self.frozen_teacher(batch["future_frames"])

        return {
            "pred": pred,
            "target": target,
            "n_future_frames": batch["n_future_frames"],
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
