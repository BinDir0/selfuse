"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

"""

import logging
from typing import Optional, Tuple

import torch
from torch import nn
from torch._dynamo import disable
import random

from src.model.common.kv_cache import KVCache
from src.model.common.modules import (
    SinusoidalPosEmb,
    TimeEncoder,
)
from src.utils.monitor import log_execution_time

log = logging.getLogger(__name__)


class LegendVLA(nn.Module):
    @log_execution_time(log)
    def __init__(
        self, 
        cfg,
        shape_meta,
        action_encoder_ar, 
        latent_condition_projector,
        action_encoder,
        action_decoder,
        depth_encoder, 
        vision_tower,
        multi_modal_projector,
        joint_model,
        diffloss,
    ):
        super().__init__()
        self.cfg = cfg
        self.shape_meta = shape_meta    
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.state_token_index = cfg.state_token_index
        self.action_token_index = cfg.action_token_index
        self.use_lm_head = cfg.get("use_lm_head", False)
        self.use_action_position_ids_continue_from_vlm = cfg.get(
            "use_action_position_ids_continue_from_vlm", False
        )

        self.max_vlm_tokens = cfg.max_vlm_tokens
        self.num_action_tokens = shape_meta["action"]["horizon"]

        # Get hidden sizes from joint_model config
        self.vlm_hidden_size = joint_model.config.mixture.vlm.hidden_size
        self.action_hidden_size = joint_model.config.mixture.action.hidden_size

        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = shape_meta["action"]["horizon"]
        self.action_dim = shape_meta["action"]["shape"][0]
        self.flow_sig_min = cfg.get("flow_sig_min", 0.001)

        # text input only
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.vlm_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters

        # Vision
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector
        
        # Depth encoder (optional)
        self.use_depth = cfg.use_depth
        if self.use_depth:
            self.depth_dropout = cfg.get("depth_dropout", 0.1)
            self.depth_encoder = depth_encoder
            self.depth_missing_embeddings = nn.Parameter(
                torch.zeros(self.depth_encoder.depth_seq_len, self.depth_encoder.output_dim)
            )

        # Mixtures
        self.joint_model = joint_model

        # Diffusion loss
        self.diffloss = diffloss
        self.diffloss_micro_batch_size = cfg.get("diffloss_micro_batch_size", 4)
        self.ar_action_noise_std = cfg.get("ar_action_noise_std", 0.02)
        self.ar_action_chunk_size = cfg.get("ar_action_chunk_size", 4)

        # Action, time encoders
        self.action_expert_adaptive_mode = cfg.action_expert_adaptive_mode
        if self.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.time_embedding = nn.Sequential(
                SinusoidalPosEmb(cfg.time_hidden_size, cfg.time_min_period, cfg.time_max_period), 
                TimeEncoder(cfg.time_hidden_size), 
            )
        else:  # matching pi0
            self.time_embedding = SinusoidalPosEmb(
                self.action_hidden_size, cfg.time_max_period
            )
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder

        # Action/state encoder for continuous autoregressive modeling
        self.action_encoder_ar = action_encoder_ar
        # Latent condition projector for continuous autoregressive modeling
        self.latent_condition_projector = latent_condition_projector

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.vlm_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights

        # Gemma2-specific: Final logit softcapping for numerical stability
        self.final_logit_softcapping = cfg.get("final_logit_softcapping", None)

        self.CELoss = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index, reduction='sum')
        self.ignore_index = cfg.ignore_index
        self.loss_weights = cfg.loss_weights

    def _apply_final_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply final logit softcapping (Gemma2 feature).
        
        Limits the range of output logits to prevent extreme values that could cause
        numerical instability during training or inference.
        
        Args:
            logits (torch.Tensor): Raw logits from language model head
        
        Returns:
            torch.Tensor: Softcapped logits (same shape as input)
        """
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits

    @property
    def attn_weights(self):
        """
        Get all attention weights for the joint model.
        """
        return self.joint_model.attn_weights

    @property
    def action_expert_parameters(self):
        """
        Get all trainable parameters for the action experts.
        
        Returns:
            List[torch.nn.Parameter]: Parameters from:
                - Action encoder
                - Action decoder  
                - Action mixture
        
        """
        return (
            list(self.action_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.joint_model.mixtures["action"].parameters())
            + list(self.time_embedding.parameters())
        )

    @property
    def trainable_vlm_parameters(self):
        """
        Get all trainable parameters for the VLM components.
        
        Returns:
            List[torch.nn.Parameter]: Parameters from:
                - Vision tower (SigLIP)
                - Multi-modal projector
                - Trainable Gemma parameters
        """
        return (
            list(self.vision_tower.parameters())
            + list(self.multi_modal_projector.parameters())
            + self.trainable_gemma_parameters
            + self.trainable_depth_parameters
        )

    @property
    def trainable_depth_parameters(self):
        """
        Get all trainable parameters for the depth encoder.
        """
        if not self.use_depth:
            return []
        return (
            list(self.depth_encoder.parameters())
            + [self.depth_missing_embeddings]
        )

    @property
    def lora_trainable_vlm_parameters(self):
        """
        Get all LoRA trainable parameters for the VLM components.
        
        Returns:
            List[torch.nn.Parameter]: LoRA parameters from:
                - Vision tower (SigLIP)
                - Multi-modal projector
                - Gemma language model
        """
        params = []
        for name, param in self.vision_tower.named_parameters():
            if "lora_" in name:
                params.append(param)
        for name, param in self.multi_modal_projector.named_parameters():
            if "lora_" in name:
                params.append(param)
        params.extend(self.trainable_lora_gemma_parameters)

        params.extend(list(self.embed_tokens.parameters()))
        return params

    @property
    def trainable_gemma_parameters(self):
        """
        Get all trainable parameters for the Gemma language model.
        
        Returns:
            List[torch.nn.Parameter]: Trainable Gemma parameters
        """
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            gemma_parameters.append(param)
        
        gemma_parameters.extend(list(self.embed_tokens.parameters()))
        return gemma_parameters

    @property
    def trainable_lora_gemma_parameters(self):
        """
        Get all LoRA trainable parameters for the Gemma language model.
        
        Excludes unused parameters and only includes LoRA parameters.
        
        Returns:
            List[torch.nn.Parameter]: Trainable LoRA Gemma parameters
        """
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if "lora_" in name:
                gemma_parameters.append(param)

        gemma_parameters.extend(list(self.embed_tokens.parameters()))
        return gemma_parameters
    
    @property
    def diffloss_parameters(self):
        """
        Get all trainable parameters for the DiffLoss module.
        
        Returns:
            List[torch.nn.Parameter]: Trainable DiffLoss parameters
        """
        return list(self.diffloss.parameters()) \
            + list(self.action_encoder_ar.parameters()) \
            + list(self.latent_condition_projector.parameters())

    @torch.no_grad()
    def init_motion_token_embeddings(self, motion_token_list):
        """
        Initialize the motion token embeddings.

        Args:
            motion_token_list: List of motion token IDs
        """
        from src.policy.legendvla_utils import init_motion_token_embeddings as _init_motion_token_embeddings
        _init_motion_token_embeddings(self.embed_tokens, self.vlm_hidden_size, motion_token_list)

    @log_execution_time(log)
    def load_pretrained_vlm_weights(self):
        """
        Load pre-trained weights from PaliGemma checkpoint.

        Loads weights for:
        - Vision tower (SigLIP)
        - Multi-modal projector
        - Language model (Gemma)
        - Text embeddings

        The weights are loaded from safetensors files in the pretrained_model_path.
        LoRA weights are preserved and not overwritten.
        """
        from src.policy.legendvla_utils import load_pretrained_vlm_weights as _load_pretrained_vlm_weights
        _load_pretrained_vlm_weights(self)

    @log_execution_time(log)
    def load_pretrained_pi05_weights(self):
        """
        Load pre-trained weights from Pi0.5 checkpoint.

        Loads weights for:
        - Vision tower (SigLIP)
        - Multi-modal projector
        - Language model (Gemma 2B) - VLM mixture
        - Action expert (Gemma 300M) - Action mixture
        - LM head
        - Time embedding MLPs

        Skips only:
        - action_in_proj (action encoder, incompatible dimensions)
        - action_out_proj (action decoder, incompatible dimensions)

        The weights are loaded from safetensors file in the pretrained_model_path.
        LoRA weights are preserved and not overwritten.
        """
        from src.policy.legendvla_utils import load_pretrained_pi05_weights as _load_pretrained_pi05_weights
        _load_pretrained_pi05_weights(self)

    def freeze_non_lora_weights_in_vlm(self):
        """
        Freeze non-LoRA weights in VLM components while keeping LoRA weights trainable.

        This method freezes:
        - Vision tower weights (except LoRA)
        - Multi-modal projector weights (except LoRA)
        - Language model weights (except LoRA)
        - Token embeddings

        Only LoRA parameters remain trainable for efficient fine-tuning.
        """
        from src.policy.legendvla_utils import freeze_non_lora_weights_in_vlm as _freeze_non_lora_weights_in_vlm
        _freeze_non_lora_weights_in_vlm(self.vision_tower, self.multi_modal_projector, self.joint_model, self.embed_tokens)

    def freeze_non_lora_weights_in_ae(self):
        """
        Freeze non-LoRA weights in VLM components while keeping LoRA weights trainable.

        This method freezes:
        - Action encoder weights (except LoRA)
        - Action decoder weights (except LoRA)
        - Action mixture weights (except LoRA)

        Only LoRA parameters remain trainable for efficient fine-tuning.
        """
        from src.policy.legendvla_utils import freeze_non_lora_weights_in_ae as _freeze_non_lora_weights_in_ae
        _freeze_non_lora_weights_in_ae(self.action_encoder, self.action_decoder, self.joint_model)

    def freeze_weights_in_depth(self):
        """
        Freeze weights in depth encoder and depth missing embeddings.
        """
        if self.use_depth:
            from src.policy.legendvla_utils import freeze_weights_in_depth as _freeze_weights_in_depth
            _freeze_weights_in_depth(self.depth_encoder, self.depth_missing_embeddings)

    def freeze_all_weights(self):
        """
        Freeze all trainable parameters in the model.

        Sets requires_grad=False for all parameters, making the model non-trainable.
        Useful for inference-only scenarios.
        """
        from src.policy.legendvla_utils import freeze_all_weights as _freeze_all_weights
        _freeze_all_weights(self)

    def build_text_cache(self):
        """
        Create a new KV cache for text generation.

        Returns:
            KVCache: Empty key-value cache for storing attention states during text generation
        """
        from src.policy.legendvla_utils import build_text_cache as _build_text_cache
        return _build_text_cache()

    # ---------- Input preparation ---------- #
    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, answer_start_idx: torch.Tensor, n_actions: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        """
        Build causal attention masks and position IDs for different token types.
        Delegates to standalone function in legendvla_utils.
        """
        from src.policy.legendvla_utils import build_causal_mask_and_position_ids as _build_causal_mask_and_position_ids
        return _build_causal_mask_and_position_ids(
            attention_mask, answer_start_idx, n_actions, self.num_action_tokens, dtype
        )

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor, max_vlm_tokens: int
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Split the full causal mask into separate masks for different model components.
        Delegates to standalone function in legendvla_utils.
        """
        from src.policy.legendvla_utils import split_full_mask_into_submasks as _split_full_mask_into_submasks
        return _split_full_mask_into_submasks(causal_mask, max_vlm_tokens, self.num_action_tokens)

    def build_causal_mask_and_position_ids_for_text(
        self,
        q_len: int,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Build causal mask and position IDs for autoregressive generation.
        Delegates to standalone function in legendvla_utils.
        """
        from src.policy.legendvla_utils import build_causal_mask_and_position_ids_for_text as _build_causal_mask_and_position_ids_for_text
        return _build_causal_mask_and_position_ids_for_text(q_len, attention_mask, kv_cache, dtype)

    # ---------- Inference ----------#
    @disable(recursive=False)
    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor = None,
        depth_values: Optional[torch.FloatTensor] = None,
        has_depth_values: Optional[torch.LongTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        n_states: Optional[torch.LongTensor] = None,
        n_actions: Optional[torch.LongTensor] = None,
        is_vla_data = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.FloatTensor:
        """
        Forward pass through SigLIP vision encoder and text embedding, then combine them.
        
        Args:
            input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
            pixel_values (torch.FloatTensor): [B, C, H, W] or [B, T, C, H, W] Image pixel values (normalized)
            depth_values (Optional[torch.FloatTensor]): [Bd, C, H, W] or [Bd, T, C, H, W] Depth images (optional)
            has_depth_values (Optional[torch.LongTensor]): [B] Bool data indicating whether this sample has valid depth (optional)
            states: [B, state_len, state_dim]
            actions: [B, action_len, action_dim]
            n_states: [B]
            n_actions: [B]
            is_vla_data: [B]
            dtype: torch.dtype
        
        Returns:
            torch.FloatTensor: [B, seq_len, hidden_size] Combined image and text embeddings
        """
        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.embed_tokens(input_ids)
        device = inputs_embeds.device

        if pixel_values is not None:
            # image features from siglip and projector
            # [Batch_Size, Channels, Height, Width] or [Batch_Size, Time, Channels, Height, Width] 
            # -> [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
            if pixel_values.ndim == 5:
                B, T, C, H, W = pixel_values.shape
                # pixel_values = rearrange(pixel_values, "B T C H W -> (B T) C H W")
                pixel_values = pixel_values.view(B * T, C, H, W)
            else:
                T = None

            # Extract RGB vision features
            rgb_image_features = self.vision_tower(pixel_values)
            
            # Extract depth features if enabled
            if self.use_depth and depth_values is not None:
                # Handle depth images similar to pixel_values
                if depth_values.ndim == 5:
                    Bd, T, C, H, W = depth_values.shape
                    # pixel_values = rearrange(pixel_values, "B T C H W -> (B T) C H W")
                    depth_values = depth_values.view(Bd * T, C, H, W)
                else:
                    T = None
                
                # Extract depth features using DINOv2
                depth_image_features = self.depth_encoder(depth_values)
            else: 
                depth_image_features = None

            if T is not None:
                # image_features = rearrange(image_features, "(B T) P D -> B (T P) D", B=B, T=T)
                rgb_image_features = rgb_image_features.view(B, -1, rgb_image_features.shape[-1])
                if depth_image_features is not None:
                    depth_image_features = depth_image_features.view(Bd, -1, depth_image_features.shape[-1])

        # normalize the image features
        bsz, seq_len = input_ids.shape

        # put embedding together - image, text, answer, padding
        final_embedding = torch.full(
            (bsz, seq_len, self.vlm_hidden_size), 0, dtype=dtype, device=device
        )

        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        final_embedding[text_mask] = inputs_embeds[text_mask].to(final_embedding.dtype)
        state_mask = input_ids == self.state_token_index
        action_mask = input_ids == self.action_token_index
        if n_states is not None:
            assert torch.all(n_states == state_mask.sum(dim=1))
        if n_actions is not None: 
            assert torch.all(n_actions == action_mask.sum(dim=1))
        # The features will be scaled internally in the joint model
        if states is not None:
            state_features = self.action_encoder_ar(states) / (self.vlm_hidden_size**0.5)
        if actions is not None:
            actions_input = actions
            noise = torch.randn_like(actions_input) * self.ar_action_noise_std
            actions_input = actions_input + noise
            action_features = self.action_encoder_ar(actions_input) / (self.vlm_hidden_size**0.5)
        if pixel_values is not None:
            image_mask = input_ids == self.image_token_index
            # autocast does not cast nn.Embedding to the correct dtype, we need to cast manually
            
        for i in range(bsz):
            if pixel_values is not None: 
                image_indices = image_mask[i].nonzero(as_tuple=True)[0]
                if depth_image_features is None:
                    depth_image_feature = None
                elif has_depth_values is not None and has_depth_values[i] and \
                    not (self.training and random.random() < self.depth_dropout):
                    # Each RGB token is paired with corresponding depth token
                    depth_image_feature = depth_image_features[i]
                else: 
                    if T is not None:
                        depth_image_feature = self.depth_missing_embeddings.repeat(T, 1)
                    else:
                        depth_image_feature = self.depth_missing_embeddings
                if depth_image_feature is not None:
                    paired_image_features = torch.cat([
                        rgb_image_features[i], depth_image_feature
                    ], dim=-1) # [num_patches, rgb_embed_dim+depth_embed_dim]
                else:  
                    paired_image_features = rgb_image_features[i]
                paired_image_features = paired_image_features.view(-1, paired_image_features.shape[-1])
                paired_image_features = self.multi_modal_projector(paired_image_features)
                scaled_image_features = paired_image_features / (self.vlm_hidden_size**0.5)
                final_embedding[i, image_indices] = scaled_image_features
            if is_vla_data is not None and is_vla_data[i]:
                if n_states is not None:
                    final_embedding[i, state_mask[i]] = state_features[i, :n_states[i]].to(final_embedding.dtype)
                if n_actions is not None:
                    final_embedding[i, action_mask[i]] = action_features[i, :n_actions[i]].to(final_embedding.dtype)
        return final_embedding

    @torch.inference_mode()
    def infer_action(self, input: dict, return_attn_weights: bool = False) -> torch.FloatTensor:
        from src.policy.legendvla_inference import infer_action as _infer_action
        return _infer_action(self, input, return_attn_weights)

    @torch.inference_mode()
    def infer_single_step(self, input: dict, kv_cache=None, dtype=torch.float32, return_attn_weights=False) -> dict:
        from src.policy.legendvla_inference import infer_single_step as _infer_single_step
        return _infer_single_step(self, input, kv_cache, dtype, return_attn_weights)

    @torch.inference_mode()
    def infer_vlm(self, input: dict, max_new_tokens: int, temperature: float = 1.0,
                  top_k: int = 10, top_p: float = 1.0, allowed_token_ids=None,
                  eos_token_id=None, return_kv_cache: bool = False,
                  return_attn_weights: bool = False) -> dict:
        from src.policy.legendvla_inference import infer_vlm as _infer_vlm
        return _infer_vlm(self, input, max_new_tokens, temperature, top_k, top_p,
                          allowed_token_ids, eos_token_id, return_kv_cache, return_attn_weights)

    @torch.inference_mode()
    def infer_vla(self, input: dict, max_new_tokens: int, temperature: float = 1.0,
                  return_attn_weights: bool = False, cfg: float = 1.0, **kwargs) -> dict:
        from src.policy.legendvla_inference import infer_vla as _infer_vla
        return _infer_vla(self, input, max_new_tokens, temperature, return_attn_weights, cfg, **kwargs)

    # ---------- Flow matching training ----------#
    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        from src.policy.legendvla_loss import psi_t as _psi_t
        return _psi_t(x, x1, t, self.flow_sig_min)

    # TODO: Deprecated method, to be updated
    def compute_ar_loss(self, batch: dict) -> dict:
        from src.policy.legendvla_loss import compute_ar_loss as _compute_ar_loss
        return _compute_ar_loss(self, batch)

    # TODO: Deprecated method, to be updated
    def compute_flow_loss(self, batch: dict) -> dict:
        from src.policy.legendvla_loss import compute_flow_loss as _compute_flow_loss
        return _compute_flow_loss(self, batch)

    @torch.compile
    def compute_celoss(self, hidden_states: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        from src.policy.legendvla_loss import compute_celoss as _compute_celoss
        return _compute_celoss(
            self.lm_head, self.final_logit_softcapping,
            self.CELoss, self.ignore_index, hidden_states, labels
        )

    def compute_loss(self, batch: dict) -> dict:
        from src.policy.legendvla_loss import compute_loss as _compute_loss
        return _compute_loss(self, batch)

    def forward(self, mode: str, batch: dict, **kwargs) -> dict:
        from src.policy.legendvla_loss import compute_loss, compute_ar_loss, compute_flow_loss
        from src.policy.legendvla_inference import infer_action, infer_vlm, infer_vla
        if mode == "train":
            return compute_loss(self, batch, **kwargs)
        elif mode == "train_ar":
            return compute_ar_loss(self, batch, **kwargs)
        elif mode == "train_flow":
            return compute_flow_loss(self, batch, **kwargs)
        elif mode == "infer_action":
            return infer_action(self, batch, **kwargs)
        elif mode == "infer_vla":
            return infer_vla(self, batch, **kwargs)
        elif mode == "infer_vlm":
            return infer_vlm(self, batch, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        

class LegendVLAInference(nn.Module):
    """
    Implementation of the VLA inference logic.
    This class is 'Device-Agnostic' - it focuses on the sequence of operations:
    Observation -> Preprocessing -> State Normalization -> Model Forward -> Action Unnormalization.

    Moved to src/policy/legendvla_inference.py. This import alias preserves backward compatibility.
    """
    def __new__(cls, *args, **kwargs):
        from src.policy.legendvla_inference import LegendVLAInference as _LegendVLAInference
        return _LegendVLAInference(*args, **kwargs)
