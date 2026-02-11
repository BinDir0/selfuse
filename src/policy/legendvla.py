"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

"""

import logging
import pathlib
import pickle
from typing import Any, Optional, Tuple, List, Union, Dict

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn
from torch._dynamo import disable
import random

from src.utils.pytorch_util import dict_apply
from src.model.common.kv_cache import KVCache
from src.model.common.modules import (
    SinusoidalPosEmb,
    TimeEncoder,
)
from src.utils.monitor import log_execution_time
from src.utils.generation_utils import sample_token, concat_attn_weights

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
        device = self.embed_tokens.weight.device
        indices = torch.LongTensor(motion_token_list).to(device)
        init_values = torch.randn(
            len(indices), 
            self.vlm_hidden_size, 
            dtype=self.embed_tokens.weight.dtype,
            device=device,
        ) * 0.02
        self.embed_tokens.weight[indices] = init_values

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
        import glob
        import os

        from safetensors import safe_open

        # load tensors from files
        # Note: pretrained_model_path should be passed from training config
        # For now, we'll need to get it from the parent config or pass it separately
        pretrained_model_path = getattr(self.cfg, 'pretrained_model_path', None)
        if pretrained_model_path is None:
            raise ValueError("pretrained_model_path not found in cfg. Please add it to policy.cfg or pass it separately.")
        
        safetensors_files = glob.glob(
            os.path.join(pretrained_model_path, "*.safetensors")
        )
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # load embed tokens
        embed_tokens_state_dict = self.embed_tokens.state_dict()
        for k, v in tensors.items():
            if "embed_tokens" in k:
                new_key = k.replace("language_model.model.embed_tokens.", "")
                embed_tokens_state_dict[new_key] = v
        self.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)
        log.info("Loaded pre-trained weights for embed tokens")

        # load vision tower --- "vision_tower.vision_model" -> "vision_model"
        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if "vision_tower" in k:
                new_key = k.replace("vision_tower.", "")
                vision_tower_state_dict[new_key] = v
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        log.info("Loaded pre-trained weights for vision tower")

        # load projector --- "multi_modal_projector.linear" -> "linear"
        # Note: The new projector has concatenated RGB (1152) + Depth (768) = 1920 input dims
        # We only load the first 1152 dims (RGB) from pretrained weights, and zero-init the rest (Depth)
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "multi_modal_projector" in k:
                new_key = k.replace("multi_modal_projector.", "")
                if new_key in multi_modal_projector_state_dict:
                    current_param = multi_modal_projector_state_dict[new_key]
                    
                    # Handle weight matrix: shape [input_dim, output_dim]
                    # For regular Linear: weight shape is [out_features, in_features] (transposed)
                    # For LoRA: weight shape is [out_features, in_features], 
                    # lora_A is [r, in_features], lora_B is [out_features, r]
                    if "weight" in new_key and len(current_param.shape) == 2:
                        # Check if current model has larger input dimension (first dim for transposed weight)
                        if current_param.shape[1] > v.shape[1]:
                            # Current model input dim (1920) > pretrained input dim (1152)
                            # Load pretrained weights to the first part, zero-init the rest
                            current_param[:, :v.shape[1]] = v
                            current_param[:, v.shape[1]:] = 0.0
                            multi_modal_projector_state_dict[new_key] = current_param
                            log.info(f"Loaded partial weights for {new_key}: \
                                    first {v.shape[1]} input dims from pretrained, \
                                    remaining {current_param.shape[1] - v.shape[1]} dims zero-initialized.")
                        else:
                            # Dimensions match or current is smaller, load normally
                            multi_modal_projector_state_dict[new_key] = v
                    else:
                        # For bias, lora_B, or other parameters, load normally
                        multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(
            multi_modal_projector_state_dict, strict=True
        )
        log.info("Loaded pre-trained weights for projector (RGB part only, Depth part zero-initialized)")

        # load lm --- do not change any lora weights
        joint_model_state_dict = self.joint_model.state_dict()
        lora_keys = []
        for key in (
            joint_model_state_dict.keys()
        ):  # avoid RuntimeError: OrderedDict mutated during iteration
            if "lora_" in key:
                lora_keys.append(key)
        for key in lora_keys:
            del joint_model_state_dict[key]
        for k, v in tensors.items():
            if "language_model.model" in k:
                new_key = k.replace("language_model.model.", "mixtures.vlm.")
                joint_model_state_dict[new_key] = v
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        log.info("Loaded pre-trained weights for lm part of the joint model")

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
        import os
        import glob

        from safetensors import safe_open

        # load tensors from file
        pretrained_model_path = getattr(self.cfg, 'pretrained_pi05_model_path', None)
        if pretrained_model_path is None:
            raise ValueError(
                "pretrained_pi05_model_path not found in cfg. "
            )
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Pi0.5 model file not found: {pretrained_model_path}")
        log.info(f"Loading Pi0.5 model from: {pretrained_model_path}")
        
        # Load all tensors from the safetensors file
        safetensors_files = glob.glob(
            os.path.join(pretrained_model_path, "*.safetensors")
        )
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
        log.info(f"Loaded {len(tensors)} tensors from Pi0.5 checkpoint")

        # Get target dtype from model (check a parameter to determine current dtype)
        target_dtype = next(self.parameters()).dtype
        source_dtype = next(iter(tensors.values())).dtype
        log.info(f"Target model dtype: {target_dtype}, Pi0.5 checkpoint dtype: {source_dtype}")
        if target_dtype != source_dtype:
            log.info(f"Will convert loaded weights from {source_dtype} to {target_dtype}")

        # Track which parameters are loaded
        loaded_my_model_params = set()
        used_pi05_params = set()

        # load vision tower --- "paligemma_with_expert.paligemma.model.vision_tower.vision_model" -> "vision_model"
        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if "paligemma_with_expert.paligemma.model.vision_tower" in k:
                new_key = k.replace("paligemma_with_expert.paligemma.model.vision_tower.", "")
                if new_key in vision_tower_state_dict:
                    # Convert to target dtype if needed
                    vision_tower_state_dict[new_key] = v.to(dtype=target_dtype)
                    loaded_my_model_params.add(f"vision_tower.{new_key}")
                    used_pi05_params.add(k)
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        log.info("Loaded vision tower weights")

        # load projector --- "paligemma_with_expert.paligemma.model.multi_modal_projector" -> ""
        # Note: The new projector has concatenated RGB (1152) + Depth (768) = 1920 input dims
        # We only load the first 1152 dims (RGB) from pretrained weights, and zero-init the rest (Depth)
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "paligemma_with_expert.paligemma.model.multi_modal_projector" in k:
                new_key = k.replace("paligemma_with_expert.paligemma.model.multi_modal_projector.", "")
                if new_key in multi_modal_projector_state_dict:
                    current_param = multi_modal_projector_state_dict[new_key]
                    
                    # Handle weight matrix: shape [out_features, in_features] (transposed)
                    if "weight" in new_key and len(current_param.shape) == 2:
                        # Check if current model has larger input dimension
                        if current_param.shape[1] > v.shape[1]:
                            # Current model input dim (1920) > pretrained input dim (1152)
                            # Load pretrained weights to the first part, zero-init the rest
                            current_param[:, :v.shape[1]] = v.to(dtype=target_dtype)
                            current_param[:, v.shape[1]:] = 0.0
                            multi_modal_projector_state_dict[new_key] = current_param
                            log.info(f"Loaded partial weights for {new_key}: "
                                    f"first {v.shape[1]} input dims from pretrained, "
                                    f"remaining {current_param.shape[1] - v.shape[1]} dims zero-initialized.")
                        else:
                            # Dimensions match or current is smaller, load normally
                            multi_modal_projector_state_dict[new_key] = v.to(dtype=target_dtype)
                    else:
                        # For bias or other parameters, load normally
                        multi_modal_projector_state_dict[new_key] = v.to(dtype=target_dtype)
                        log.info(f"Loaded weights for {new_key}: {v.shape} -> {current_param.shape}")
                    
                    loaded_my_model_params.add(f"multi_modal_projector.{new_key}")
                    used_pi05_params.add(k)
        self.multi_modal_projector.load_state_dict(
            multi_modal_projector_state_dict, strict=True
        )
        log.info("Loaded multi-modal projector weights (RGB part only, Depth part zero-initialized)")

        # load joint model (both VLM and action mixtures)
        # preserve LoRA weights
        joint_model_state_dict = self.joint_model.state_dict()
        lora_keys = []
        for key in joint_model_state_dict.keys():
            if "lora_" in key:
                lora_keys.append(key)
        # Remove LoRA keys from state dict to avoid overwriting
        for key in lora_keys:
            del joint_model_state_dict[key]
        
        # load VLM mixture --- "paligemma_with_expert.paligemma.model.language_model" -> "mixtures.vlm"
        for k, v in tensors.items():
            if "paligemma_with_expert.paligemma.model.language_model" in k:
                new_key = k.replace("paligemma_with_expert.paligemma.model.language_model.", "mixtures.vlm.")
                if new_key in joint_model_state_dict:
                    # Convert to target dtype if needed
                    joint_model_state_dict[new_key] = v.to(dtype=target_dtype)
                    loaded_my_model_params.add(f"joint_model.{new_key}")
                    used_pi05_params.add(k)
        
        # load action expert mixture --- "paligemma_with_expert.gemma_expert.model" -> "mixtures.action"
        for k, v in tensors.items():
            if "paligemma_with_expert.gemma_expert.model" in k:
                new_key = k.replace("paligemma_with_expert.gemma_expert.model.", "mixtures.action.")
                new_key = new_key.replace("dense", "modulation") # Pi0.5 uses name dense for AdaLN-Zero
                if new_key in joint_model_state_dict:
                    # Convert to target dtype if needed
                    joint_model_state_dict[new_key] = v.to(dtype=target_dtype)
                    loaded_my_model_params.add(f"joint_model.{new_key}")
                    used_pi05_params.add(k)
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        log.info("Loaded joint model weights (VLM mixture + action mixture)")

        # load lm_head if present in our model
        if self.use_lm_head and hasattr(self, 'lm_head'):
            lm_head_state_dict = self.lm_head.state_dict()
            for k, v in tensors.items():
                if k == "paligemma_with_expert.paligemma.lm_head.weight":
                    # Note: In PaliGemma, lm_head.weight is tied with embed_tokens.weight
                    # We also tie them in our model, so loading lm_head will also update embed_tokens
                    # pi05 vocab only contains all the useful tokens, so we need to slice the weights
                    lm_head_state_dict["weight"][:v.shape[0]] = v.to(dtype=target_dtype)
                    loaded_my_model_params.add("lm_head.weight")
                    loaded_my_model_params.add("embed_tokens.weight")  # tied weights
                    used_pi05_params.add(k)
            self.lm_head.load_state_dict(lm_head_state_dict, strict=True)
            log.info("Loaded lm_head weights (tied with embed_tokens)")
        else:
            log.warning("lm_head not found or use_lm_head=False, skipping lm_head weight loading")

        # load time embedding MLPs --- "time_mlp_in/out" -> "time_embedding"
        # Pi0.5 uses separate time_mlp_in and time_mlp_out
        # Map to our TimeEncoder: time_embedding = nn.Sequential(SinusoidalPosEmb, TimeEncoder)
        # TimeEncoder has linear_1 and linear_2
        time_embedding_state_dict = self.time_embedding.state_dict()
        for k, v in tensors.items():
            if k.startswith("time_mlp_in."):
                # Map time_mlp_in to TimeEncoder's linear_1
                # time_mlp_in.weight/bias -> time_embedding.1.linear_1.weight/bias
                param_name = k.replace("time_mlp_in.", "1.linear_1.")
                # Convert to target dtype if needed
                time_embedding_state_dict[param_name] = v.to(dtype=target_dtype)
                loaded_my_model_params.add(f"time_embedding.{param_name}")
                used_pi05_params.add(k)
            elif k.startswith("time_mlp_out."):
                # Map time_mlp_out to TimeEncoder's linear_2
                # time_mlp_out.weight/bias -> time_embedding.1.linear_2.weight/bias
                param_name = k.replace("time_mlp_out.", "1.linear_2.")
                # Convert to target dtype if needed
                time_embedding_state_dict[param_name] = v.to(dtype=target_dtype)
                loaded_my_model_params.add(f"time_embedding.{param_name}")
                used_pi05_params.add(k)
        self.time_embedding.load_state_dict(time_embedding_state_dict, strict=True)
        log.info("Loaded time embedding weights (TimeEncoder)")

    def freeze_non_lora_weights_in_vlm(self):
        """
        Freeze non-LoRA weights in VLM components while keeping LoRA weights trainable.
        
        This method freezes:
        - Vision tower weights (except LoRA)
        - Multi-modal projector weights (except LoRA)  
        - Language model weights (except LoRA)
        
        Only LoRA parameters remain trainable for efficient fine-tuning.
        """
        for name, param in self.vision_tower.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in vision tower")

        for name, param in self.multi_modal_projector.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in projector")

        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in lm part of the joint model")

    def freeze_non_lora_weights_in_ae(self):
        """
        Freeze non-LoRA weights in VLM components while keeping LoRA weights trainable.
        
        This method freezes:
        - Action encoder weights (except LoRA)
        - Action decoder weights (except LoRA)  
        - Action mixture weights (except LoRA)
        
        Only LoRA parameters remain trainable for efficient fine-tuning.
        """
        for name, param in self.action_encoder.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in action encoder")

        for name, param in self.action_decoder.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in action decoder")

        for name, param in self.joint_model.mixtures["action"].named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in action mixture")

    def freeze_all_weights(self):
        """
        Freeze all trainable parameters in the model.
        
        Sets requires_grad=False for all parameters, making the model non-trainable.
        Useful for inference-only scenarios.
        """
        for _, param in self.named_parameters():
            param.requires_grad = False

    def build_text_cache(self):
        """
        Create a new KV cache for text generation.
        
        Returns:
            KVCache: Empty key-value cache for storing attention states during text generation
        """
        return KVCache()

    # ---------- Input preparation ---------- #

    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, answer_start_idx: torch.Tensor, n_actions: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        """
        Build causal attention masks and position IDs for different token types.
        
        Creates block-diagonal attention patterns:
        - Image/text tokens can attend to themselves
        - Answer only tokens can attend to image/text and the answer tokens before them
        - Action tokens can attend to image/text, and themselves (causal)
        
        Args:
            attention_mask (torch.Tensor): [B, seq_len] Attention mask indicating valid tokens
            answer_start_idx (torch.Tensor): [B] Index of the first answer token
            n_actions (torch.Tensor): [B] Number of action tokens for each sample
            dtype (torch.dtype): Data type for the causal mask
        
        Returns:
            Tuple containing:
                - causal_mask (torch.FloatTensor): [B, 1, total_len, total_len] 
                  Causal attention mask with block structure (broadcasts to all heads)
                - vlm_position_ids (torch.LongTensor): [B, seq_len] Position IDs for VLM tokens
                - action_position_ids (torch.LongTensor): [B, num_actions] Position IDs for action tokens
                
        block attention --- padding for unused text tokens

                       img/text img/text answer answer answer (padding) action action
        img/text          x        x
        img/text          x        x
        answer            x        x       x           
        answer            x        x       x      x   
        answer            x        x       x      x      x
        (padding)
        action            x        x                                       x      x
        action            x        x                                       x      x
        """
        bsz = attention_mask.size(0)
        device = attention_mask.device
        max_vlm_tokens = attention_mask.shape[-1]
        total_num_tokens = max_vlm_tokens + self.num_action_tokens
        action_start = max_vlm_tokens
        vlm_token_cnts = torch.sum(attention_mask, dim=1)
        causal_mask = torch.full(
            (bsz, total_num_tokens, total_num_tokens),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )  # smallest value, avoid using inf for softmax nan issues with padding
        for idx in range(bsz):
            cnt = vlm_token_cnts[idx].item()
            start = answer_start_idx[idx].item()
            answer_len = cnt - start
            n_action = n_actions[idx].item()
            causal_mask[idx, :cnt, :start] = 0  # image/text/answer attend to image/text
            mask = torch.tril(torch.ones((answer_len, answer_len), dtype=torch.bool, device=device))
            causal_mask[idx, start:cnt, start:cnt] = torch.where(
                mask, 0, torch.finfo(dtype).min
            ) # answer tokens attend to answer tokens before them
            causal_mask[idx, action_start:action_start+n_action, :start] = (
                0  # action attend to image/text
            )
            causal_mask[idx, action_start:action_start+n_action, action_start:action_start+n_action] = (
                0  # action attend to itself
            )

        # add the head dimension for broadcasting to all attention heads
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, 1, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks --- start at 1
        vlm_position_ids = torch.arange(1, max_vlm_tokens + 1, device=device).repeat(
            bsz, 1
        )
        # action position ids start from answer_start_idx for each sample
        action_position_ids = torch.arange(
            0,
            self.num_action_tokens,
            device=device,
        ).unsqueeze(0) + answer_start_idx.unsqueeze(1) + 1
        return causal_mask, vlm_position_ids, action_position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor, max_vlm_tokens: int
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Split the full causal mask into separate masks for different model components.
        
        Args:
            causal_mask (torch.FloatTensor): [B, 1, total_len, total_len] 
              Full causal attention mask (broadcasts to all heads)
        
        Returns:
            Tuple containing:
                - vlm_mask (torch.FloatTensor): [B, 1, seq_len, seq_len]
                  Attention mask for image/text tokens (broadcasts to all heads)
                - action_mask (torch.FloatTensor): [B, 1, action_len, total_len]
                  Attention mask for action tokens (broadcasts to all heads)
        """
        vlm_mask = causal_mask[..., : max_vlm_tokens, : max_vlm_tokens] 
        action_mask = causal_mask[..., -self.num_action_tokens :, :]
        return vlm_mask, action_mask

    def build_causal_mask_and_position_ids_for_text(
        self,
        q_len: int,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Build causal mask and position IDs for autoregressive generation.
        
        Creates attention masks for autoregressive generation with optional KV cache.
        - Prefill phase: No masking (all tokens can attend to each other)
        - Generation phase: No masking (query can attend to all cached tokens)
        
        Args:
            q_len (int): Length of the current query sequence
            attention_mask (torch.Tensor): [B, seq_len] Attention mask for input tokens (left padding)
            kv_cache (Optional[KVCache]): Optional KV cache for generation
        
        Returns:
            Tuple containing:
                - causal_mask (torch.FloatTensor): [B, 1, q_len, kv_len] Attention mask (broadcasts to all heads)
                - position_ids (torch.LongTensor): [B, q_len] Position IDs for query tokens
        """
        device = attention_mask.device
        bsz = attention_mask.size(0)

        if kv_cache is None or kv_cache.num_items() == 0:
            # Assert left padding: once we see a valid token (1), all subsequent tokens must be valid
            # Check that there are no padding tokens after the first valid token
            has_padding = (attention_mask == 0).any(dim=-1)  # [B], True if batch has padding
            if has_padding.any():
                # For batches with padding, check left padding property
                for b in range(bsz):
                    if has_padding[b]:
                        mask = attention_mask[b]  # [seq_len]
                        # Find first valid token
                        first_valid_idx = (mask != 0).nonzero(as_tuple=True)[0]
                        assert len(first_valid_idx) > 0, "Expect left padding: no valid tokens found"
                        first_valid_idx = first_valid_idx[0].item()
                        # All tokens before first_valid_idx should be padding (0)
                        assert (mask[:first_valid_idx] == 0).all(), \
                            f"Expect left padding: found valid tokens before first valid token at position {first_valid_idx}"
                        # All tokens from first_valid_idx onwards should be valid (non-zero)
                        assert (mask[first_valid_idx:] != 0).all(), \
                            f"Expect left padding: found padding tokens after first valid token at position {first_valid_idx}"
            # Prefill phase: create bidirectional mask
            # During inference, we use left padding by default
            # Initialize all positions to minimum value (masked out by default)
            causal_mask = torch.full(
                (bsz, q_len, q_len),
                torch.finfo(dtype).min,
                dtype=dtype,
                device=device,
            )  # Use smallest value to avoid softmax nan issues with padding
            
            # For left padding: attention_mask[i] == 0 means position i is a padding token
            # Unmask valid positions (where both query and key are not padding)
            assert attention_mask.size(-1) == q_len, "Attention mask must have the same length as the total sequence"
            valid_mask = (attention_mask != 0)  # True for valid (non-padding) positions
            causal_mask = causal_mask.masked_fill(valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(1), 0)
        else:
            # Generation phase: using KV cache for incremental decoding
            assert q_len == 1, "Using KV cache so should only use one single token"
            kv_len = kv_cache.num_items() + q_len
            
            # During inference with left padding, the KV cache contains padding tokens at the beginning
            # Initialize all positions to minimum value (masked out by default)
            causal_mask = torch.full(
                (bsz, q_len, kv_len),
                torch.finfo(dtype).min,
                dtype=dtype,
                device=device,
            )  # Use smallest value to avoid softmax nan issues with padding
            
            assert attention_mask.size(-1) == kv_len, "Attention mask must have the same length as the total sequence"
            valid_mask = (attention_mask != 0)  # True for valid (non-padding) positions
            causal_mask = causal_mask.masked_fill(valid_mask.unsqueeze(1), 0)  # [B, 1, kv_len]

        # add the head dimension for broadcasting to all attention heads
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, 1, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # use the last location
            position_ids = attention_mask.cumsum(-1)[:, -1:]
        else:
            # create position_ids based on the size of the attention_mask
            # for padded tokens, use number 1
            position_ids = (attention_mask.cumsum(-1)).masked_fill_(
                (attention_mask == 0), 1
            )
        return causal_mask, position_ids

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
            action_features = self.action_encoder_ar(actions) / (self.vlm_hidden_size**0.5)
        if pixel_values is not None:
            image_mask = input_ids == self.image_token_index
            # autocast does not cast nn.Embedding to the correct dtype, we need to cast manually
            
        for i in range(bsz):
            if pixel_values is not None: 
                image_indices = image_mask[i].nonzero(as_tuple=True)[0]
                if has_depth_values is not None and has_depth_values[i] and \
                    not (self.training and random.random() < self.depth_dropout):
                    # Each RGB token is paired with corresponding depth token
                    depth_image_feature = depth_image_features[i]
                else: 
                    if T is not None:
                        depth_image_feature = self.depth_missing_embeddings.repeat(T, 1)
                    else:
                        depth_image_feature = self.depth_missing_embeddings
                paired_image_features = torch.cat([
                    rgb_image_features[i], depth_image_feature
                ], dim=-1) # [num_patches, rgb_embed_dim+depth_embed_dim] 
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
    def infer_action(
        self,
        input: dict,
        return_attn_weights: bool = False
    ) -> torch.FloatTensor:
        """
        Inference function for action generation using flow matching.
        
        Args:
            input (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - pixel_values (torch.FloatTensor): [B, 3, H, W] or [B, T, 3, H, W] Image pixel values (normalized)
                - vlm_mask (torch.FloatTensor): [B, 1, seq_len, seq_len] 
                  Attention mask for image/text/proprio tokens (broadcasts to all heads)
                - action_mask (torch.FloatTensor): [B, 1, action_len, total_len] 
                  Attention mask for action tokens (broadcasts to all heads)
                - vlm_position_ids (torch.LongTensor): [B, seq_len] Position IDs for VLM tokens
                - action_position_ids (torch.LongTensor): [B, num_actions] Position IDs for action tokens
            return_attn_weights (bool): Whether to return attention weights
        
        Returns:
            torch.FloatTensor: [B, horizon_steps, action_dim] Generated action sequence
        """
        # Extract inputs from dict
        input_ids = input["input_ids"]
        pixel_values = input["pixel_values"]
        vlm_mask = input["vlm_mask"]
        action_mask = input["action_mask"]
        vlm_position_ids = input["vlm_position_ids"]
        action_position_ids = input["action_position_ids"]

        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        if 'depth_values' in input:
            depth_values = input["depth_values"]
            has_depth_values = input["has_depth_values"]
        else:
            depth_values = None
            has_depth_values = None
        inputs_embeds = self._forward_siglip_and_text_embedding(
            input_ids=input_ids,
            pixel_values=pixel_values,
            depth_values=depth_values,
            has_depth_values=has_depth_values,
            states=input["states"],
            n_states=input["n_states"],
            is_vla_data=input["is_vla_data"],
            dtype=pixel_values.dtype
        )
        
        # forward pass thru the vlm, cache the kv
        _, kv_caches = self.joint_model(
            attention_mask=vlm_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )
        # [num_layers, B, num_heads, seq_len, seq_len]
        vlm_attn_weights = torch.stack(self.attn_weights, dim=0).detach().clone()
        action_expert_attn_weights = None

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of vlm
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for step_idx in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            action_embeds = action_embeds / (self.action_hidden_size**0.5)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm
            )["action"]
            if step_idx == 0:
                action_expert_attn_weights = torch.stack(self.attn_weights, dim=0).detach().clone()
            
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        if return_attn_weights:
            return action, vlm_attn_weights, action_expert_attn_weights
        return action

    @torch.inference_mode()
    def infer_single_step(
        self,
        input: dict,
        kv_cache: Optional[KVCache] = None,
        dtype: torch.dtype = torch.float32,
        return_attn_weights: bool = False,
    ) -> dict:
        """
        Inference function for discrete action generation.
        
        Args:
            input (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - pixel_values (torch.FloatTensor, Optional): [B, 3, H, W] or [B, T, 3, H, W] Image pixel values (normalized)
                  If None, only text tokens are processed
                - attention_mask (torch.LongTensor): [B, seq_len] Attention mask
                - states (torch.FloatTensor, Optional): [B, state_dim] State features
                - actions (torch.FloatTensor, Optional): [B, action_dim] Action features
                - n_states (torch.LongTensor, Optional): [B] Number of state features
                - n_actions (torch.LongTensor, Optional): [B] Number of action features
            kv_cache (Optional[KVCache]): Key-value cache for the generated tokens
            dtype (torch.dtype): Data type for the input and output
            return_attn_weights (bool): Whether to return attention weights

        Returns:
            dict: Dictionary containing:
                - logits (torch.FloatTensor): [B, seq_len, vocab_size] Logits for the generated tokens
                - kv_cache (KVCache): Key-value cache for the generated tokens
                - attn_weights (torch.FloatTensor, optional): [num_layers, B, num_heads, q_len, k_len]
        """
        input_ids = input["input_ids"]
        attention_mask = input["attention_mask"]
        q_len = input_ids.size(1)

        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(
            input_ids=input_ids,
            pixel_values=input.get("pixel_values"),
            depth_values=input.get("depth_values"),
            has_depth_values=input.get("has_depth_values"),
            states=input.get("states"),
            actions=input.get("actions"),
            n_states=input.get("n_states"),
            n_actions=input.get("n_actions"),
            is_vla_data=input.get("is_vla_data"),
            dtype=dtype
        )

        # build causal mask and position ids for text
        (
            causal_mask,
            position_ids,
        ) = self.build_causal_mask_and_position_ids_for_text(
            q_len, attention_mask, kv_cache, dtype
        )

        hidden_states = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={"vlm": position_ids},
            embeds_all={"vlm": inputs_embeds},
            kv_caches={"vlm": kv_cache},
            cache_mode="append",  # new tokens for the active mixture
            final_layer_post_attn_skip_names=[],  # do not skip vlm last layer
        )["vlm"]
        output = {
            "hidden_states": hidden_states,
        }
        if return_attn_weights:
            output["attn_weights"] = torch.stack(self.attn_weights, dim=0).detach().clone()
        if kv_cache is not None:
            output["kv_cache"] = kv_cache
        return output

    @torch.inference_mode()
    def infer_vlm(
        self,
        input: dict,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 10,
        top_p: float = 1.0,
        allowed_token_ids: Optional[Union[torch.LongTensor, List[int], Tuple[int, int]]] = None,
        eos_token_id: Optional[int] = None,
        return_kv_cache: bool = False,
        return_attn_weights: bool = False,
    ) -> dict:
        """
        Multi-step autoregressive generation function for VLM, divided into prefill and incremental prediction phases.
        
        Args:
            input (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - pixel_values (torch.FloatTensor): [B, 3, H, W] or [B, T, 3, H, W] Image pixel values (normalized)
                - depth_values (torch.FloatTensor, Optional): [Bd, T, 3, H, W] Depth image values (normalized)
                - has_depth_values (torch.LongTensor, Optional): [B] Depth image IDs
                - attention_mask (torch.LongTensor): [B, seq_len] Attention mask (optional)
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Temperature parameter controlling sampling randomness
            top_k (int): Top-k sampling parameter
            top_p (float): Nucleus sampling parameter
            allowed_token_ids (Optional[Union[torch.LongTensor, List[int], Tuple[int, int]]]): 
                Allowed token ID range for sampling. Can be:
                - Boolean mask tensor [vocab_size]
                - List of token IDs
                - Tuple (min_id, max_id) representing a range
            eos_token_id (Optional[int]): End-of-sequence token ID, early stopping if this token is generated
            return_kv_cache (bool): Whether to return KV cache
            return_attn_weights (bool): Whether to return attention weights
        
        Returns:
            dict: Dictionary containing:
                - generated_ids (torch.LongTensor): [B, prefill_len + num_generated] Generated token IDs
                - kv_cache (KVCache, optional): KV cache (if return_kv_cache=True)
                - attn_weights (torch.FloatTensor, optional): [num_layers, B, num_heads, total_q_len, total_k_len]
        """
        input_ids = input["input_ids"]
        pixel_values = input["pixel_values"]
        attention_mask = input.get("attention_mask")
        
        batch_size = input_ids.size(0)
        device, dtype = input_ids.device, pixel_values.dtype
        
        # Create all-ones mask if attention_mask is not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        
        # Initialize KV cache
        kv_cache = KVCache()
        
        # ========== Prefill Phase ==========
        # Process initial input sequence and build KV cache
        # Pass empty kv_cache to let the model build the cache
        prefill_input = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "depth_values": input.get("depth_values"),
            "has_depth_values": input.get("has_depth_values"),
            "attention_mask": attention_mask,
        }
        
        attn_weights_steps = [] if return_attn_weights else None
        prefill_output = self.infer_single_step(
            prefill_input,
            kv_cache=kv_cache,
            dtype=dtype,
            return_attn_weights=return_attn_weights,
        )
        prefill_hidden_states = prefill_output["hidden_states"]
        prefill_logits = self.lm_head(prefill_hidden_states)
        prefill_logits = self._apply_final_logit_softcapping(prefill_logits) # [B, seq_len, vocab_size]
        kv_cache = prefill_output.get("kv_cache", kv_cache)
        if return_attn_weights:
            attn_weights_steps.append(prefill_output.get("attn_weights"))
        
        # Sample first new token from the last position of prefill
        next_token_logits = prefill_logits[:, -1, :]  # [B, vocab_size]
        next_token_ids = sample_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            allowed_token_ids=allowed_token_ids,
        )  # [B]
        
        # Collect generated token IDs
        generated_ids = [input_ids.clone()]  # Save original input first
        generated_ids.append(next_token_ids.unsqueeze(1))  # [B, 1]
        
        # Track finish status for each batch
        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            finished_mask = (next_token_ids == eos_token_id)
        
        # ========== Generation Phase ==========
        # Incrementally generate new tokens
        for _ in range(max_new_tokens - 1):
            # Stop if all batches are finished
            if finished_mask.all():
                break
            
            # For finished batches, set next_token_ids to pad_token_id
            # For unfinished batches, perform inference
            active_mask = ~finished_mask
            
            # Prepare input_ids: use pad_token_id for finished batches, actual tokens for active batches
            step_input_ids = next_token_ids.clone()
            step_input_ids[finished_mask] = self.pad_token_id
            
            # Single-step inference (using KV cache)
            # In generation phase, only need to pass the newly generated token
            # Image tokens have already been processed in prefill phase
            attention_mask = torch.cat([
                attention_mask, torch.ones(
                    (batch_size, 1), dtype=torch.long, device=device
                ) * active_mask.unsqueeze(1),
            ], dim=-1)
            step_input = {
                "input_ids": step_input_ids.unsqueeze(1),  # [B, 1]
                "attention_mask": attention_mask,  # [B, seq_len]
            }
            
            step_output = self.infer_single_step(
                step_input,
                kv_cache=kv_cache,
                dtype=dtype,
                return_attn_weights=return_attn_weights,
            )
            step_hidden_states = step_output["hidden_states"]
            step_logits = self.lm_head(step_hidden_states)
            step_logits = self._apply_final_logit_softcapping(step_logits) # [B, 1, vocab_size]
            kv_cache = step_output.get("kv_cache", kv_cache)
            if return_attn_weights:
                attn_weights_steps.append(step_output.get("attn_weights"))
            
            # Initialize next_token_ids with pad_token_id for all batches
            next_token_ids = torch.full(
                (batch_size,), self.pad_token_id, dtype=torch.long, device=device
            )
            
            # Sample next token only for active batches
            next_token_logits = step_logits[:, -1, :]  # [B, vocab_size]
            next_token_ids_active = sample_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                allowed_token_ids=allowed_token_ids,
            )  # [B]
            # Update active batches with sampled tokens
            next_token_ids[active_mask] = next_token_ids_active[active_mask]
            
            # Update finish status
            if eos_token_id is not None:
                finished_mask = finished_mask | (next_token_ids == eos_token_id)
            
            # Save generated token
            generated_ids.append(next_token_ids.unsqueeze(1))  # [B, 1]
        
        # Concatenate all generated token IDs
        generated_ids_tensor = torch.cat(generated_ids, dim=1)  # [B, prefill_len + num_generated]
        
        result = {
            "generated_ids": generated_ids_tensor,
        }
        
        if return_kv_cache:
            result["kv_cache"] = kv_cache
        if return_attn_weights:
            result["attn_weights"] = concat_attn_weights(attn_weights_steps)
        
        return result

    @torch.inference_mode()
    def infer_vla(
        self,
        input: dict,
        max_new_tokens: int,
        temperature: float = 1.0,
        return_attn_weights: bool = False,
        cfg: float = 1.0,
        **kwargs,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Autoregressive action inference for VLA using DiffLoss sampling.
        
        Args:
            input (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - pixel_values (torch.FloatTensor): [B, 3, H, W] or [B, T, 3, H, W] Image pixel values (normalized)
                - attention_mask (torch.LongTensor): [B, seq_len] Attention mask (optional)
                - states (torch.FloatTensor, optional): [B, state_len, state_dim]
                - n_states (torch.LongTensor, optional): [B]
                - is_vla_data (torch.BoolTensor, optional): [B]
            max_new_tokens (int): Number of action tokens to generate
            temperature (float): Sampling temperature for diffusion
            return_attn_weights (bool): Whether to return attention weights
            cfg (float): Classifier-free guidance scale for diffusion sampling
        
        Returns:
            torch.FloatTensor: [B, max_new_tokens, action_dim] Generated action sequence
            (optional) torch.FloatTensor: Attention weights if return_attn_weights=True
        """
        input_ids = input["input_ids"]
        pixel_values = input.get("pixel_values")
        attention_mask = input.get("attention_mask")
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        dtype = pixel_values.dtype if pixel_values is not None else torch.float32
        
        is_vla_data = input.get("is_vla_data")
        if is_vla_data is None:
            is_vla_data = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Initialize KV cache
        kv_cache = KVCache()
        
        # ========== Prefill Phase ==========
        prefill_input = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "depth_values": input.get("depth_values"),
            "has_depth_values": input.get("has_depth_values"),
            "attention_mask": attention_mask,
            "states": input.get("states"),
            "n_states": input.get("n_states"),
            "is_vla_data": is_vla_data,
        }
        
        attn_weights_steps = [] if return_attn_weights else None
        prefill_output = self.infer_single_step(
            prefill_input,
            kv_cache=kv_cache,
            dtype=dtype,
            return_attn_weights=return_attn_weights,
        )
        prefill_hidden_states = prefill_output["hidden_states"]
        kv_cache = prefill_output.get("kv_cache", kv_cache)
        if return_attn_weights:
            attn_weights_steps.append(prefill_output.get("attn_weights"))
        
        # Sample the first action from the last token
        next_condition_states = prefill_hidden_states[:, -1, :]
        latent_condition = self.latent_condition_projector(next_condition_states)
        next_action = self.diffloss.sample(latent_condition, temperature=temperature, cfg=cfg)
        generated_actions = [next_action.unsqueeze(1)]
        
        # ========== Generation Phase ==========
        for _ in range(max_new_tokens - 1):
            # Append one action token to the attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)],
                dim=-1,
            )
            step_input = {
                "input_ids": torch.full(
                    (batch_size, 1),
                    self.action_token_index,
                    dtype=input_ids.dtype,
                    device=device,
                ),
                "attention_mask": attention_mask,
                "actions": next_action.unsqueeze(1),
                "n_actions": torch.ones(batch_size, dtype=torch.long, device=device),
                "is_vla_data": is_vla_data,
            }
            
            step_output = self.infer_single_step(
                step_input,
                kv_cache=kv_cache,
                dtype=dtype,
                return_attn_weights=return_attn_weights,
            )
            step_hidden_states = step_output["hidden_states"]
            kv_cache = step_output.get("kv_cache", kv_cache)
            if return_attn_weights:
                attn_weights_steps.append(step_output.get("attn_weights"))
            
            latent_condition = self.latent_condition_projector(step_hidden_states[:, -1, :])
            next_action = self.diffloss.sample(latent_condition, temperature=temperature, cfg=cfg)
            generated_actions.append(next_action.unsqueeze(1))
        
        generated_actions_tensor = torch.cat(generated_actions, dim=1)
        if return_attn_weights:
            return generated_actions_tensor, concat_attn_weights(attn_weights_steps)
        return generated_actions_tensor

    # ---------- Flow matching training ----------#
    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Conditional flow function for flow matching.
        
        Interpolates between noise x and target x1 based on time t.
        
        Args:
            x (torch.FloatTensor): [B, horizon_steps, action_dim] Initial noise
            x1 (torch.FloatTensor): [B, horizon_steps, action_dim] Target action
            t (torch.FloatTensor): [B, 1, 1] Time parameter (0 to 1)
        
        Returns:
            torch.FloatTensor: [B, horizon_steps, action_dim] Interpolated action at time t
        """
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    # TODO: Deprecated method, to be updated
    def compute_ar_loss(
        self,
        batch: dict,
    ) -> torch.FloatTensor:
        """
        Compute autoregressive loss for action prediction and vision language understanding.
        
        This method computes action prediction loss and vision language understanding loss
        
        Args:
            batch (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - labels (torch.LongTensor, optional): [B, seq_len] Labels for language modeling loss
                - pixel_values (torch.FloatTensor): [B, 3, H, W] or [B, T, 3, H, W] Image pixel values (normalized)
                - causal_mask (torch.FloatTensor): [B, 1, total_len, total_len] Full causal attention mask
                - vlm_position_ids (torch.LongTensor): [B, seq_len] Position IDs for VLM tokens
        
        Returns:
            dict: Dictionary containing:
                - ce_loss (torch.FloatTensor): Cross-entropy loss for Autoregressive VLA
        """
        # Extract inputs from batch dict
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        pixel_values = batch["pixel_values"]
        causal_mask = batch["causal_mask"]
        vlm_position_ids = batch["vlm_position_ids"]

        # text tokens + image tokens
        if 'depth_values' in batch:
            depth_values = batch["depth_values"]
            has_depth_values = batch["has_depth_values"]
        else:
            depth_values = None
            has_depth_values = None
        inputs_embeds = self._forward_siglip_and_text_embedding(
            input_ids=input_ids,
            pixel_values=pixel_values,
            depth_values=depth_values,
            has_depth_values=has_depth_values,
            states=batch["states"],
            actions=batch["actions"],
            n_states=batch["n_states"],
            n_actions=batch["n_actions"],
            is_vla_data=batch["is_vla_data"],
            dtype=pixel_values.dtype
        )
        
        output = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
            },
            kv_caches={},  # no caching during training
            final_layer_post_attn_skip_names=[],  # do not skip vlm last layer
        )
        hidden_states = output["vlm"]

        logits = self.lm_head(hidden_states)
        logits = self._apply_final_logit_softcapping(logits)
        logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        labels = labels[:, 1:].contiguous().view(-1)

        ce_loss = self.CELoss(logits, labels)
        valid_num_labels = torch.sum(labels != self.ignore_index)
        ce_loss = ce_loss / valid_num_labels.clamp(min=1)

        return {
            "ce_loss": ce_loss,
        }

    # TODO: Deprecated method, to be updated
    def compute_flow_loss(
        self,
        batch: dict,
    ) -> torch.FloatTensor:
        """
        Forward pass for flow matching training.
        
        Args:
            batch (dict): Training batch dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - pixel_values (torch.ByteTensor): [B, 3, H, W] or [B, T, 3, H, W] Image pixel values (uint8)
                - causal_mask (torch.FloatTensor): [B, 1, total_len, total_len] 
                  Full causal attention mask for all tokens (broadcasts to all heads)
                - vlm_position_ids (torch.LongTensor): [B, seq_len] Position IDs for VLM tokens
                - action_position_ids (torch.LongTensor): [B, num_actions] Position IDs for action tokens
                - actions (torch.FloatTensor): [B, horizon_steps, action_dim] Ground truth action
                - actions_valid_mask (torch.BoolTensor): [B, horizon_steps, action_dim] Valid mask for action
                - t (torch.FloatTensor): [B] Time steps for flow matching (0 to 1)

        Note: 
            Action structure mirrors the state
        
        Returns:
            dict: Dictionary containing:
                - flow_loss (torch.FloatTensor): Flow matching loss (mean squared error)
        """
        # Extract inputs from batch dict
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]
        causal_mask = batch["causal_mask"]
        vlm_position_ids = batch["vlm_position_ids"]
        action_position_ids = batch["action_position_ids"]
        actions = batch["actions"]
        actions_valid_mask = batch["actions_valid_mask"]
        t = batch["t"]

        """flow matching loss for action prediction, no use of kv cache"""
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        # text tokens + image tokens
        if 'depth_values' in batch:
            depth_values = batch["depth_values"]
            has_depth_values = batch["has_depth_values"]
        else:
            depth_values = None
            has_depth_values = None
        inputs_embeds = self._forward_siglip_and_text_embedding(
            input_ids=input_ids,
            pixel_values=pixel_values,
            depth_values=depth_values,
            has_depth_values=has_depth_values,
            states=batch["states"],
            actions=batch["actions"],
            n_states=batch["n_states"],
            n_actions=batch["n_actions"],
            is_vla_data=batch["is_vla_data"],
            dtype=pixel_values.dtype
        )

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)
        action_embeds = action_embeds / (self.action_hidden_size**0.5)
        action_embeds = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "action": action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "action": action_embeds,
            },
            time_cond=time_cond,
            kv_caches={},  # no caching during training
        )["action"]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0

        loss = (v_psi - d_psi) ** 2
        # Use element-wise multiplication to make sure the gradient can always be propagated to the action expert 
        masked_loss = actions_valid_mask * loss
        actions_valid_num = torch.sum(actions_valid_mask)
        flow_loss = torch.sum(masked_loss) / actions_valid_num.clamp(min=1)
        return {
            "flow_loss": flow_loss,
        }

    @torch.compile
    def compute_celoss(self, hidden_states: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """
        Compute cross-entropy loss for language modeling.

        Args:
            hidden_states (torch.FloatTensor): [B, seq_len, hidden_size] Hidden states from the language model
            labels (torch.LongTensor): [B, seq_len] Labels for language modeling loss

        Returns:
            torch.FloatTensor: Cross-entropy loss
        """
        logits = self.lm_head(hidden_states)
        logits = self._apply_final_logit_softcapping(logits)
        logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        labels = labels[:, 1:].contiguous().view(-1)

        ce_loss = self.CELoss(logits, labels)
        valid_num_labels = torch.sum(labels != self.ignore_index)
        ce_loss = ce_loss / valid_num_labels.clamp(min=1)
        return ce_loss

    def compute_loss(self, batch: dict) -> dict:
        """
        Compute combined VLA loss with two components: cross-entropy loss (VLM) and flow matching loss (whole VLA).
        
        This method computes both vision-language understanding loss and action prediction loss
        for comprehensive VLA training.
        
        Args:
            batch (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - labels (torch.LongTensor, optional): [B, seq_len] Labels for language modeling loss
                - pixel_values (torch.FloatTensor): [B, 3, H, W] or [B, T, 3, H, W] Image pixel values (normalized)
                - causal_mask (torch.FloatTensor): [B, 1, total_len, total_len] Full causal attention mask
                - vlm_position_ids (torch.LongTensor): [B, seq_len] Position IDs for VLM tokens
                - action_position_ids (torch.LongTensor): [B, num_actions] Position IDs for action tokens
                - actions (torch.FloatTensor): [B, horizon_steps, action_dim] Ground truth action
                - actions_valid_mask (torch.BoolTensor): [B, horizon_steps, action_dim] Valid mask for action
                - t (torch.FloatTensor): [B] Time steps for flow matching (0 to 1)
        
        Returns:
            dict: Dictionary containing:
                - total_loss (torch.FloatTensor): Combined loss
                - ce_loss (torch.FloatTensor): Cross-entropy loss for Autoregressive VLA
                - flow_loss (torch.FloatTensor): Flow matching loss for action prediction
        """
        # Extract inputs from batch dict
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        pixel_values = batch["pixel_values"]
        causal_mask = batch["causal_mask"]
        vlm_position_ids = batch["vlm_position_ids"]
        action_position_ids = batch["action_position_ids"]
        actions = batch["actions"]  # (B, horizon_steps, action_dim)
        actions_valid_mask = batch["actions_valid_mask"]
        t = batch["t"]
        states = batch["states"]
        answer_start_idx = batch["answer_start_idx"]
        is_vla_data = batch["is_vla_data"]
        is_vlm_data = (is_vla_data != True)
        n_actions = batch["n_actions"]
        n_states = batch["n_states"]

        """flow matching loss for action prediction, no use of kv cache"""
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        # text tokens + image tokens
        if 'depth_values' in batch:
            depth_values = batch["depth_values"]
            has_depth_values = batch["has_depth_values"]
        else:
            depth_values = None
            has_depth_values = None
        inputs_embeds = self._forward_siglip_and_text_embedding(
            input_ids=input_ids,
            pixel_values=pixel_values,
            depth_values=depth_values,
            has_depth_values=has_depth_values,
            states=states,
            actions=actions,
            n_states=n_states,
            n_actions=n_actions,
            is_vla_data=is_vla_data,
            dtype=pixel_values.dtype
        )
        
        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)
        action_embeds = action_embeds / (self.action_hidden_size**0.5)
        output = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "action": action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "action": action_embeds,
            },
            time_cond=time_cond,
            kv_caches={},  # no caching during training
            final_layer_post_attn_skip_names=[],  # do not skip vlm last layer
        )
        hidden_states = output["vlm"]
        action_embeds = output["action"]

        if torch.any(is_vlm_data): 
            ce_loss = self.compute_celoss(hidden_states[is_vlm_data], labels[is_vlm_data])
        else:
            ce_loss = self.compute_celoss(hidden_states, labels)

        # diffusion loss
        device = hidden_states.device
        vla_hidden = hidden_states[is_vla_data]
        vla_action = actions[is_vla_data]

        if torch.any(is_vla_data):
            # 1. 获取维度的基本信息
            max_vlm_tokens = hidden_states.shape[1]
            num_action_tokens = self.num_action_tokens

            # 2. 构建索引序列 (0, 1, 2, ..., max_len-1)
            # range_hidden: (1, seq_len)
            range_hidden = torch.arange(max_vlm_tokens, device=device).unsqueeze(0)
            # range_action: (1, action_seq_len) 
            # 注意：如果 vla_action 的长度和 vla_hidden 不一致，需要单独生成 range
            range_action = torch.arange(num_action_tokens, device=device).unsqueeze(0)

            # 3. 调整 start 和 end 的形状以支持广播 (N, 1)
            starts = answer_start_idx[is_vla_data].unsqueeze(1)
            ends = (answer_start_idx[is_vla_data] + n_actions[is_vla_data]).unsqueeze(1)
            action_ends = n_actions[is_vla_data].unsqueeze(1)

            # 4. 生成掩码 (N, seq_len) 和 (N, action_seq_len)
            # 逻辑：当前索引 >= start 且 当前索引 < start + n
            mask_hidden = (range_hidden >= (starts - 1)) & (range_hidden < (ends - 1))
            mask_action = range_action < action_ends

            # 5. 使用布尔索引提取数据
            # 这会将所有 True 的位置“压扁”提取出来，直接得到 (diff_bsz, dim)
            vla_hidden_z = vla_hidden[mask_hidden]  # (diff_bsz, hidden_dim)
            action_gt = vla_action[mask_action]  # (diff_bsz, action_dim)
            vla_hidden_z_repeated = vla_hidden_z.repeat_interleave(self.diffloss_micro_batch_size, dim=0)
            action_gt_repeated = action_gt.repeat_interleave(self.diffloss_micro_batch_size, dim=0)
            latent_condition_embeds = self.latent_condition_projector(vla_hidden_z_repeated)
            diff_loss = self.diffloss(action_gt_repeated, latent_condition_embeds) / self.diffloss_micro_batch_size
        else:
            diff_loss = torch.tensor(0.0, device=device, dtype=ce_loss.dtype)

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0

        flow_loss = (v_psi - d_psi) ** 2
        # Use element-wise multiplication to make sure the gradient can always be propagated to the action expert 
        masked_loss = actions_valid_mask * flow_loss
        actions_valid_num = torch.sum(actions_valid_mask)
        flow_loss = torch.sum(masked_loss) / actions_valid_num.clamp(min=1)

        total_loss = self.loss_weights.ce_loss_weight * ce_loss + self.loss_weights.diffusion_loss_weight * diff_loss + self.loss_weights.flow_loss_weight * flow_loss
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "diffusion_loss": diff_loss,
            "flow_loss": flow_loss,
        }

    def forward(self, mode: str, batch: dict, **kwargs) -> dict:
        # Helper functions for distributed training
        if mode == "train":
            return self.compute_loss(batch)
        elif mode == "train_ar": 
            return self.compute_ar_loss(batch)
        elif mode == "train_flow":
            return self.compute_flow_loss(batch)
        elif mode == "infer_action":
            return self.infer_action(batch, **kwargs)
        elif mode == "infer_vla":
            return self.infer_vla(batch, **kwargs)
        elif mode == "infer_vlm":
            return self.infer_vlm(batch, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        

class LegendVLAInference(nn.Module):
    """
    Implementation of the VLA inference logic.
    This class is 'Device-Agnostic' - it focuses on the sequence of operations:
    Observation -> Preprocessing -> State Normalization -> Model Forward -> Action Unnormalization.
    """
    def __init__(
        self,
        model_config_path: str,
        checkpoint_path: str = None, 
        mode: str = "flow",
        use_mixed_precision: bool = True,
        tokenizer_padding: str = "longest",
        default_instruction: str | None = None,
        diffusion_sampling_steps: int = None,
        diffusion_use_ddim_sampling: bool = False,
        flow_sampling_steps: int = None,
        ar_max_new_tokens: int | None = None,
        ar_temperature: float = 1.0,
        ar_cfg: float = 1.0,
    ) -> None:
        super().__init__()
        model_config_path = pathlib.Path(model_config_path)
        model_cfg = OmegaConf.load(model_config_path)
        self.model: nn.Module = hydra.utils.instantiate(model_cfg.policy)
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        self.model.eval()

        # Setup sampling config.
        if diffusion_sampling_steps:
            self.model.diffloss.num_sampling_steps = diffusion_sampling_steps
        if diffusion_use_ddim_sampling:
            self.model.diffloss.use_ddim_sampling = diffusion_use_ddim_sampling
        if flow_sampling_steps:
            self.model.num_inference_steps = flow_sampling_steps

        # Setup Data Processor (Vision/Language)
        self.processor = hydra.utils.instantiate(model_cfg.vla_processor)
        if hasattr(self.processor, "tokenizer_padding"):
            self.processor.tokenizer_padding = tokenizer_padding

        self.normalizer, self.use_relative_action = self._load_normalizer(model_cfg)

        # Hyperparameters & Meta
        self.mode = mode
        self.dtype = torch.bfloat16 if use_mixed_precision else torch.float32
        self.default_instruction = default_instruction
        self.action_horizon = int(self.model.shape_meta["action"]["horizon"])
        self.action_dim = int(self.model.shape_meta["action"]["shape"][0])
        self.ar_max_new_tokens = ar_max_new_tokens or self.action_horizon
        self.ar_temperature = ar_temperature
        self.ar_cfg = ar_cfg
        
        self.metadata = {
            "mode": mode,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
        }

    def _load_checkpoint(self, path: str) -> None:
        """Load model weights from a given path."""
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        state_dict = torch.load(path, map_location="cpu")
        # Handle various state_dict keys
        for key in ["model", "module", "model_state_dict"]:
            if key in state_dict:
                state_dict = state_dict[key]
                break
        self.model.load_state_dict(state_dict)
        print(f"Successfully load model checkpoint from {path}")

    def _load_normalizer(self, model_cfg: Any) -> Tuple[Optional[Dict], bool]:
        """Load normalization stats for actions."""
        normalizer_path = None
        use_relative_action = False
        if hasattr(model_cfg, "training") and model_cfg.training.get("normalizer_path"):
            normalizer_path = pathlib.Path(model_cfg.training.normalizer_path)
        if hasattr(model_cfg, "dataset"):
            use_relative_action = bool(OmegaConf.select(model_cfg, "dataset.vla_dataset.use_relative_action", default=False))
            
        if normalizer_path and normalizer_path.exists():
            with open(normalizer_path, "rb") as f:
                return pickle.load(f), use_relative_action
        return None, use_relative_action

    def prepare_process(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw data (numpy/strings, after geometry transformation) into standard processor inputs."""
        instruction = obs.get("instruction") or self.default_instruction
        images = obs["image"]
        states = obs["states"]
        n_states = torch.full((1, ), states.shape[0], dtype=torch.int32)
        intrinsic = self._extract_intrinsic(obs["intrinsic"])
        
        processor_mode = "infer" if self.mode == "flow" else "infer-ar"
        processed = self.processor(
            text=instruction,
            images=images,
            states=states,
            actions=np.zeros((self.action_horizon, self.action_dim), dtype=np.float32),
            intrinsic=intrinsic,
            depth_images=obs.get("depth"),
            mode=processor_mode,
        )
        
        # Batching: Add [None, ...] dimension and convert to Torch
        from src.utils.pytorch_util import dict_apply
        processed = dict_apply(processed, lambda x: torch.from_numpy(x)[None, ...])
        if self.normalizer is not None: 
            if self.use_relative_action:
                states = self.normalizer["states"](states)
            else:
                states = self.normalizer["motions"](states)
        states = torch.from_numpy(states)[None, ...]
        return {"processed": processed, "states": states, "n_states": n_states}

    def build_model_inputs(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        """Construct tensors required by the model forward pass (Masks, Position ids)."""
        processed = prepared["processed"]
        input_ids = processed["input_ids"]
        batch_size = input_ids.shape[0]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": processed["attention_mask"],
            "pixel_values": processed["pixel_values"].to(self.dtype),
            "n_states": prepared["n_states"], 
            "states": prepared["states"].to(self.dtype),
            "is_vla_data": torch.ones(batch_size, dtype=torch.bool),
        }

        # Flow-matching specific logic (Causal Masks)
        if self.mode == "flow":
            inputs["n_actions"] = torch.zeros(batch_size, dtype=torch.long)
            inputs["answer_start_idx"] = processed["answer_start_idx"]
            
            m = self.model.module if hasattr(self.model, "module") else self.model
            causal_mask, vlm_pos, act_pos = m.build_causal_mask_and_position_ids(
                inputs["attention_mask"], inputs["answer_start_idx"], inputs["n_actions"], self.dtype
            )
            max_vlm_tokens = input_ids.shape[-1]
            vlm_mask, action_mask = m.split_full_mask_into_submasks(causal_mask, max_vlm_tokens)
            inputs.update({
                "causal_mask": causal_mask, 
                "vlm_position_ids": vlm_pos, 
                "action_position_ids": act_pos,
                "vlm_mask": vlm_mask,
                "action_mask": action_mask,
            })

        return inputs

    def post_process(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert model output [-1, 1] back to physical world units."""
        if self.normalizer is None:
            return actions
        key = "actions" if self.use_relative_action else "motions"
        return self.normalizer[key].unnormalize(actions)

    def _extract_intrinsic(self, intrinsic: np.ndarray) -> np.ndarray:
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        if intrinsic.shape == (3, 3):
            intrinsic = np.array([intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]])
        return intrinsic.reshape(-1)

    @torch.inference_mode()
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Step 3: Actual model forward pass."""
        if self.mode == "flow":
            return self.model("infer_action", inputs)
        else:
            return self.model("infer_vla", inputs, max_new_tokens=self.ar_max_new_tokens,
                              temperature=self.ar_temperature, cfg=self.ar_cfg)
