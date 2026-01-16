"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

"""

import logging
from typing import Optional, Tuple

import hydra
import torch
from torch import nn

from src.model.common.kv_cache import KVCache
from src.model.common.modules import (
    ActionEncoder,
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
        depth_encoder, 
        vision_tower,
        multi_modal_projector,
        depth_modal_projector,
        joint_model,
    ):
        super().__init__()
        self.cfg = cfg
        self.shape_meta = shape_meta    
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.use_lm_head = cfg.get("use_lm_head", False)

        self.max_vlm_tokens = cfg.max_vlm_tokens
        self.num_action_tokens = shape_meta["action"]["horizon"]
        self.max_num_tokens = (
            self.max_vlm_tokens
            + self.num_action_tokens
        )

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
            self.depth_encoder = depth_encoder
            self.depth_modal_projector = depth_modal_projector

        # Mixtures
        self.joint_model = joint_model

        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = cfg.action_expert_adaptive_mode
        if self.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=False,
            )
            self.time_embedding = nn.Sequential(
                SinusoidalPosEmb(cfg.time_hidden_size, cfg.time_min_period, cfg.time_max_period), 
                TimeEncoder(cfg.time_hidden_size), 
            )
        else:  # matching pi0
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=True,
            )
            self.time_embedding = SinusoidalPosEmb(
                self.action_hidden_size, cfg.time_max_period
            )
        # Action decoder
        self.action_decoder = nn.Linear(
            self.action_hidden_size,
            self.action_dim,
        )

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.vlm_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights

        self.CELoss = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
        self.loss_weights = cfg.loss_weights

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
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "multi_modal_projector" in k:
                new_key = k.replace("multi_modal_projector.", "")
                multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(
            multi_modal_projector_state_dict, strict=True
        )
        log.info("Loaded pre-trained weights for projector")

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
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "paligemma_with_expert.paligemma.model.multi_modal_projector" in k:
                new_key = k.replace("paligemma_with_expert.paligemma.model.multi_modal_projector.", "")
                if new_key in multi_modal_projector_state_dict:
                    # Convert to target dtype if needed
                    multi_modal_projector_state_dict[new_key] = v.to(dtype=target_dtype)
                    loaded_my_model_params.add(f"multi_modal_projector.{new_key}")
                    used_pi05_params.add(k)
        self.multi_modal_projector.load_state_dict(
            multi_modal_projector_state_dict, strict=True
        )
        log.info("Loaded multi-modal projector weights")

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
        self, attention_mask: torch.Tensor, answer_start_idx: torch.Tensor, dtype: torch.dtype
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
            causal_mask[idx, :cnt, :start] = 0  # image/text/answer attend to image/text
            mask = torch.tril(torch.ones((answer_len, answer_len), dtype=torch.bool, device=device))
            causal_mask[idx, start:cnt, start:cnt] = torch.where(
                mask, 0, torch.finfo(dtype).min
            ) # answer tokens attend to answer tokens before them
            causal_mask[idx, action_start:, :start] = (
                0  # action attend to image/text
            )
        causal_mask[:, action_start:, action_start:] = (
            0  # action attend to itself
        )

        # add the head dimension for broadcasting to all attention heads
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, 1, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks --- start at 1
        vlm_position_ids = torch.arange(1, max_vlm_tokens + 1, device=device).repeat(
            bsz, 1
        )
        action_position_ids = torch.arange(
            1,
            self.num_action_tokens + 1,
            device=device,
        ).repeat(bsz, 1)
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
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Build causal mask and position IDs for text generation.
        
        Creates attention masks for autoregressive text generation with optional KV cache.
        - Prefill phase: No masking (all tokens can attend to each other)
        - Generation phase: No masking (query can attend to all cached tokens)
        
        Args:
            q_len (int): Length of the current query sequence
            attention_mask (torch.Tensor): [B, seq_len] Attention mask for input tokens
            kv_cache (Optional[KVCache]): Optional KV cache for generation
        
        Returns:
            Tuple containing:
                - causal_mask (torch.FloatTensor): [B, 1, q_len, kv_len] Attention mask (broadcasts to all heads)
                - position_ids (torch.LongTensor): [B, q_len] Position IDs for query tokens
        """
        dtype, device = attention_mask.dtype, attention_mask.device
        bsz = attention_mask.size(0)

        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask any token, because we're in the prefill phase
            # assume no padding
            causal_mask = torch.full((bsz, q_len, q_len), 0, dtype=dtype, device=device)
        else:
            assert q_len == 1, "Using KV cache so should only use one single token"
            kv_len = kv_cache.num_items() + q_len
            # also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # this only works when we have no padding
            causal_mask = torch.full(
                (bsz, q_len, kv_len), 0, dtype=dtype, device=device
            )

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
    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        depth_values: Optional[torch.FloatTensor] = None,
        depth_ids: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass through SigLIP vision encoder and text embedding, then combine them.
        
        Args:
            input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
            pixel_values (torch.FloatTensor): [B, C, H, W] or [B, T, C, H, W] Image pixel values (normalized)
            depth_values (Optional[torch.FloatTensor]): [B, C, H, W] or [B, T, C, H, W] Depth images (optional)
        
        Returns:
            torch.FloatTensor: [B, seq_len, hidden_size] Combined image and text embeddings
        """
        dtype, device = pixel_values.dtype, pixel_values.device

        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.embed_tokens(input_ids)

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
        rgb_image_features = self.multi_modal_projector(rgb_image_features)
        
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
            depth_image_features = self.depth_modal_projector(depth_image_features)
        else: 
            depth_image_features = None

        if T is not None:
            # image_features = rearrange(image_features, "(B T) P D -> B (T P) D", B=B, T=T)
            rgb_image_features = rgb_image_features.view(B, -1, rgb_image_features.shape[-1])
            if depth_image_features is not None:
                depth_image_features = depth_image_features.view(Bd, -1, depth_image_features.shape[-1])

        # normalize the image features
        _, _, embed_dim = rgb_image_features.shape
        bsz, seq_len = input_ids.shape
        scaled_image_features = rgb_image_features / (self.vlm_hidden_size**0.5)
        if depth_image_features is not None:
            scaled_depth_features = depth_image_features / (self.vlm_hidden_size**0.5)
        else: 
            scaled_depth_features = None

        # put embedding together - image, text, padding
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim), self.pad_token_id, dtype=dtype, device=device
        )

        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.image_token_index
        # autocast does not cast nn.Embedding to the correct dtype, we need to cast manually
        final_embedding[text_mask] = inputs_embeds[text_mask].to(final_embedding.dtype)
        for i in range(bsz):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_indices)
            if depth_ids is not None and depth_ids[i] >= 0:
                # Each RGB token is paired with corresponding depth token
                paired_image_features = torch.cat([
                    scaled_image_features[i].view(T, -1, embed_dim),
                    scaled_depth_features[depth_ids[i]].view(T, -1, embed_dim),
                ], dim=1) # [T, 2*num_patches, embed_dim] 
                paired_image_features = paired_image_features.view(-1, embed_dim)
            else: 
                paired_image_features = scaled_image_features[i]
            final_embedding[i, image_indices] = paired_image_features[:num_image_tokens]
        return final_embedding

    @torch.inference_mode()
    def infer_action(
        self,
        input: dict,
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
            depth_ids = input["depth_ids"]
        else:
            depth_values = None
            depth_ids = None
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values, depth_values, depth_ids)
        
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

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of vlm
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        return action

    @torch.inference_mode()
    def infer_action_naive(
        self,
        input: dict,
    ) -> torch.FloatTensor:
        """
        Naive inference function for action generation (runs VLM at each step).
        
        Args:
            input (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - pixel_values (torch.FloatTensor): [B, 3, H, W] or [B, T, 3, H, W] Image pixel values (normalized)
                - causal_mask (torch.FloatTensor): [B, 1, total_len, total_len] 
                  Full causal attention mask for all tokens (broadcasts to all heads)
                - vlm_position_ids (torch.LongTensor): [B, seq_len] Position IDs for VLM tokens
                - action_position_ids (torch.LongTensor): [B, num_actions] Position IDs for action tokens
        
        Returns:
            torch.FloatTensor: [B, horizon_steps, action_dim] Generated action sequence
        """
        # Extract inputs from dict
        input_ids = input["input_ids"]
        pixel_values = input["pixel_values"]
        causal_mask = input["causal_mask"]
        vlm_position_ids = input["vlm_position_ids"]
        action_position_ids = input["action_position_ids"]

        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        if 'depth_values' in input:
            depth_values = input["depth_values"]
            depth_ids = input["depth_ids"]
        else:
            depth_values = None
            depth_ids = None
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values, depth_values, depth_ids)

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- run vlm in each step, which is unnecessary
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            action_embeds = self.joint_model(
                attention_mask=causal_mask,
                position_ids_all={
                    "vlm": vlm_position_ids,
                    "action": action_position_ids,
                },
                embeds_all={
                    "vlm": inputs_embeds.clone(),  # clone needed due to modified in-place
                    "action": action_embeds,
                },
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="no_append",  # no new tokens
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        return action

    @torch.inference_mode()
    def infer_discrete_action(
        self,
        input: dict,
    ) -> Tuple:
        """
        Discrete action generation inference function using Autoregressive VLA.
        
        Args:
            input (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - pixel_values (torch.FloatTensor): [B, 3, H, W] or [B, T, 3, H, W] Image pixel values (normalized)
                - attention_mask (torch.Tensor): [B, seq_len] Attention mask for text tokens
                - kv_cache (Optional[KVCache]): Optional KV cache for autoregressive generation
        
        Returns:
            Tuple: Output dictionary containing:
                - logits (torch.FloatTensor): [B, seq_len, vocab_size] Next token logits
                - kv_cache (Optional[KVCache]): Updated KV cache if provided
        """
        # Extract inputs from dict
        input_ids = input["input_ids"]
        pixel_values = input["pixel_values"]
        attention_mask = input["attention_mask"]
        kv_cache = input.get("kv_cache", None)
        q_len = input_ids.size(1)

        # text tokens + image tokens
        if 'depth_values' in input:
            depth_values = input["depth_values"]
            depth_ids = input["depth_ids"]
        else:
            depth_values = None
            depth_ids = None
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values, depth_values, depth_ids)

        # build causal mask and position ids for text
        (
            causal_mask,
            position_ids,
        ) = self.build_causal_mask_and_position_ids_for_text(
            q_len, attention_mask, kv_cache
        )

        hidden_states = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={"vlm": position_ids},
            embeds_all={"vlm": inputs_embeds},
            kv_caches={"vlm": kv_cache},
            cache_mode="append",  # new tokens for the active mixture
            final_layer_post_attn_skip_names=[],  # do not skip vlm last layer
        )["vlm"]
        logits = self.lm_head(hidden_states)
        output = {
            "logits": logits,
        }
        if kv_cache is not None:
            output["kv_cache"] = kv_cache
        return output

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
            depth_ids = batch["depth_ids"]
        else:
            depth_values = None
            depth_ids = None
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values, depth_values, depth_ids)
        
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
        logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        labels = labels[:, 1:].contiguous().view(-1)

        ce_loss = self.CELoss(logits, labels)

        return {
            "ce_loss": ce_loss,
        }


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
            depth_ids = batch["depth_ids"]
        else:
            depth_values = None
            depth_ids = None
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values, depth_values, depth_ids)

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)
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
        if actions_valid_num == 0:
            actions_valid_num = 1
        flow_loss = torch.sum(masked_loss) / actions_valid_num
        return {
            "flow_loss": flow_loss,
        }


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
            depth_ids = batch["depth_ids"]
        else:
            depth_values = None
            depth_ids = None
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values, depth_values, depth_ids)
        
        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)
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

        logits = self.lm_head(hidden_states)
        logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
        labels = labels[:, 1:].contiguous().view(-1)

        ce_loss = self.CELoss(logits, labels)

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0

        flow_loss = (v_psi - d_psi) ** 2
        # Use element-wise multiplication to make sure the gradient can always be propagated to the action expert 
        masked_loss = actions_valid_mask * flow_loss
        actions_valid_num = torch.sum(actions_valid_mask)
        if actions_valid_num == 0:
            actions_valid_num = 1
        flow_loss = torch.sum(masked_loss) / actions_valid_num

        total_loss = self.loss_weights.ce_loss_weight * ce_loss + self.loss_weights.flow_loss_weight * flow_loss
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "flow_loss": flow_loss,
        }

    def forward(self, mode: str, batch: dict) -> dict:
        if mode == "train":
            return self.compute_loss(batch)
        elif mode == "train_ar": 
            return self.compute_ar_loss(batch)
        elif mode == "train_flow":
            return self.compute_flow_loss(batch)
        elif mode == "infer_action":
            return self.infer_action(batch)
        elif mode == "infer_action_naive":
            return self.infer_action_naive(batch)
        elif mode == "infer_discrete_action":
            return self.infer_discrete_action(batch)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        

class LegendVLAInference(LegendVLA):
    def forward(
        self,
        input: dict,
    ) -> torch.FloatTensor:
        """
        Inference wrapper for LegendVLA that calls infer_action.
        
        Args:
            input (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - pixel_values (torch.FloatTensor): [B, 3, H, W] Image pixel values (normalized)
                - vlm_mask (torch.FloatTensor): [B, 1, seq_len, seq_len] 
                  Attention mask for image/text/answer tokens (broadcasts to all heads)
                - action_mask (torch.FloatTensor): [B, 1, action_len, total_len] 
                  Attention mask for action tokens (broadcasts to all heads)
                - vlm_position_ids (torch.LongTensor): [B, seq_len] Position IDs for VLM tokens
                - action_position_ids (torch.LongTensor): [B, num_actions] Position IDs for action tokens
        
        Returns:
            torch.FloatTensor: [B, horizon_steps, action_dim] Generated action sequence
        """
        return super().infer_action(input)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import hydra
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # allows arbitrary python code execution in configs using the ${eval:''} resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load("src/config/experiment/pretrain_legendvla_deepspeed.yaml")
    config = OmegaConf.to_yaml(cfg.policy, resolve=True)
    print(config)
    model = hydra.utils.instantiate(cfg.policy)
    model.load_pretrained_vlm_weights()
    from src.utils.embedding_analysis import analyze_embedding_distribution, print_analysis_report
    model.init_motion_token_embeddings([i for i in range(257152, 257216)])
    embeddings = model.embed_tokens.weight.data[257152:257216]
    print(f"embeddings shape: {embeddings.shape}")
    results = analyze_embedding_distribution(
        embeddings,
        sample_size=200,
        plot=True,
        save_path=f"outputs/embedding_analysis.png",
    )
    print_analysis_report(results)
    # model.load_pretrained_pi05_weights()
