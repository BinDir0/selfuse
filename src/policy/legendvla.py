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
from einops import rearrange

from src.model.common.normalizer import LinearNormalizer
from src.model.common.kv_cache import KVCache
from src.model.common.modules import (
    ActionEncoder,
    SinusoidalPosEmb,
)
from src.utils.monitor import log_execution_time

log = logging.getLogger(__name__)


class LegendVLA(nn.Module):
    @log_execution_time(log)
    def __init__(
        self, 
        cfg,
        shape_meta,
        vision_tower,
        multi_modal_projector,
        joint_model,
    ):
        super().__init__()
        self.cfg = cfg
        self.shape_meta = shape_meta
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.use_lm_head = cfg.get("use_lm_head", False)

        self.max_image_text_tokens = cfg.max_image_text_tokens
        self.num_proprio_tokens = shape_meta["obs"]["state"]["horizon"]
        self.num_human_action_tokens = shape_meta["action"]["horizon"]
        self.total_num_tokens = (
            self.max_image_text_tokens
            + self.num_proprio_tokens
            + self.num_human_action_tokens
        )

        # Get hidden sizes from joint_model config
        self.image_text_hidden_size = joint_model.config.mixture.vlm.hidden_size
        self.proprio_hidden_size = joint_model.config.mixture.proprio.hidden_size
        self.human_action_hidden_size = joint_model.config.mixture.human_action.hidden_size

        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = shape_meta["action"]["horizon"]
        self.human_action_dim = shape_meta["action"]["shape"][0]
        self.proprio_dim = shape_meta["obs"]["state"]["shape"][0]
        self.flow_sig_min = cfg.get("flow_sig_min", 0.001)

        # text input only
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters

        # Vision
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector

        # Mixtures
        self.joint_model = joint_model

        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = cfg.action_expert_adaptive_mode
        if self.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.human_action_encoder = ActionEncoder(
                self.human_action_dim,
                self.human_action_hidden_size,
                time_cond=False,
            )
            self.time_embedding = SinusoidalPosEmb(
                cfg.time_hidden_size, cfg.time_max_period
            )
        else:  # matching pi0
            self.human_action_encoder = ActionEncoder(
                self.human_action_dim,
                self.human_action_hidden_size,
                time_cond=True,
            )
            self.time_embedding = SinusoidalPosEmb(
                self.human_action_hidden_size, cfg.time_max_period
            )
        self.proprio_encoder = nn.Linear(
            self.proprio_dim,
            self.proprio_hidden_size,
        )
        # Action decoder
        self.human_action_decoder = nn.Linear(
            self.human_action_hidden_size,
            self.human_action_dim,
        )

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.image_text_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights

        self.normalizer = LinearNormalizer()

    @property
    def human_action_expert_parameters(self):
        """
        Get all trainable parameters for the action and proprio experts.
        
        Returns:
            List[torch.nn.Parameter]: Parameters from:
                - Human action encoder
                - Human action decoder  
                - Proprio encoder
                - Human action mixture (shared with proprio)
        
        Note: Human action and proprio share weights through tie_action_proprio_weights()
        """
        return (
            list(self.human_action_encoder.parameters())
            + list(self.human_action_decoder.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.joint_model.mixtures["human_action"].parameters())
        )  # note: human_action and proprio share weights

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
        return params

    @property
    def trainable_gemma_parameters(self):
        """
        Get all trainable parameters for the Gemma language model.
        
        Excludes parameters that are not needed for training based on
        _check_gemma_unused_parameter_by_name().
        
        Returns:
            List[torch.nn.Parameter]: Trainable Gemma parameters
        """
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                gemma_parameters.append(param)
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
            if not self._check_gemma_unused_parameter_by_name(name):
                if "lora_" in name:
                    gemma_parameters.append(param)
        return gemma_parameters

    @log_execution_time(log)
    def load_pretrained_weights(self):
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

    # TODO: may need to change this when we use Knowledge Insulation training recipe
    def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
        """
        Check if a Gemma parameter should be excluded from training.
        
        Excludes parameters from the last layer that are not needed for training:
        - Post-attention normalization
        - MLP layers
        - Output projection (o_proj)
        - Value projection (v_proj)
        
        Args:
            name (str): Parameter name to check
        
        Returns:
            bool: True if parameter should be excluded, False otherwise
        """
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        if (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_proj" in name
            or f"{last_hidden_layer_index}.self_attn.v_proj" in name
        ):  # final norm is not initialized
            return True
        return False

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
            if not self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in lm part of the joint model")

    def freeze_unused_weights(self):
        """
        Freeze weights that are not used during training.
        
        Freezes:
        - Text embedding weights
        - Last layer post-attention, MLP, and output projection weights in VLM
        - Any LoRA weights in the unused parts
        """
        self.embed_tokens.weight.requires_grad = False
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = False

    def freeze_all_weights(self):
        """
        Freeze all trainable parameters in the model.
        
        Sets requires_grad=False for all parameters, making the model non-trainable.
        Useful for inference-only scenarios.
        """
        for _, param in self.named_parameters():
            param.requires_grad = False

    def tie_action_proprio_weights(self):
        """
        Tie the proprio and human action mixture weights.
        
        This method shares the same mixture module between proprio and human action,
        allowing them to share learned representations and reduce model size.
        """
        self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["human_action"]

    def build_text_cache(self):
        """
        Create a new KV cache for text generation.
        
        Returns:
            KVCache: Empty key-value cache for storing attention states during text generation
        """
        return KVCache()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

        self.normalizer.eval()
    
        for param in self.normalizer.parameters():
            param.requires_grad = False

    # ---------- Input preparation ---------- #

    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Build causal attention masks and position IDs for different token types.
        
        Creates block-diagonal attention patterns:
        - Image/text tokens can attend to themselves
        - Proprio tokens can attend to image/text and themselves  
        - Action tokens can attend to image/text, proprio, and themselves (causal)
        
        Args:
            attention_mask (torch.Tensor): [B, seq_len] Attention mask indicating valid tokens
            dtype (torch.dtype): Data type for the causal mask
        
        Returns:
            Tuple containing:
                - causal_mask (torch.FloatTensor): [B, 1, total_len, total_len] 
                  Causal attention mask with block structure (broadcasts to all heads)
                - vlm_position_ids (torch.LongTensor): [B, seq_len] Position IDs for VLM tokens
                - proprio_position_ids (torch.LongTensor): [B, num_proprio] Position IDs for proprio tokens
                - human_action_position_ids (torch.LongTensor): [B, num_actions] Position IDs for action tokens
                
        block attention --- padding for unused text tokens

                       img/text img/text (padding) proprio human_action human_action
        img/text          x        x
        img/text          x        x
        (padding)
        proprio           x        x                  x
        human_action      x        x                  x         x           x
        human_action      x        x                  x         x           x 
        """
        bsz = attention_mask.size(0)
        proprio_start = self.max_image_text_tokens
        proprio_end = proprio_start + self.num_proprio_tokens
        human_action_start = proprio_end
        image_text_token_cnts = torch.sum(attention_mask, dim=1)
        causal_mask = torch.full(
            (bsz, self.total_num_tokens, self.total_num_tokens),
            torch.finfo(dtype).min,
            dtype=dtype,
        )  # smallest value, avoid using inf for softmax nan issues with padding
        for idx in range(bsz):
            cnt = image_text_token_cnts[idx].item()
            causal_mask[idx, :cnt, :cnt] = 0  # image/text attend to itself
            causal_mask[idx, proprio_start:, :cnt] = (
                0  # proprio/human_action attend to image/text
            )
        causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
            0  # proprio attend to itself
        )
        causal_mask[:, human_action_start:, proprio_start:] = (
            0  # human_action attend to proprio, and itself (causal)
        )

        # add the head dimension for broadcasting to all attention heads
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, 1, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks --- start at 1
        vlm_position_ids = torch.arange(1, self.max_image_text_tokens + 1).repeat(
            bsz, 1
        )
        proprio_position_ids = torch.arange(1, self.num_proprio_tokens + 1).repeat(
            bsz, 1
        )
        human_action_position_ids = torch.arange(
            self.num_proprio_tokens + 1,
            self.num_proprio_tokens + self.num_human_action_tokens + 1,
        ).repeat(bsz, 1)
        return causal_mask, vlm_position_ids, proprio_position_ids, human_action_position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Split the full causal mask into separate masks for different model components.
        
        Args:
            causal_mask (torch.FloatTensor): [B, 1, total_len, total_len] 
              Full causal attention mask (broadcasts to all heads)
        
        Returns:
            Tuple containing:
                - image_text_proprio_mask (torch.FloatTensor): [B, 1, seq_len+proprio_len, seq_len+proprio_len]
                  Attention mask for image/text/proprio tokens (broadcasts to all heads)
                - human_action_mask (torch.FloatTensor): [B, 1, action_len, total_len]
                  Attention mask for human action tokens (broadcasts to all heads)
        """
        image_text_proprio_mask = causal_mask[
            ...,
            : self.max_image_text_tokens + self.num_proprio_tokens,
            : self.max_image_text_tokens + self.num_proprio_tokens,
        ]
        human_action_mask = causal_mask[..., -self.num_human_action_tokens :, :]
        return image_text_proprio_mask, human_action_mask

    # TODO: need to change this when we use Knowledge Insulation training recipe
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
    ) -> torch.FloatTensor:
        """
        Forward pass through SigLIP vision encoder and text embedding, then combine them.
        
        Args:
            input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
            pixel_values (torch.FloatTensor): [B, C, H, W] or [B, T, C, H, W] Image pixel values (normalized)
        
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
            pixel_values = rearrange(pixel_values, "B T C H W -> (B T) C H W")
            # pixel_values = pixel_values.view(B * T, C, H, W)
        else:
            T = None

        selected_image_feature = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(selected_image_feature)

        if T is not None:
            image_features = rearrange(image_features, "(B T) P D -> B (T P) D", B=B, T=T)
            # image_features = image_features.view(B, -1, image_features.shape[-1])

        # normalize the image features
        _, _, embed_dim = image_features.shape
        bsz, seq_len = input_ids.shape
        scaled_image_features = image_features / (self.image_text_hidden_size**0.5)

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
            final_embedding[i, image_indices] = scaled_image_features[
                i, :num_image_tokens
            ]
        return final_embedding

    def infer_human_action(
        self,
        input: dict,
    ) -> torch.FloatTensor:
        """
        Inference function for human action generation using flow matching.
        
        Args:
            input (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - pixel_values (torch.FloatTensor): [B, 3, H, W] or [B, T, 3, H, W] Image pixel values (normalized)
                - image_text_proprio_mask (torch.FloatTensor): [B, 1, seq_len, seq_len] 
                  Attention mask for image/text/proprio tokens (broadcasts to all heads)
                - human_action_mask (torch.FloatTensor): [B, 1, action_len, total_len] 
                  Attention mask for human action tokens (broadcasts to all heads)
                - vlm_position_ids (torch.LongTensor): [B, seq_len] Position IDs for VLM tokens
                - proprio_position_ids (torch.LongTensor): [B, num_proprio] Position IDs for proprio tokens
                - human_action_position_ids (torch.LongTensor): [B, num_actions] Position IDs for action tokens
                - proprios (torch.FloatTensor): [B, num_proprio, proprio_dim] Proprioceptive state features
        
        Returns:
            torch.FloatTensor: [B, horizon_steps, human_action_dim] Generated human action sequence
        """
        # Extract inputs from dict
        input_ids = input["input_ids"]
        pixel_values = input["pixel_values"]
        image_text_proprio_mask = input["image_text_proprio_mask"]
        human_action_mask = input["human_action_mask"]
        vlm_position_ids = input["vlm_position_ids"]
        proprio_position_ids = input["proprio_position_ids"]
        human_action_position_ids = input["human_action_position_ids"]
        proprios = input["proprios"]

        # normalize proprio
        wrist_dim = self.shape_meta['obs']['state']['wrist']['shape'][0]
        proprios[..., wrist_dim:] = self.normalizer['state/hand'](proprios[..., wrist_dim:])

        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)
        
        # forward pass thru the vlm, proprio, cache the kv
        _, kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )

        # sample pure action noise
        human_action = torch.randn(
            (bsz, self.horizon_steps, self.human_action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of vlm and proprio
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                human_action_embeds = self.human_action_encoder(human_action)
            else:
                human_action_embeds = self.human_action_encoder(human_action, time_cond)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            human_action_embeds = self.joint_model(
                attention_mask=human_action_mask,
                position_ids_all={"human_action": human_action_position_ids},
                embeds_all={"human_action": human_action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
            )["human_action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            human_action_vel = self.human_action_decoder(human_action_embeds)
            human_action += delta_t * human_action_vel
            t += delta_t

        # normalize action
        human_action[..., wrist_dim:] = self.normalizer['action/hand'](human_action[..., wrist_dim:])
        return human_action

    def infer_human_action_naive(
        self,
        input: dict,
    ) -> torch.FloatTensor:
        """
        Naive inference function for human action generation (runs VLM at each step).
        
        Args:
            input (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - pixel_values (torch.FloatTensor): [B, 3, H, W] or [B, T, 3, H, W] Image pixel values (normalized)
                - causal_mask (torch.FloatTensor): [B, 1, total_len, total_len] 
                  Full causal attention mask for all tokens (broadcasts to all heads)
                - vlm_position_ids (torch.LongTensor): [B, seq_len] Position IDs for VLM tokens
                - proprio_position_ids (torch.LongTensor): [B, num_proprio] Position IDs for proprio tokens
                - human_action_position_ids (torch.LongTensor): [B, num_actions] Position IDs for action tokens
                - proprios (torch.FloatTensor): [B, num_proprio, proprio_dim] Proprioceptive state features
        
        Returns:
            torch.FloatTensor: [B, horizon_steps, human_action_dim] Generated human action sequence
        """
        # Extract inputs from dict
        input_ids = input["input_ids"]
        pixel_values = input["pixel_values"]
        causal_mask = input["causal_mask"]
        vlm_position_ids = input["vlm_position_ids"]
        proprio_position_ids = input["proprio_position_ids"]
        human_action_position_ids = input["human_action_position_ids"]
        proprios = input["proprios"]

        # normalize proprio
        wrist_dim = self.shape_meta['obs']['state']['wrist']['shape'][0]
        proprios[..., wrist_dim:] = self.normalizer['state/hand'](proprios[..., wrist_dim:])

        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # encode proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # sample pure action noise
        human_action = torch.randn(
            (bsz, self.horizon_steps, self.human_action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- run vlm in each step, which is unnecessary
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                human_action_embeds = self.human_action_encoder(human_action)
            else:
                human_action_embeds = self.human_action_encoder(human_action, time_cond)
            human_action_embeds = self.joint_model(
                attention_mask=causal_mask,
                position_ids_all={
                    "vlm": vlm_position_ids,
                    "proprio": proprio_position_ids,
                    "human_action": human_action_position_ids,
                },
                embeds_all={
                    "vlm": inputs_embeds.clone(),  # clone needed due to modified in-place
                    "proprio": proprio_embeds.clone(),
                    "human_action": human_action_embeds,
                },
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="no_append",  # no new tokens
            )["human_action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            human_action_vel = self.human_action_decoder(human_action_embeds)
            human_action += delta_t * human_action_vel
            t += delta_t

        # normalize action
        human_action[..., wrist_dim:] = self.normalizer['action/hand'](human_action[..., wrist_dim:])
        return human_action

    def infer_text(
        self,
        input: dict,
    ) -> Tuple:
        """
        Text generation inference function using VLM.
        
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
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

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

    # TODO: add indicator for only one hand or two hands
    def forward(
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
                - proprio_position_ids (torch.LongTensor): [B, num_proprio] Position IDs for proprio tokens
                - human_action_position_ids (torch.LongTensor): [B, num_actions] Position IDs for action tokens
                - proprios (torch.FloatTensor): [B, num_proprio, proprio_dim] Proprioceptive state features
                - human_actions (torch.FloatTensor): [B, horizon_steps, human_action_dim] Ground truth human actions
                - t (torch.FloatTensor): [B] Time steps for flow matching (0 to 1)

        Note: 
            Proprioceptive state structure: [wrist, hand]
            - Wrist: [translation(left, right), rotation(left, right)]
            - Hand: [left_hand_features, right_hand_features]
            Action structure mirrors the state
        
        Returns:
            torch.FloatTensor: [1] Flow matching loss (mean squared error)
        """
        # Extract inputs from batch dict
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]
        causal_mask = batch["causal_mask"]
        vlm_position_ids = batch["vlm_position_ids"]
        proprio_position_ids = batch["proprio_position_ids"]
        human_action_position_ids = batch["human_action_position_ids"]
        proprios = batch["proprios"]
        human_actions = batch["human_actions"]
        t = batch["t"]

        # normalize proprio
        wrist_dim = self.shape_meta['obs']['state']['wrist']['shape'][0]
        proprios[..., wrist_dim:] = self.normalizer['state/hand'](proprios[..., wrist_dim:])

        """flow matching loss for action prediction, no use of kv cache"""
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(human_actions, device=t.device, dtype=t.dtype)
        x1 = human_actions
        # normalize action
        x1[..., wrist_dim:] = self.normalizer['action/hand'](x1[..., wrist_dim:])
        psi_t = self.psi_t(x0, x1, t)

        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)
        
        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            human_action_embeds = self.human_action_encoder(psi_t)
        else:
            human_action_embeds = self.human_action_encoder(psi_t, time_cond)
        human_action_embeds = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
                "human_action": human_action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "human_action": human_action_embeds,
            },
            time_cond=time_cond,
            kv_caches={},  # no caching during training
        )["human_action"]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.human_action_decoder(human_action_embeds)

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)


class LegendVLAInference(LegendVLA):
    def forward(
        self,
        input: dict,
    ) -> torch.FloatTensor:
        """
        Inference wrapper for LegendVLA that calls infer_human_action.
        
        Args:
            input (dict): Input dictionary containing:
                - input_ids (torch.LongTensor): [B, seq_len] Text token IDs including image tokens
                - pixel_values (torch.FloatTensor): [B, 3, H, W] Image pixel values (normalized)
                - image_text_proprio_mask (torch.FloatTensor): [B, 1, seq_len, seq_len] 
                  Attention mask for image/text/proprio tokens (broadcasts to all heads)
                - human_action_mask (torch.FloatTensor): [B, 1, action_len, total_len] 
                  Attention mask for human action tokens (broadcasts to all heads)
                - vlm_position_ids (torch.LongTensor): [B, seq_len] Position IDs for VLM tokens
                - proprio_position_ids (torch.LongTensor): [B, num_proprio] Position IDs for proprio tokens
                - human_action_position_ids (torch.LongTensor): [B, num_actions] Position IDs for action tokens
                - proprios (torch.FloatTensor): [B, num_proprio, proprio_dim] Proprioceptive state features
        
        Returns:
            torch.FloatTensor: [B, horizon_steps, human_action_dim] Generated human action sequence
        """
        return super().infer_human_action(input)

