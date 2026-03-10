"""
Utility functions extracted from LegendVLA.

Contains:
- Weight loading (load_pretrained_vlm_weights, load_pretrained_pi05_weights)
- Weight freezing (freeze_non_lora_weights_in_vlm, freeze_non_lora_weights_in_ae, freeze_all_weights)
- Token embedding initialization (init_motion_token_embeddings)
- Causal mask and position ID construction (build_causal_mask_and_position_ids, etc.)
"""

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from src.model.common.kv_cache import KVCache
from src.utils.monitor import log_execution_time

log = logging.getLogger(__name__)


# ---------- Token embedding initialization ---------- #

@torch.no_grad()
def init_motion_token_embeddings(embed_tokens: nn.Embedding, vlm_hidden_size: int, motion_token_list):
    """
    Initialize the motion token embeddings.

    Args:
        embed_tokens: The embedding layer to initialize
        vlm_hidden_size: Hidden size of the VLM
        motion_token_list: List of motion token IDs
    """
    device = embed_tokens.weight.device
    indices = torch.LongTensor(motion_token_list).to(device)
    init_values = torch.randn(
        len(indices),
        vlm_hidden_size,
        dtype=embed_tokens.weight.dtype,
        device=device,
    ) * 0.02
    embed_tokens.weight[indices] = init_values


# ---------- Weight loading ---------- #

@log_execution_time(log)
def load_pretrained_vlm_weights(model):
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
    pretrained_model_path = getattr(model.cfg, 'pretrained_model_path', None)
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
    embed_tokens_state_dict = model.embed_tokens.state_dict()
    for k, v in tensors.items():
        if "embed_tokens" in k:
            new_key = k.replace("language_model.model.embed_tokens.", "")
            embed_tokens_state_dict[new_key] = v
    model.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)
    log.info("Loaded pre-trained weights for embed tokens")

    # load vision tower --- "vision_tower.vision_model" -> "vision_model"
    vision_tower_state_dict = model.vision_tower.state_dict()
    for k, v in tensors.items():
        if "vision_tower" in k:
            new_key = k.replace("vision_tower.", "")
            vision_tower_state_dict[new_key] = v
    model.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
    log.info("Loaded pre-trained weights for vision tower")

    # load projector --- "multi_modal_projector.linear" -> "linear"
    # Note: The new projector has concatenated RGB (1152) + Depth (768) = 1920 input dims
    # We only load the first 1152 dims (RGB) from pretrained weights, and zero-init the rest (Depth)
    multi_modal_projector_state_dict = model.multi_modal_projector.state_dict()
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
    model.multi_modal_projector.load_state_dict(
        multi_modal_projector_state_dict, strict=True
    )
    log.info("Loaded pre-trained weights for projector (RGB part only, Depth part zero-initialized)")

    # load lm --- do not change any lora weights
    joint_model_state_dict = model.joint_model.state_dict()
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
    model.joint_model.load_state_dict(joint_model_state_dict, strict=False)
    log.info("Loaded pre-trained weights for lm part of the joint model")


@log_execution_time(log)
def load_pretrained_pi05_weights(model):
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
    pretrained_model_path = getattr(model.cfg, 'pretrained_pi05_model_path', None)
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
    target_dtype = next(model.parameters()).dtype
    source_dtype = next(iter(tensors.values())).dtype
    log.info(f"Target model dtype: {target_dtype}, Pi0.5 checkpoint dtype: {source_dtype}")
    if target_dtype != source_dtype:
        log.info(f"Will convert loaded weights from {source_dtype} to {target_dtype}")

    # Track which parameters are loaded
    loaded_my_model_params = set()
    used_pi05_params = set()

    # load vision tower --- "paligemma_with_expert.paligemma.model.vision_tower.vision_model" -> "vision_model"
    vision_tower_state_dict = model.vision_tower.state_dict()
    for k, v in tensors.items():
        if "paligemma_with_expert.paligemma.model.vision_tower" in k:
            new_key = k.replace("paligemma_with_expert.paligemma.model.vision_tower.", "")
            if new_key in vision_tower_state_dict:
                # Convert to target dtype if needed
                vision_tower_state_dict[new_key] = v.to(dtype=target_dtype)
                loaded_my_model_params.add(f"vision_tower.{new_key}")
                used_pi05_params.add(k)
    model.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
    log.info("Loaded vision tower weights")

    # load projector --- "paligemma_with_expert.paligemma.model.multi_modal_projector" -> ""
    # Note: The new projector has concatenated RGB (1152) + Depth (768) = 1920 input dims
    # We only load the first 1152 dims (RGB) from pretrained weights, and zero-init the rest (Depth)
    multi_modal_projector_state_dict = model.multi_modal_projector.state_dict()
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
    model.multi_modal_projector.load_state_dict(
        multi_modal_projector_state_dict, strict=True
    )
    log.info("Loaded multi-modal projector weights (RGB part only, Depth part zero-initialized)")

    # load joint model (both VLM and action mixtures)
    # preserve LoRA weights
    joint_model_state_dict = model.joint_model.state_dict()
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
    model.joint_model.load_state_dict(joint_model_state_dict, strict=False)
    log.info("Loaded joint model weights (VLM mixture + action mixture)")

    # load lm_head if present in our model
    if model.use_lm_head and hasattr(model, 'lm_head'):
        lm_head_state_dict = model.lm_head.state_dict()
        for k, v in tensors.items():
            if k == "paligemma_with_expert.paligemma.lm_head.weight":
                # Note: In PaliGemma, lm_head.weight is tied with embed_tokens.weight
                # We also tie them in our model, so loading lm_head will also update embed_tokens
                # pi05 vocab only contains all the useful tokens, so we need to slice the weights
                lm_head_state_dict["weight"][:v.shape[0]] = v.to(dtype=target_dtype)
                loaded_my_model_params.add("lm_head.weight")
                loaded_my_model_params.add("embed_tokens.weight")  # tied weights
                used_pi05_params.add(k)
        model.lm_head.load_state_dict(lm_head_state_dict, strict=True)
        log.info("Loaded lm_head weights (tied with embed_tokens)")
    else:
        log.warning("lm_head not found or use_lm_head=False, skipping lm_head weight loading")

    # load time embedding MLPs --- "time_mlp_in/out" -> "time_embedding"
    # Pi0.5 uses separate time_mlp_in and time_mlp_out
    # Map to our TimeEncoder: time_embedding = nn.Sequential(SinusoidalPosEmb, TimeEncoder)
    # TimeEncoder has linear_1 and linear_2
    time_embedding_state_dict = model.time_embedding.state_dict()
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
    model.time_embedding.load_state_dict(time_embedding_state_dict, strict=True)
    log.info("Loaded time embedding weights (TimeEncoder)")


# ---------- Weight freezing ---------- #

def freeze_non_lora_weights_in_vlm(vision_tower, multi_modal_projector, joint_model, embed_tokens=None):
    """
    Freeze non-LoRA weights in VLM components while keeping LoRA weights trainable.

    This method freezes:
    - Vision tower weights (except LoRA)
    - Multi-modal projector weights (except LoRA)
    - Language model weights (except LoRA)
    - Token embeddings (if provided)

    Only LoRA parameters remain trainable for efficient fine-tuning.
    """
    for name, param in vision_tower.named_parameters():
        param.requires_grad = True if "lora_" in name else False
    log.info("Froze non-lora weights in vision tower")

    for name, param in multi_modal_projector.named_parameters():
        param.requires_grad = True if "lora_" in name else False
    log.info("Froze non-lora weights in projector")

    for name, param in joint_model.mixtures["vlm"].named_parameters():
        param.requires_grad = True if "lora_" in name else False
    log.info("Froze non-lora weights in lm part of the joint model")

    if embed_tokens is not None:
        for name, param in embed_tokens.named_parameters():
            param.requires_grad = False
        log.info("Froze token embeddings")


def freeze_non_lora_weights_in_ae(action_encoder, action_decoder, joint_model):
    """
    Freeze non-LoRA weights in action expert components while keeping LoRA weights trainable.

    This method freezes:
    - Action encoder weights (except LoRA)
    - Action decoder weights (except LoRA)
    - Action mixture weights (except LoRA)

    Only LoRA parameters remain trainable for efficient fine-tuning.
    """
    for name, param in action_encoder.named_parameters():
        param.requires_grad = True if "lora_" in name else False
    log.info("Froze non-lora weights in action encoder")

    for name, param in action_decoder.named_parameters():
        param.requires_grad = True if "lora_" in name else False
    log.info("Froze non-lora weights in action decoder")

    for name, param in joint_model.mixtures["action"].named_parameters():
        param.requires_grad = True if "lora_" in name else False
    log.info("Froze non-lora weights in action mixture")


def freeze_weights_in_depth(depth_encoder, depth_missing_embeddings):
    """
    Freeze weights in depth encoder and depth missing embeddings.
    """
    if depth_encoder is not None:
        for param in depth_encoder.parameters():
            param.requires_grad = False
    
    if depth_missing_embeddings is not None:
        depth_missing_embeddings.requires_grad = False
    
    log.info("Froze weights in depth encoder and depth missing embeddings")


def freeze_all_weights(model):
    """
    Freeze all trainable parameters in the model.

    Sets requires_grad=False for all parameters, making the model non-trainable.
    Useful for inference-only scenarios.
    """
    for _, param in model.named_parameters():
        param.requires_grad = False


# ---------- KV Cache ---------- #

def build_text_cache():
    """
    Create a new KV cache for text generation.

    Returns:
        KVCache: Empty key-value cache for storing attention states during text generation
    """
    return KVCache()


# ---------- Causal mask and position ID construction ---------- #

def build_causal_mask_and_position_ids(
    attention_mask: torch.Tensor,
    answer_start_idx: torch.Tensor,
    n_actions: torch.Tensor,
    num_action_tokens: int,
    dtype: torch.dtype,
) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
    """
    Build causal attention masks and position IDs for different token types.

    Creates block-diagonal attention patterns:
    - Image/text tokens can attend to themselves
    - Answer only tokens can attend to image/text and the answer tokens before them
    - Action tokens can attend to image/text, and themselves

    Args:
        attention_mask (torch.Tensor): [B, seq_len] Attention mask indicating valid tokens
        answer_start_idx (torch.Tensor): [B] Index of the first answer token
        n_actions (torch.Tensor): [B] Number of action tokens for each sample
        num_action_tokens (int): Maximum number of action tokens
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
    total_num_tokens = max_vlm_tokens + num_action_tokens
    action_start = max_vlm_tokens
    vlm_token_cnts = torch.sum(attention_mask, dim=1, dtype=torch.long)

    q_idx = torch.arange(total_num_tokens, device=device).view(1, total_num_tokens, 1)
    k_idx = torch.arange(total_num_tokens, device=device).view(1, 1, total_num_tokens)
    cnt = vlm_token_cnts.view(bsz, 1, 1)
    answer_start = answer_start_idx.view(bsz, 1, 1)
    num_valid_actions = n_actions.view(bsz, 1, 1)

    vlm_queries = q_idx < cnt
    image_text_keys = k_idx < answer_start
    answer_queries = (q_idx >= answer_start) & (q_idx < cnt)
    answer_keys = (k_idx >= answer_start) & (k_idx < cnt)
    action_queries = (q_idx >= action_start) & (q_idx < action_start + num_valid_actions)
    action_keys = (k_idx >= action_start) & (k_idx < action_start + num_valid_actions)

    allow_image_text = vlm_queries & image_text_keys
    allow_answer = answer_queries & answer_keys & (k_idx <= q_idx)
    allow_action = action_queries & (image_text_keys | action_keys)
    allow_mask = allow_image_text | allow_answer | allow_action

    causal_mask = torch.full(
        (bsz, total_num_tokens, total_num_tokens),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    causal_mask = torch.where(allow_mask, torch.zeros(1, dtype=dtype, device=device), causal_mask)
    causal_mask = causal_mask.unsqueeze(1)

    vlm_position_ids = torch.arange(1, max_vlm_tokens + 1, device=device).repeat(bsz, 1)
    action_position_ids = (
        torch.arange(0, num_action_tokens, device=device).unsqueeze(0)
        + answer_start_idx.unsqueeze(1)
        + 1
    )
    return causal_mask, vlm_position_ids, action_position_ids


def split_full_mask_into_submasks(
    causal_mask: torch.FloatTensor, max_vlm_tokens: int, num_action_tokens: int,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Split the full causal mask into separate masks for different model components.

    Args:
        causal_mask (torch.FloatTensor): [B, 1, total_len, total_len]
          Full causal attention mask (broadcasts to all heads)
        max_vlm_tokens (int): Number of VLM tokens
        num_action_tokens (int): Number of action tokens

    Returns:
        Tuple containing:
            - vlm_mask (torch.FloatTensor): [B, 1, seq_len, seq_len]
              Attention mask for image/text tokens (broadcasts to all heads)
            - action_mask (torch.FloatTensor): [B, 1, action_len, total_len]
              Attention mask for action tokens (broadcasts to all heads)
    """
    vlm_mask = causal_mask[..., : max_vlm_tokens, : max_vlm_tokens]
    action_mask = causal_mask[..., -num_action_tokens :, :]
    return vlm_mask, action_mask


def build_causal_mask_and_position_ids_for_text(
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
        dtype (torch.dtype): Data type for the causal mask

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
