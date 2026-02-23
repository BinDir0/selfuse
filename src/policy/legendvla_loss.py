"""
Loss computation functions extracted from LegendVLA.

Contains:
- compute_celoss: Cross-entropy loss for language modeling
- compute_ar_loss: Autoregressive loss (deprecated, forward + CE)
- compute_flow_loss: Flow matching loss (deprecated, forward + flow)
- compute_loss: Combined VLA loss (CE + flow + diffusion)
"""

import torch
from torch import nn


def _apply_final_logit_softcapping(logits: torch.Tensor, final_logit_softcapping: float = None) -> torch.Tensor:
    """
    Apply final logit softcapping (Gemma2 feature). Pure function.

    Args:
        logits: Raw logits from language model head
        final_logit_softcapping: Softcapping value. None means disabled (Gemma1).

    Returns:
        Softcapped logits (same shape as input)
    """
    if final_logit_softcapping is not None:
        logits = logits / final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * final_logit_softcapping
    return logits


@torch.compile
def compute_celoss(
    lm_head: nn.Module,
    final_logit_softcapping: float,
    ce_loss_fn: nn.Module,
    ignore_index: int,
    hidden_states: torch.FloatTensor,
    labels: torch.LongTensor,
) -> torch.FloatTensor:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        lm_head: Language model head module
        final_logit_softcapping: Softcapping value (float or None)
        ce_loss_fn: Cross-entropy loss function (sum reduction)
        ignore_index: Label index to ignore in loss computation
        hidden_states: [B, seq_len, hidden_size] Hidden states from the language model
        labels: [B, seq_len] Labels for language modeling loss

    Returns:
        torch.FloatTensor: Normalized cross-entropy loss
    """
    logits = lm_head(hidden_states)
    logits = _apply_final_logit_softcapping(logits, final_logit_softcapping)
    logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    labels = labels[:, 1:].contiguous().view(-1)
    ce_loss = ce_loss_fn(logits, labels)
    valid_num_labels = torch.sum(labels != ignore_index)
    return ce_loss / valid_num_labels.clamp(min=1)


def psi_t(
    x: torch.FloatTensor,
    x1: torch.FloatTensor,
    t: torch.FloatTensor,
    flow_sig_min: float,
) -> torch.FloatTensor:
    """
    Conditional flow function for flow matching.

    Interpolates between noise x and target x1 based on time t.

    Args:
        x: [B, horizon_steps, action_dim] Initial noise
        x1: [B, horizon_steps, action_dim] Target action
        t: [B] Time parameter (0 to 1)
        flow_sig_min: Minimum sigma for flow matching

    Returns:
        torch.FloatTensor: [B, horizon_steps, action_dim] Interpolated action at time t
    """
    t = t[:, None, None]  # (B, 1, 1)
    return (1 - (1 - flow_sig_min) * t) * x + t * x1


# TODO: Deprecated method, to be updated
def compute_ar_loss(model, batch: dict) -> dict:
    """
    Compute autoregressive loss for action prediction and vision language understanding.

    Args:
        model: LegendVLA model instance
        batch: Input dictionary containing input_ids, labels, pixel_values,
               causal_mask, vlm_position_ids, states, actions, n_states, n_actions, is_vla_data

    Returns:
        dict: {"ce_loss": cross-entropy loss}
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    pixel_values = batch["pixel_values"]
    causal_mask = batch["causal_mask"]
    vlm_position_ids = batch["vlm_position_ids"]

    if 'depth_values' in batch:
        depth_values = batch["depth_values"]
        has_depth_values = batch["has_depth_values"]
    else:
        depth_values = None
        has_depth_values = None
    inputs_embeds = model._forward_siglip_and_text_embedding(
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

    output = model.joint_model(
        attention_mask=causal_mask,
        position_ids_all={"vlm": vlm_position_ids},
        embeds_all={"vlm": inputs_embeds},
        kv_caches={},
        final_layer_post_attn_skip_names=[],
    )
    hidden_states = output["vlm"]

    ce_loss = compute_celoss(
        model.lm_head, model.final_logit_softcapping,
        model.CELoss, model.ignore_index, hidden_states, labels
    )
    return {"ce_loss": ce_loss}


# TODO: Deprecated method, to be updated
def compute_flow_loss(model, batch: dict) -> dict:
    """
    Forward pass for flow matching training.

    Args:
        model: LegendVLA model instance
        batch: Input dictionary containing input_ids, pixel_values, causal_mask,
               vlm_position_ids, action_position_ids, actions, actions_valid_mask, t,
               states, n_states, n_actions, is_vla_data

    Returns:
        dict: {"flow_loss": flow matching loss}
    """
    input_ids = batch["input_ids"]
    pixel_values = batch["pixel_values"]
    causal_mask = batch["causal_mask"]
    vlm_position_ids = batch["vlm_position_ids"]
    action_position_ids = batch["action_position_ids"]
    actions = batch["actions"]
    actions_valid_mask = batch["actions_valid_mask"]
    t = batch["t"]

    # noisy action
    x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
    x1 = actions
    psi_t_val = psi_t(x0, x1, t, model.flow_sig_min)

    if 'depth_values' in batch:
        depth_values = batch["depth_values"]
        has_depth_values = batch["has_depth_values"]
    else:
        depth_values = None
        has_depth_values = None
    inputs_embeds = model._forward_siglip_and_text_embedding(
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

    time_cond = model.time_embedding(t)
    if model.action_expert_adaptive_mode:
        action_embeds = model.action_encoder(psi_t_val)
    else:
        action_embeds = model.action_encoder(psi_t_val, time_cond)
    action_embeds = action_embeds / (model.action_hidden_size**0.5)
    action_embeds = model.joint_model(
        attention_mask=causal_mask,
        position_ids_all={"vlm": vlm_position_ids, "action": action_position_ids},
        embeds_all={"vlm": inputs_embeds, "action": action_embeds},
        time_cond=time_cond,
        kv_caches={},
    )["action"]

    v_psi = model.action_decoder(action_embeds)
    d_psi = x1 - (1 - model.flow_sig_min) * x0

    loss = (v_psi - d_psi) ** 2
    masked_loss = actions_valid_mask * loss
    actions_valid_num = torch.sum(actions_valid_mask)
    flow_loss = torch.sum(masked_loss) / actions_valid_num.clamp(min=1)
    return {"flow_loss": flow_loss}


def compute_loss(model, batch: dict) -> dict:
    """
    Compute combined VLA loss: cross-entropy (VLM) + flow matching + diffusion loss.

    Args:
        model: LegendVLA model instance
        batch: Input dictionary containing input_ids, labels, pixel_values, causal_mask,
               vlm_position_ids, action_position_ids, actions, actions_valid_mask, t,
               states, answer_start_idx, is_vla_data, n_actions, n_states

    Returns:
        dict: {"total_loss", "ce_loss", "diffusion_loss", "flow_loss"}
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    pixel_values = batch["pixel_values"]
    causal_mask = batch["causal_mask"]
    vlm_position_ids = batch["vlm_position_ids"]
    action_position_ids = batch["action_position_ids"]
    actions = batch["actions"]
    actions_valid_mask = batch["actions_valid_mask"]
    t = batch["t"]
    states = batch["states"]
    answer_start_idx = batch["answer_start_idx"]
    is_vla_data = batch["is_vla_data"]
    is_vlm_data = (is_vla_data != True)
    n_actions = batch["n_actions"]
    n_states = batch["n_states"]

    # noisy action
    x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
    x1 = actions
    psi_t_val = psi_t(x0, x1, t, model.flow_sig_min)

    if 'depth_values' in batch:
        depth_values = batch["depth_values"]
        has_depth_values = batch["has_depth_values"]
    else:
        depth_values = None
        has_depth_values = None
    inputs_embeds = model._forward_siglip_and_text_embedding(
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
    time_cond = model.time_embedding(t)
    if model.action_expert_adaptive_mode:
        action_embeds = model.action_encoder(psi_t_val)
    else:
        action_embeds = model.action_encoder(psi_t_val, time_cond)
    action_embeds = action_embeds / (model.action_hidden_size**0.5)
    output = model.joint_model(
        attention_mask=causal_mask,
        position_ids_all={"vlm": vlm_position_ids, "action": action_position_ids},
        embeds_all={"vlm": inputs_embeds, "action": action_embeds},
        time_cond=time_cond,
        kv_caches={},
        final_layer_post_attn_skip_names=[],
    )
    hidden_states = output["vlm"]
    action_embeds = output["action"]

    if torch.any(is_vlm_data):
        ce_loss = compute_celoss(
            model.lm_head, model.final_logit_softcapping,
            model.CELoss, model.ignore_index,
            hidden_states[is_vlm_data], labels[is_vlm_data]
        )
    else:
        ce_loss = compute_celoss(
            model.lm_head, model.final_logit_softcapping,
            model.CELoss, model.ignore_index,
            hidden_states, labels
        )

    # diffusion loss
    device = hidden_states.device
    max_vlm_tokens = hidden_states.shape[1]
    num_action_tokens = model.num_action_tokens
    ar_action_chunk_size = model.ar_action_chunk_size
    vla_hidden = hidden_states[is_vla_data]
    vla_action = actions[is_vla_data]  # (B, H, D)
    ar_action_chunks = vla_action.unfold(dimension=1, size=ar_action_chunk_size, step=1)  # (B, H-chunk_size+1, D, chunk_size)
    ar_action_chunks_flat = ar_action_chunks.flatten(start_dim=2)  # (B, H-chunk_size+1, D*chunk_size)

    # 构建索引序列
    range_hidden = torch.arange(max_vlm_tokens, device=device).unsqueeze(0)
    range_action = torch.arange(num_action_tokens - ar_action_chunk_size + 1, device=device).unsqueeze(0)

    starts = answer_start_idx[is_vla_data].unsqueeze(1)
    ends = (answer_start_idx[is_vla_data] + n_actions[is_vla_data]).unsqueeze(1) - ar_action_chunk_size + 1
    action_ends = n_actions[is_vla_data].unsqueeze(1) - ar_action_chunk_size + 1

    mask_hidden = (range_hidden >= (starts - 1)) & (range_hidden < (ends - 1))
    mask_action = range_action < action_ends

    vla_hidden_z = vla_hidden[mask_hidden]
    action_gt = ar_action_chunks_flat[mask_action]
    assert vla_hidden_z.shape[0] == action_gt.shape[0], \
        f"The number of vla hidden and action gt should be the same, but got {vla_hidden_z.shape[0]} and {action_gt.shape[0]}"
    if action_gt.numel() == 0:
        dummy_vla_hidden_z = hidden_states[0:1, 0, :]
        dummy_action_gt = action_gt.new_zeros(1, action_gt.shape[-1])
        dummy_latent_condition_embeds = model.latent_condition_projector(dummy_vla_hidden_z)
        dummy_diff_loss = model.diffloss(dummy_action_gt, dummy_latent_condition_embeds)
        diff_loss = dummy_diff_loss * 0
    else:
        vla_hidden_z_repeated = vla_hidden_z.repeat_interleave(model.diffloss_micro_batch_size, dim=0)
        action_gt_repeated = action_gt.repeat_interleave(model.diffloss_micro_batch_size, dim=0)
        latent_condition_embeds = model.latent_condition_projector(vla_hidden_z_repeated)
        diff_loss = model.diffloss(action_gt_repeated, latent_condition_embeds) / model.diffloss_micro_batch_size

    # flow loss
    v_psi = model.action_decoder(action_embeds)
    d_psi = x1 - (1 - model.flow_sig_min) * x0
    flow_loss = (v_psi - d_psi) ** 2
    masked_loss = actions_valid_mask * flow_loss
    actions_valid_num = torch.sum(actions_valid_mask)
    flow_loss = torch.sum(masked_loss) / actions_valid_num.clamp(min=1)

    total_loss = (model.loss_weights.ce_loss_weight * ce_loss
                  + model.loss_weights.diffusion_loss_weight * diff_loss
                  + model.loss_weights.flow_loss_weight * flow_loss)
    return {
        "total_loss": total_loss,
        "ce_loss": ce_loss,
        "diffusion_loss": diff_loss,
        "flow_loss": flow_loss,
    }
