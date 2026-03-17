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
    if t.ndim == 1:
        t = t[:, None, None]
    elif t.ndim == 2:
        t = t[:, :, None]
    else:
        raise ValueError(f"Unsupported time shape for psi_t: {t.shape}")

    return (1 - (1 - flow_sig_min) * t) * x + t * x1

def _sample_rtc_delay(
    valid_action_len: torch.LongTensor,
    strategy: str = "uniform",
    max_delay: int | None = None,
    forced_delay: torch.LongTensor | None = None,
) -> torch.LongTensor:
    """
    Sample one RTC prefix delay per batch element.

    The sampled delay determines how many leading action tokens are treated as
    known action-prefix conditions during RTC flow training.

    Args:
        valid_action_len:
            [B] Number of valid action steps for each sample. Values are expected
            to be in the range [0, horizon_steps].
        strategy:
            Delay sampling strategy. `uniform` samples all valid delays with
            equal probability. `exp` matches the Kinetix RTC
            implementation and biases toward smaller delays via
            `exp(arange(upper)[::-1])`.
        max_delay:
            Optional upper bound for the sampled delay. When provided, the actual
            sampling upper bound becomes min(valid_action_len, max_delay) for each
            sample.
        forced_delay:
            Optional [B] tensor used to bypass random sampling. This is intended
            for deterministic tests and debugging.

    Returns:
        torch.LongTensor:
            [B] Sampled prefix delays. Each entry is in the range
            [0, min(valid_action_len_i, max_delay)) when the upper bound is
            positive, or 0 when the sample has no valid action tokens.
    """
    if forced_delay is not None:
        return forced_delay.to(device=valid_action_len.device, dtype=torch.long)

    delay_upper = valid_action_len.clamp(min=0)
    if max_delay is not None:
        delay_upper = torch.minimum(
            delay_upper,
            torch.full_like(delay_upper, max_delay),
        )
    delay_upper = delay_upper.clamp(min=0)
    strategy = str(strategy).lower()

    if strategy == "uniform":
        random_delay = torch.rand(delay_upper.shape, device=valid_action_len.device, dtype=torch.float32)
        return torch.floor(random_delay * delay_upper.to(torch.float32)).to(dtype=torch.long)

    if strategy == "exp":
        max_upper = delay_upper.max().item()
        if max_upper <= 0:
            return torch.zeros_like(delay_upper, device=valid_action_len.device, dtype=torch.long)
        w = torch.exp(torch.arange(max_upper - 1, -1, -1, device=valid_action_len.device, dtype=torch.float32))
        w = w / w.sum()
        sampled = torch.multinomial(w.unsqueeze(0).expand(delay_upper.shape[0], -1), num_samples=1).squeeze(1)
        sampled = sampled % delay_upper.clamp(min=1)
        return sampled.to(dtype=torch.long)


    raise ValueError(f"Unsupported RTC delay strategy: {strategy}")

def _build_rtc_flow_inputs(
    *,
    actions: torch.FloatTensor,
    actions_valid_mask: torch.Tensor,
    postfix_time: torch.FloatTensor,
    n_actions: torch.LongTensor | None,
    rtc_delay_strategy: str = "uniform",
    rtc_max_delay: int | None = None,
    forced_delay: torch.LongTensor | None = None,
) -> tuple[torch.FloatTensor, torch.BoolTensor, torch.FloatTensor, torch.LongTensor]:
    """
    Build token-wise RTC conditioning inputs for flow-matching training.

    This helper converts a per-sample postfix diffusion time into token-wise
    times by assigning:
    - time = 1.0 for prefix tokens that are treated as known action conditions
    - time = postfix_time for postfix tokens that remain denoising targets

    It also constructs the postfix validity mask used to restrict the flow loss
    to valid non-prefix action tokens.

    Args:
        actions:
            [B, H, D] Action targets, where B is batch size, H is action horizon,
            and D is action dimension.
        actions_valid_mask:
            [B, H, D] or broadcast-compatible mask indicating which action values
            are valid for loss computation. Padding positions must be False.
        postfix_time:
            [B] Per-sample flow time used for RTC postfix tokens.
        n_actions:
            Optional [B] tensor containing the number of valid action steps for
            each sample. If None, valid lengths are inferred from
            actions_valid_mask.
        rtc_max_delay:
            Optional cap on the sampled prefix length.
        forced_delay:
            Optional [B] deterministic delay tensor for tests/debugging.

    Returns:
        tuple containing:
            - token_t (torch.FloatTensor):
              [B, H] Token-wise flow times, equal to 1.0 on the prefix and
              postfix_time on the postfix.
            - postfix_valid_mask (torch.BoolTensor):
              [B, H, D]-compatible boolean mask that is True only on postfix
              positions that are also valid action targets.
            - prefix_mask (torch.BoolTensor):
              [B, H] Boolean mask marking the RTC action prefix.
            - delay (torch.LongTensor):
              [B] Sampled prefix lengths.
    """
    batch_size, horizon_steps = actions.shape[:2]
    device = actions.device

    if n_actions is not None:
        valid_action_len = n_actions.to(device=device, dtype=torch.long)
    else:
        valid_action_len = actions_valid_mask.reshape(batch_size, horizon_steps, -1).any(dim=-1).sum(dim=1)
    valid_action_len = valid_action_len.clamp(min=0, max=horizon_steps)

    delay = _sample_rtc_delay(
        valid_action_len,
        strategy=rtc_delay_strategy,
        max_delay=rtc_max_delay,
        forced_delay=forced_delay,
    )
    positions = torch.arange(horizon_steps, device=device).unsqueeze(0)
    prefix_mask = positions < delay.unsqueeze(1)
    token_t = torch.where(prefix_mask, torch.ones_like(postfix_time[:, None]), postfix_time[:, None])
    postfix_mask = (~prefix_mask).unsqueeze(-1)
    postfix_valid_mask = postfix_mask & actions_valid_mask.to(dtype=torch.bool)
    return token_t, postfix_valid_mask, prefix_mask, delay

def _compute_flow_loss(
    *,
    model,
    actions: torch.FloatTensor,
    actions_valid_mask: torch.Tensor,
    pred_v_t: torch.FloatTensor,
    noise: torch.FloatTensor,
    rtc_mask: torch.Tensor | None = None,
) -> torch.FloatTensor:
    """
    Compute the masked flow-matching regression loss.

    This helper is shared by the legacy flow-training path and the RTC path.
    When rtc_mask is None, the loss is computed on all valid action positions.
    When rtc_mask is provided, the loss is restricted to RTC postfix positions.

    Args:
        model:
            LegendVLA model instance. Only model.flow_sig_min is used here.
        actions:
            [B, H, D] Ground-truth action tensor.
        actions_valid_mask:
            [B, H, D] or broadcast-compatible validity mask for the legacy
            training path.
        pred_v_t:
            [B, H, D] Predicted flow velocity from the action decoder.
        noise:
            [B, H, D] Gaussian noise used to construct the noisy action input.
        rtc_mask:
            Optional [B, H, D] or broadcast-compatible boolean mask selecting the
            RTC postfix region to supervise.

    Returns:
        torch.FloatTensor:
            Scalar normalized flow loss.
    """
    target_v_t = actions - (1 - model.flow_sig_min) * noise
    flow_loss = (pred_v_t - target_v_t) ** 2

    # Apply per-dimension weighting
    if hasattr(model, 'action_dim_weights'):
        flow_loss = flow_loss * model.action_dim_weights

    loss_mask = rtc_mask if rtc_mask is not None else actions_valid_mask
    masked_loss = flow_loss * loss_mask.to(dtype=flow_loss.dtype)
    valid_count = loss_mask.to(dtype=flow_loss.dtype).sum()
    return torch.sum(masked_loss) / valid_count.clamp(min=1)

def _build_dense_diffloss_inputs(
    model,
    hidden_states: torch.FloatTensor,
    actions: torch.FloatTensor,
    answer_start_idx: torch.LongTensor,
    n_actions: torch.LongTensor,
    vla_sample_mask: torch.BoolTensor,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
    vla_hidden = hidden_states[vla_sample_mask]
    vla_action = actions[vla_sample_mask]
    vla_answer_start_idx = answer_start_idx[vla_sample_mask]
    vla_n_actions = n_actions[vla_sample_mask]

    ar_action_chunk_size = model.ar_action_chunk_size
    ar_action_chunks = vla_action.unfold(dimension=1, size=ar_action_chunk_size, step=1)
    ar_action_chunks_flat = ar_action_chunks.flatten(start_dim=2)

    max_chunk_count = ar_action_chunks_flat.shape[1]
    chunk_indices = torch.arange(max_chunk_count, device=hidden_states.device).unsqueeze(0)
    hidden_positions = vla_answer_start_idx.unsqueeze(1) - 1 + chunk_indices
    hidden_positions = hidden_positions.clamp(min=0, max=hidden_states.shape[1] - 1)

    gather_index = hidden_positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    vla_hidden_z = torch.gather(vla_hidden, dim=1, index=gather_index)

    valid_chunk_counts = (
        vla_n_actions - ar_action_chunk_size + 1
    ).clamp(min=0, max=max_chunk_count).unsqueeze(1)
    diffloss_mask = chunk_indices < valid_chunk_counts

    return (
        vla_hidden_z.reshape(-1, vla_hidden_z.shape[-1]),
        ar_action_chunks_flat.reshape(-1, ar_action_chunks_flat.shape[-1]),
        diffloss_mask.reshape(-1),
    )


# TODO: Deprecated method, to be updated
def compute_ar_loss(model, batch: dict, return_attn_weights: bool = False) -> dict:
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
        return_attn_weights=return_attn_weights,
    )
    hidden_states = output["vlm"]

    ce_loss = compute_celoss(
        model.lm_head, model.final_logit_softcapping,
        model.CELoss, model.ignore_index, hidden_states, labels
    )
    return {"ce_loss": ce_loss}


# TODO: Deprecated method, to be updated
def compute_flow_loss(model, batch: dict, return_attn_weights: bool = False) -> dict:
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
    use_rtc = model.use_rtc
    rtc_mask = None
    # noisy action
    x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
    x1 = actions
    time_for_model = t
    if use_rtc:
        time_for_model, rtc_mask, prefix_mask, _ = _build_rtc_flow_inputs(
            actions=x1,
            actions_valid_mask=actions_valid_mask,
            postfix_time=t,
            n_actions=batch.get("n_actions"),
            rtc_delay_strategy=model.rtc_delay_strategy,
            rtc_max_delay=model.rtc_max_delay,
        )
    psi_t_val = psi_t(x0, x1, time_for_model, model.flow_sig_min)
    if use_rtc:
        psi_t_val = torch.where(prefix_mask.unsqueeze(-1), x1, psi_t_val)

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

    if time_for_model.ndim == 2 and use_rtc:
        time_cond = model.time_embedding(time_for_model.reshape(-1)).reshape(
            time_for_model.shape[0], time_for_model.shape[1], -1
        )
    else:
        time_cond = model.time_embedding(time_for_model)

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
        return_attn_weights=return_attn_weights,
    )["action"]

    v_psi = model.action_decoder(action_embeds)
    flow_loss = _compute_flow_loss(
        model=model,
        actions=x1,
        actions_valid_mask=actions_valid_mask,
        pred_v_t=v_psi,
        noise=x0,
        rtc_mask=rtc_mask,
    )
    return {"flow_loss": flow_loss}


def compute_loss(model, batch: dict, return_attn_weights: bool = False) -> dict:
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
    is_vla_data = batch["is_vla_data"].to(dtype=torch.bool)
    n_actions = batch["n_actions"]
    n_states = batch["n_states"]
    use_rtc = model.use_rtc
    # noisy action
    x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
    x1 = actions
    rtc_mask = None
    time_for_model = t
    if use_rtc:
        time_for_model, rtc_mask, prefix_mask, _ = _build_rtc_flow_inputs(
            actions=x1,
            actions_valid_mask=actions_valid_mask,
            postfix_time=t,
            n_actions=n_actions,
            rtc_delay_strategy=model.rtc_delay_strategy,
            rtc_max_delay=model.rtc_max_delay,
        )
    psi_t_val = psi_t(x0, x1, time_for_model, model.flow_sig_min)
    if use_rtc:
        psi_t_val = torch.where(prefix_mask.unsqueeze(-1), x1, psi_t_val)
        
    depth_values = batch.get("depth_values")
    has_depth_values = batch.get("has_depth_values")
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
    if time_for_model.ndim == 2 and use_rtc:
        time_cond = model.time_embedding(time_for_model.reshape(-1)).reshape(
            time_for_model.shape[0], time_for_model.shape[1], -1
        )
    else:
        time_cond = model.time_embedding(time_for_model)
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
        return_attn_weights=return_attn_weights,
    )
    hidden_states = output["vlm"]
    action_embeds = output["action"]

    ce_loss = compute_celoss(
        model.lm_head, model.final_logit_softcapping,
        model.CELoss, model.ignore_index,
        hidden_states[~is_vla_data],
        labels[~is_vla_data]
    )

    # diffusion loss
    vla_hidden_z, action_gt, diffloss_mask = _build_dense_diffloss_inputs(
        model,
        hidden_states,
        actions,
        answer_start_idx,
        n_actions,
        is_vla_data,
    )
    vla_hidden_z_repeated = vla_hidden_z.repeat_interleave(model.diffloss_micro_batch_size, dim=0)
    action_gt_repeated = action_gt.repeat_interleave(model.diffloss_micro_batch_size, dim=0)
    diffloss_mask_repeated = diffloss_mask.repeat_interleave(model.diffloss_micro_batch_size, dim=0)
    latent_condition_embeds = model.latent_condition_projector(vla_hidden_z_repeated)
    diff_loss = model.diffloss(
        action_gt_repeated,
        latent_condition_embeds,
        mask=diffloss_mask_repeated.to(dtype=action_gt_repeated.dtype),
    ) / model.diffloss_micro_batch_size

    # flow loss
    v_psi = model.action_decoder(action_embeds)
    flow_loss = _compute_flow_loss(
        model=model,
        actions=x1,
        actions_valid_mask=actions_valid_mask,
        pred_v_t=v_psi,
        noise=x0,
        rtc_mask=rtc_mask,
    )
    total_loss = (model.loss_weights.ce_loss_weight * ce_loss
                  + model.loss_weights.diffusion_loss_weight * diff_loss
                  + model.loss_weights.flow_loss_weight * flow_loss)
    return {
        "total_loss": total_loss,
        "ce_loss": ce_loss,
        "diffusion_loss": diff_loss,
        "flow_loss": flow_loss,
    }
