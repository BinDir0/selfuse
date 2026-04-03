from __future__ import annotations

from typing import Any

import torch

from src.utils.generation_utils import sample_token


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _build_action_valid_mask(batch: dict[str, Any], action_dim: int) -> torch.Tensor:
    if "actions_valid_mask" in batch:
        return batch["actions_valid_mask"].to(dtype=torch.bool)

    input_ids = batch["input_ids"]
    action_token_id = batch.get("action_token_id")
    if action_token_id is None:
        raise ValueError("Action inference requires either `actions_valid_mask` or `action_token_id`.")
    action_counts = (input_ids == int(action_token_id)).sum(dim=1)

    if "n_actions" in batch and batch["n_actions"] is not None:
        n_actions = batch["n_actions"].to(device=input_ids.device, dtype=torch.long)
    else:
        n_actions = action_counts.to(device=input_ids.device, dtype=torch.long)

    max_actions = int(n_actions.max().item()) if n_actions.numel() > 0 else 0
    step_mask = torch.arange(max_actions, device=input_ids.device).unsqueeze(0) < n_actions.unsqueeze(1)
    return step_mask.unsqueeze(-1).expand(-1, -1, action_dim)



def _last_valid_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attention_mask.shape[1]
    positions = torch.arange(seq_len, device=attention_mask.device, dtype=torch.long).unsqueeze(0)
    masked_positions = positions.masked_fill(attention_mask == 0, -1)
    last_indices = masked_positions.max(dim=1).values
    return last_indices.clamp(min=0)


def prepare_prefix_memory(model, batch: dict, output_attentions: bool = False):
    prefix_batch = _clone_batch(batch)
    prefix_batch.pop("actions", None)
    prefix_batch.pop("actions_valid_mask", None)
    slot_embeds = model.build_slot_embeddings(prefix_batch, add_action_noise=False)
    return model.forward_backbone_stream(prefix_batch, slot_embeds, output_attentions=output_attentions)


def infer_flow_action(
    model, batch: dict, prev_action_chunk=None, inference_delay: int = 0,
    output_attentions: bool = False,
):
    working_batch = _clone_batch(batch)
    working_batch["action_token_id"] = model.action_token_index

    action_valid_mask = _build_action_valid_mask(working_batch, model.action_dim)
    batch_size, action_len, _ = action_valid_mask.shape
    device = working_batch["input_ids"].device
    action_dtype = working_batch.get("states", torch.empty((), device=device, dtype=torch.float32)).dtype
    if not torch.is_floating_point(torch.empty((), dtype=action_dtype)):
        action_dtype = torch.float32

    generated_actions = torch.randn(batch_size, action_len, model.action_dim, device=device, dtype=action_dtype)
    action_step_mask = action_valid_mask.any(dim=-1)

    if prev_action_chunk is not None:
        prev_action_chunk = prev_action_chunk.to(device=device, dtype=generated_actions.dtype)
    use_rtc = prev_action_chunk is not None and inference_delay > 0
    prefix_token_mask = torch.zeros(batch_size, action_len, dtype=torch.bool, device=device)
    if use_rtc:
        pinned_steps = min(inference_delay, action_len, prev_action_chunk.shape[1])
        prefix_token_mask[:, :pinned_steps] = action_step_mask[:, :pinned_steps]

    working_batch["actions_valid_mask"] = action_valid_mask
    working_batch["n_actions"] = action_step_mask.sum(dim=1).to(dtype=torch.long)

    backbone_output = prepare_prefix_memory(model, working_batch, output_attentions=output_attentions)
    vlm_attn_weights = backbone_output.attention_weights if output_attentions else None

    delta_t = 1.0 / max(model.num_inference_steps, 1)
    t = torch.zeros(batch_size, device=device, dtype=generated_actions.dtype)
    expert_attn_per_step: list | None = [] if output_attentions else None

    for _ in range(model.num_inference_steps):
        if use_rtc:
            generated_actions = torch.where(
                prefix_token_mask.unsqueeze(-1),
                prev_action_chunk[:, :action_len],
                generated_actions,
            )
            time_for_model = torch.where(
                prefix_token_mask,
                torch.ones(batch_size, action_len, device=device, dtype=generated_actions.dtype),
                t[:, None].expand(-1, action_len),
            )
        else:
            time_for_model = t[:, None].expand(-1, action_len)

        flow_inputs = {
            "noisy_actions": generated_actions,
            "time_for_model": time_for_model,
        }
        flow_output = model.forward_flow_stream(
            batch=working_batch,
            backbone_output=backbone_output,
            flow_inputs=flow_inputs,
            num_parallel_chunks=1,
            output_attentions=output_attentions,
        )
        generated_actions = generated_actions + delta_t * flow_output["pred_v"]
        t = (t + delta_t).clamp(max=1.0)
        if output_attentions:
            expert_attn_per_step.append(flow_output["expert_attention_weights"])

    if use_rtc:
        generated_actions = torch.where(
            prefix_token_mask.unsqueeze(-1),
            prev_action_chunk[:, :action_len],
            generated_actions,
        )

    result = generated_actions * action_step_mask.unsqueeze(-1).to(dtype=generated_actions.dtype)
    if output_attentions:
        return {
            "generated_actions": result,
            "vlm_attention_weights": vlm_attn_weights,
            "expert_attention_weights": expert_attn_per_step,
        }
    return result


def infer_ar_action(
    model,
    batch: dict,
    max_new_tokens: int | None = None,
    temperature: float = 1.0,
    cfg: float = 1.0,
):
    if model.diffloss is None:
        raise ValueError("Autoregressive VLA inference requires `model.diffloss`.")

    working_batch = _clone_batch(batch)
    working_batch["action_token_id"] = model.action_token_index
    action_valid_mask = _build_action_valid_mask(working_batch, model.action_dim)
    action_step_mask = action_valid_mask.any(dim=-1)
    batch_size, action_len, _ = action_valid_mask.shape
    if max_new_tokens is not None:
        action_len = min(action_len, int(max_new_tokens))
        action_valid_mask = action_valid_mask[:, :action_len]
        action_step_mask = action_step_mask[:, :action_len]

    device = working_batch["input_ids"].device
    action_dtype = working_batch.get("states", torch.empty((), device=device, dtype=torch.float32)).dtype
    if not torch.is_floating_point(torch.empty((), dtype=action_dtype)):
        action_dtype = torch.float32

    generated_actions = torch.zeros(batch_size, action_len, model.action_dim, device=device, dtype=action_dtype)
    generated_hidden = []

    working_batch["n_actions"] = action_step_mask.sum(dim=1).to(dtype=torch.long)

    for step_idx in range(action_len):
        step_batch = _clone_batch(working_batch)
        step_batch["actions"] = generated_actions
        slot_embeds = model.build_slot_embeddings(step_batch, add_action_noise=False)
        backbone_output = model.forward_backbone_stream(step_batch, slot_embeds)
        action_hidden = _gather_action_hidden_states(
            hidden_states=backbone_output.last_hidden_states,
            answer_start_idx=step_batch["answer_start_idx"],
            action_len=action_len,
        )
        current_hidden = action_hidden[:, step_idx, :]
        generated_hidden.append(current_hidden)

        latent_condition = model.latent_condition_projector(current_hidden)
        next_action = model.diffloss.sample(latent_condition, temperature=temperature, cfg=cfg)
        if next_action.ndim == 1:
            next_action = next_action.unsqueeze(0)
        if next_action.ndim == 3:
            next_action = next_action[:, 0, :]
        elif next_action.shape[-1] == model.action_dim * model.ar_action_train_config.chunk_size:
            next_action = next_action.view(next_action.shape[0], model.ar_action_train_config.chunk_size, model.action_dim)[:, 0, :]
        elif next_action.shape[-1] != model.action_dim:
            raise ValueError(
                "DiffLoss AR inference must return either action_dim or "
                "action_dim * ar_action_chunk_size channels. "
                f"Got {next_action.shape[-1]} and expected {model.action_dim} or "
                f"{model.action_dim * model.ar_action_train_config.chunk_size}."
            )
        valid_step = action_step_mask[:, step_idx].unsqueeze(-1)
        generated_actions[:, step_idx] = torch.where(
            valid_step,
            next_action.to(dtype=generated_actions.dtype),
            generated_actions[:, step_idx],
        )

    hidden_tensor = torch.stack(generated_hidden, dim=1) if generated_hidden else None
    return {
        "generated_actions": generated_actions,
        "generated_hidden_states": hidden_tensor,
    }


def infer_vlm_generation(
    model,
    batch: dict,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 10,
    top_p: float = 1.0,
    allowed_token_ids=None,
    eos_token_id: int | None = None,
):
    working_batch = _clone_batch(batch)
    input_ids = working_batch["input_ids"]
    attention_mask = working_batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    generated_ids = []
    eos_token_id = model.eos_token_id if eos_token_id is None else eos_token_id

    for _ in range(max_new_tokens):
        step_batch = _clone_batch(working_batch)
        step_batch["input_ids"] = input_ids
        step_batch["attention_mask"] = attention_mask
        slot_embeds = model.build_slot_embeddings(step_batch, add_action_noise=False)
        backbone_output = model.forward_backbone_stream(step_batch, slot_embeds)

        last_indices = _last_valid_indices(attention_mask)
        last_hidden = backbone_output.last_hidden_states[
            torch.arange(input_ids.shape[0], device=input_ids.device),
            last_indices,
        ]
        logits = model.lm_head(last_hidden)
        next_ids = sample_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            allowed_token_ids=allowed_token_ids,
        )
        generated_ids.append(next_ids.unsqueeze(1))

        input_ids = torch.cat([input_ids, next_ids.unsqueeze(1)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )

        if eos_token_id is not None and bool(torch.all(next_ids == eos_token_id)):
            break

    if generated_ids:
        generated_ids_tensor = torch.cat(generated_ids, dim=1)
    else:
        generated_ids_tensor = input_ids.new_zeros(input_ids.shape[0], 0)

    return {
        "generated_ids": generated_ids_tensor,
        "full_ids": input_ids,
    }


# Re-export from new module for backward compatibility
from src.policy.legendvla_inference_wrapper import LegendVLAInference  # noqa: E402,F401
