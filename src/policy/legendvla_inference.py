"""
Inference functions extracted from LegendVLA.

Contains:
- infer_action: Flow matching action inference
- infer_single_step: Single-step VLM inference
- infer_vlm: Multi-step autoregressive text generation
- infer_vla: Autoregressive VLA action inference with DiffLoss
- LegendVLAInference: Inference wrapper class
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

from src.model.common.kv_cache import KVCache
from src.utils.generation_utils import sample_token, concat_attn_weights

log = logging.getLogger(__name__)


def infer_action(
    model,
    input: dict,
    return_attn_weights: bool = False,
) -> torch.FloatTensor:
    """
    Inference function for action generation using flow matching.

    Args:
        model: LegendVLA model instance
        input: Input dictionary containing input_ids, pixel_values, vlm_mask,
               action_mask, vlm_position_ids, action_position_ids
        return_attn_weights: Whether to return attention weights

    Returns:
        torch.FloatTensor: [B, horizon_steps, action_dim] Generated action sequence
    """
    input_ids = input["input_ids"]
    pixel_values = input["pixel_values"]
    vlm_mask = input["vlm_mask"]
    action_mask = input["action_mask"]
    vlm_position_ids = input["vlm_position_ids"]
    action_position_ids = input["action_position_ids"]

    dtype, device = pixel_values.dtype, pixel_values.device
    bsz = pixel_values.size(0)

    kv_caches = model.joint_model.build_mixture_caches()

    if 'depth_values' in input:
        depth_values = input["depth_values"]
        has_depth_values = input["has_depth_values"]
    else:
        depth_values = None
        has_depth_values = None
    inputs_embeds = model._forward_siglip_and_text_embedding(
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
    _, kv_caches = model.joint_model(
        attention_mask=vlm_mask,
        position_ids_all={"vlm": vlm_position_ids},
        embeds_all={"vlm": inputs_embeds},
        kv_caches=kv_caches,
        return_caches=True,
    )
    vlm_attn_weights = torch.stack(model.attn_weights, dim=0).detach().clone()
    action_expert_attn_weights = None

    # sample pure action noise
    action = torch.randn(
        (bsz, model.horizon_steps, model.action_dim), device=device, dtype=dtype
    )

    # forward euler integration --- using kv caches of vlm
    delta_t = 1.0 / model.num_inference_steps
    t = torch.zeros(bsz, device=device, dtype=dtype)
    for step_idx in range(model.num_inference_steps):
        time_cond = model.time_embedding(t)
        if model.action_expert_adaptive_mode:
            action_embeds = model.action_encoder(action)
        else:
            action_embeds = model.action_encoder(action, time_cond)
        action_embeds = action_embeds / (model.action_hidden_size**0.5)
        action_embeds = model.joint_model(
            attention_mask=action_mask,
            position_ids_all={"action": action_position_ids},
            embeds_all={"action": action_embeds},
            time_cond=time_cond,
            kv_caches=kv_caches,
            cache_mode="append_non_active",
        )["action"]
        if step_idx == 0:
            action_expert_attn_weights = torch.stack(model.attn_weights, dim=0).detach().clone()

        action_vel = model.action_decoder(action_embeds)
        action += delta_t * action_vel
        t += delta_t

    if return_attn_weights:
        return action, vlm_attn_weights, action_expert_attn_weights
    return action


@torch.inference_mode()
def infer_single_step(
    model,
    input: dict,
    kv_cache: Optional[KVCache] = None,
    dtype: torch.dtype = torch.float32,
    return_attn_weights: bool = False,
) -> dict:
    """
    Inference function for discrete action generation (single step).

    Args:
        model: LegendVLA model instance
        input: Input dictionary containing input_ids, pixel_values (optional),
               attention_mask, states, actions, n_states, n_actions
        kv_cache: Key-value cache for the generated tokens
        dtype: Data type for the input and output
        return_attn_weights: Whether to return attention weights

    Returns:
        dict: {"hidden_states", "kv_cache" (optional), "attn_weights" (optional)}
    """
    input_ids = input["input_ids"]
    attention_mask = input["attention_mask"]
    q_len = input_ids.size(1)

    inputs_embeds = model._forward_siglip_and_text_embedding(
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

    causal_mask, position_ids = model.build_causal_mask_and_position_ids_for_text(
        q_len, attention_mask, kv_cache, dtype
    )

    hidden_states = model.joint_model(
        attention_mask=causal_mask,
        position_ids_all={"vlm": position_ids},
        embeds_all={"vlm": inputs_embeds},
        kv_caches={"vlm": kv_cache},
        cache_mode="append",
        final_layer_post_attn_skip_names=[],
    )["vlm"]
    output = {"hidden_states": hidden_states}
    if return_attn_weights:
        output["attn_weights"] = torch.stack(model.attn_weights, dim=0).detach().clone()
    if kv_cache is not None:
        output["kv_cache"] = kv_cache
    return output


@torch.inference_mode()
def infer_vlm(
    model,
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
    Multi-step autoregressive generation function for VLM.

    Args:
        model: LegendVLA model instance
        input: Input dictionary containing input_ids, pixel_values, attention_mask
        max_new_tokens: Maximum number of new tokens to generate
        temperature, top_k, top_p: Sampling parameters
        allowed_token_ids: Allowed token ID range for sampling
        eos_token_id: End-of-sequence token ID
        return_kv_cache: Whether to return KV cache
        return_attn_weights: Whether to return attention weights

    Returns:
        dict: {"generated_ids", "kv_cache" (optional), "attn_weights" (optional)}
    """
    input_ids = input["input_ids"]
    pixel_values = input["pixel_values"]
    attention_mask = input.get("attention_mask")

    batch_size = input_ids.size(0)
    device, dtype = input_ids.device, pixel_values.dtype

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    kv_cache = KVCache()

    # ========== Prefill Phase ==========
    prefill_input = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "depth_values": input.get("depth_values"),
        "has_depth_values": input.get("has_depth_values"),
        "attention_mask": attention_mask,
    }

    attn_weights_steps = [] if return_attn_weights else None
    prefill_output = infer_single_step(
        model, prefill_input, kv_cache=kv_cache, dtype=dtype,
        return_attn_weights=return_attn_weights,
    )
    prefill_hidden_states = prefill_output["hidden_states"]
    prefill_logits = model.lm_head(prefill_hidden_states)
    prefill_logits = model._apply_final_logit_softcapping(prefill_logits)
    kv_cache = prefill_output.get("kv_cache", kv_cache)
    if return_attn_weights:
        attn_weights_steps.append(prefill_output.get("attn_weights"))

    # Sample first new token from the last position of prefill
    next_token_logits = prefill_logits[:, -1, :]
    next_token_ids = sample_token(
        next_token_logits, temperature=temperature, top_k=top_k,
        top_p=top_p, allowed_token_ids=allowed_token_ids,
    )

    generated_ids = [input_ids.clone()]
    generated_ids.append(next_token_ids.unsqueeze(1))

    finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if eos_token_id is not None:
        finished_mask = (next_token_ids == eos_token_id)

    # ========== Generation Phase ==========
    for _ in range(max_new_tokens - 1):
        if finished_mask.all():
            break

        active_mask = ~finished_mask
        step_input_ids = next_token_ids.clone()
        step_input_ids[finished_mask] = model.pad_token_id

        attention_mask = torch.cat([
            attention_mask, torch.ones(
                (batch_size, 1), dtype=torch.long, device=device
            ) * active_mask.unsqueeze(1),
        ], dim=-1)
        step_input = {
            "input_ids": step_input_ids.unsqueeze(1),
            "attention_mask": attention_mask,
        }

        step_output = infer_single_step(
            model, step_input, kv_cache=kv_cache, dtype=dtype,
            return_attn_weights=return_attn_weights,
        )
        step_hidden_states = step_output["hidden_states"]
        step_logits = model.lm_head(step_hidden_states)
        step_logits = model._apply_final_logit_softcapping(step_logits)
        kv_cache = step_output.get("kv_cache", kv_cache)
        if return_attn_weights:
            attn_weights_steps.append(step_output.get("attn_weights"))

        next_token_ids = torch.full(
            (batch_size,), model.pad_token_id, dtype=torch.long, device=device
        )
        next_token_logits = step_logits[:, -1, :]
        next_token_ids_active = sample_token(
            next_token_logits, temperature=temperature, top_k=top_k,
            top_p=top_p, allowed_token_ids=allowed_token_ids,
        )
        next_token_ids[active_mask] = next_token_ids_active[active_mask]

        if eos_token_id is not None:
            finished_mask = finished_mask | (next_token_ids == eos_token_id)

        generated_ids.append(next_token_ids.unsqueeze(1))

    generated_ids_tensor = torch.cat(generated_ids, dim=1)
    result = {"generated_ids": generated_ids_tensor}
    if return_kv_cache:
        result["kv_cache"] = kv_cache
    if return_attn_weights:
        result["attn_weights"] = concat_attn_weights(attn_weights_steps)
    return result


@torch.inference_mode()
def infer_vla(
    model,
    input: dict,
    max_new_tokens: int,
    temperature: float = 1.0,
    return_attn_weights: bool = False,
    cfg: float = 1.0,
    **kwargs,
) -> Dict[str, torch.FloatTensor]:
    """
    Autoregressive action inference for VLA using DiffLoss sampling.

    Args:
        model: LegendVLA model instance
        input: Input dictionary containing input_ids, pixel_values, attention_mask,
               states, n_states, is_vla_data
        max_new_tokens: Number of action tokens to generate
        temperature: Sampling temperature for diffusion
        return_attn_weights: Whether to return attention weights
        cfg: Classifier-free guidance scale for diffusion sampling

    Returns:
        dict: {"generated_actions", "prefill_vlm_hidden_states",
               "prefill_image_hidden_states", "prefill_state_hidden_states",
               "prefill_text_hidden_states", "generated_hidden_states",
               "attn_weights" (optional)}
    """
    input_ids = input["input_ids"]
    pixel_values = input.get("pixel_values")
    attention_mask = input.get("attention_mask")

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    prefill_attention_mask = attention_mask

    batch_size = input_ids.size(0)
    device = input_ids.device
    dtype = pixel_values.dtype if pixel_values is not None else torch.float32

    is_vla_data = input.get("is_vla_data")
    if is_vla_data is None:
        is_vla_data = torch.ones(batch_size, dtype=torch.bool, device=device)

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
    prefill_output = infer_single_step(
        model, prefill_input, kv_cache=kv_cache, dtype=dtype,
        return_attn_weights=return_attn_weights,
    )
    prefill_hidden_states = prefill_output["hidden_states"]
    hidden_dim = prefill_hidden_states.shape[-1]
    kv_cache = prefill_output.get("kv_cache", kv_cache)
    if return_attn_weights:
        attn_weights_steps.append(prefill_output.get("attn_weights"))

    # Prefill VLM hidden states: keep mask==1, exclude last token
    prefill_mask = prefill_attention_mask.to(torch.bool)
    if prefill_mask.numel() > 0:
        prefill_mask[:, -1] = False
    prefill_vlm_hidden_flat = prefill_hidden_states[prefill_mask]

    # Split prefill by token type using input_ids
    image_mask = (input_ids == model.image_token_index) & prefill_mask
    state_mask = (input_ids == model.state_token_index) & prefill_mask
    text_mask = (
        (input_ids != model.image_token_index)
        & (input_ids != model.state_token_index)
        & (input_ids != model.action_token_index)
        & (input_ids != model.pad_token_id)
        & prefill_mask
    )
    prefill_image_hidden_flat = prefill_hidden_states[image_mask]
    prefill_state_hidden_flat = prefill_hidden_states[state_mask]
    prefill_text_hidden_flat = prefill_hidden_states[text_mask]

    # Sample the first action from the last token
    next_condition_states = prefill_hidden_states[:, -1, :]
    latent_condition = model.latent_condition_projector(next_condition_states)
    next_action = model.diffloss.sample(latent_condition, temperature=temperature, cfg=cfg)
    generated_actions = [next_action.unsqueeze(1)]

    # ========== Generation Phase ==========
    generated_hidden_steps = []
    for _ in range(max_new_tokens - 1):
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)],
            dim=-1,
        )
        step_input = {
            "input_ids": torch.full(
                (batch_size, 1), model.action_token_index,
                dtype=input_ids.dtype, device=device,
            ),
            "attention_mask": attention_mask,
            "actions": next_action.unsqueeze(1),
            "n_actions": torch.ones(batch_size, dtype=torch.long, device=device),
            "is_vla_data": is_vla_data,
        }

        step_output = infer_single_step(
            model, step_input, kv_cache=kv_cache, dtype=dtype,
            return_attn_weights=return_attn_weights,
        )
        step_hidden_states = step_output["hidden_states"]
        kv_cache = step_output.get("kv_cache", kv_cache)
        if return_attn_weights:
            attn_weights_steps.append(step_output.get("attn_weights"))

        generated_hidden_steps.append(step_hidden_states[:, -1, :])

        latent_condition = model.latent_condition_projector(step_hidden_states[:, -1, :])
        next_action = model.diffloss.sample(latent_condition, temperature=temperature, cfg=cfg)
        generated_actions.append(next_action.unsqueeze(1))

    generated_actions_tensor = torch.cat(generated_actions, dim=1)
    if generated_hidden_steps:
        generated_hidden_flat = torch.cat(generated_hidden_steps, dim=0)
    else:
        generated_hidden_flat = prefill_hidden_states.new_empty((0, hidden_dim))
    result = {
        "generated_actions": generated_actions_tensor,
        "prefill_vlm_hidden_states": prefill_vlm_hidden_flat,
        "prefill_image_hidden_states": prefill_image_hidden_flat,
        "prefill_state_hidden_states": prefill_state_hidden_flat,
        "prefill_text_hidden_states": prefill_text_hidden_flat,
        "generated_hidden_states": generated_hidden_flat,
    }
    if return_attn_weights:
        result["attn_weights"] = concat_attn_weights(attn_weights_steps)
    return result


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
        use_mlp_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        model_config_path = pathlib.Path(model_config_path)
        model_cfg = OmegaConf.load(model_config_path)
        
        # Patch for old checkpoint compatibility (enable LayerNorm in MLPs)
        # This is specifically for the checkpoint trained on 2026.02.17 which used LayerNorm
        if use_mlp_layer_norm:
            print("启用MLP LayerNorm兼容模式 (Configured via inference.yaml)")
            policy_cfg = model_cfg.policy
            if hasattr(policy_cfg, "action_encoder"):
                policy_cfg.action_encoder.use_mlp_layer_norm = True
            if hasattr(policy_cfg, "action_decoder"):
                policy_cfg.action_decoder.use_mlp_layer_norm = True
            if hasattr(policy_cfg, "action_encoder_ar"):
                policy_cfg.action_encoder_ar.use_mlp_layer_norm = True
            if hasattr(policy_cfg, "latent_condition_projector"):
                policy_cfg.latent_condition_projector.use_mlp_layer_norm = True
        
        self.model: nn.Module = hydra.utils.instantiate(model_cfg.policy)
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        self.model.eval()

        if diffusion_sampling_steps:
            self.model.diffloss.num_sampling_steps = diffusion_sampling_steps
        if diffusion_use_ddim_sampling:
            self.model.diffloss.use_ddim_sampling = diffusion_use_ddim_sampling
        if flow_sampling_steps:
            self.model.num_inference_steps = flow_sampling_steps
            self.model.diffloss.num_inference_steps = flow_sampling_steps

        self.processor = hydra.utils.instantiate(model_cfg.vla_processor)
        if hasattr(self.processor, "tokenizer_padding"):
            self.processor.tokenizer_padding = tokenizer_padding
        if hasattr(self.processor, "depth_clip_range") and getattr(self.processor, "depth_clip_range", None) is None:
            depth_clip_range = OmegaConf.select(model_cfg, "depth_clip_range", default=None)
            if depth_clip_range is not None:
                self.processor.depth_clip_range = tuple(float(x) for x in depth_clip_range)

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
        """Convert raw data into standard processor inputs."""
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
            "has_depth_values": processed.get("has_depth_values", torch.zeros(batch_size, dtype=torch.bool)),
            "n_states": prepared["n_states"],
            "states": prepared["states"].to(self.dtype),
            "is_vla_data": torch.ones(batch_size, dtype=torch.bool),
        }
        if "depth_values" in processed:
            inputs["depth_values"] = processed["depth_values"].to(self.dtype)

        if self.mode == "flow":
            inputs["n_actions"] = torch.full((batch_size, ), self.action_horizon, dtype=torch.long)
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
                              temperature=self.ar_temperature, cfg=self.ar_cfg)["generated_actions"]