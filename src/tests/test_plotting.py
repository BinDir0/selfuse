"""Test script for attention visualization plotting."""

import argparse
import pathlib

import numpy as np
from hydra.utils import instantiate

from src.utils.plotting import (
    _build_token_segments,
    _filter_padding_tokens,
    _load_config,
    _load_npz,
    _plot_causal_mask,
    plot_multilayer_attention_maps,
    visualize_image_attention,
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attention maps and causal masks from saved npz"
    )
    parser.add_argument("--npz", type=str, required=True, help="Path to saved npz file")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment yaml")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--layer_plot_every", type=int, default=4, help="Plot every N layers")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    processor = instantiate(cfg.vla_processor)
    image_token_id = processor.image_token_id
    state_token_id = processor.state_token_id
    action_token_id = processor.action_token_id
    sep_token_id = getattr(processor, "sep_token_id", None)
    pad_token_id = cfg.pad_token_id

    data = _load_npz(args.npz)
    for key, value in data.items():
        print(f"{key}: {value.shape}")
    attn_weights = data.get("attn_weights")
    action_expert_attn_weights = data.get("action_expert_attn_weights")
    input_ids = data["input_ids"]
    attention_mask = data.get("attention_mask")
    causal_mask = data.get("causal_mask")
    n_actions = data.get("n_actions")
    pixel_values = data.get("pixel_values")
    if n_actions is not None:
        n_actions = int(np.array(n_actions).reshape(-1)[0])

    step = data.get("update_step")
    if step is not None:
        step = int(np.array(step).reshape(-1)[0])

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(pathlib.Path(args.npz).parent / "attention_visualization")

    attention_map_items = []
    if attn_weights is not None:
        attention_map_items.append(("attn_weights", attn_weights))
    if action_expert_attn_weights is not None:
        attention_map_items.append(("action_expert_attn_weights", action_expert_attn_weights))

    token_segments_for_causal = None
    token_labels_for_causal = None
    for key, attention_maps in attention_map_items:
        total_len = attention_maps.shape[-1]
        token_segments = _build_token_segments(
            input_ids=input_ids,
            total_len=total_len,
            image_token_id=image_token_id,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
            pad_token_id=pad_token_id,
            other_name='action_expert' if 'action_expert' in key else 'answer', 
        )
        token_labels = []
        for segment_name, segment_len in token_segments:
            token_labels.extend([segment_name] * int(segment_len))
        filtered_attention_maps, filtered_segments, match_q, match_k = _filter_padding_tokens(
            attention_maps, token_labels
        )
        print(f"filtered_segments: {filtered_segments}")
        token_labels_for_causal = token_labels
        token_segments_for_causal = token_segments
        token_segments_for_plot = {
            "segments": filtered_segments,
            "draw_q": match_q,
            "draw_k": match_k,
        }
        print(f"{key} token_segments: drawing matched axes only.")
        plot_multilayer_attention_maps(
            attention_maps=filtered_attention_maps,
            output_dir=output_dir,
            step=step,
            token_segments=token_segments_for_plot,
            layer_plot_every=args.layer_plot_every,
            filename_prefix=key,
        )
    if causal_mask is not None:
        token_segments_for_plot = None
        if token_labels_for_causal is not None:
            labels_len = len(token_labels_for_causal)
            q_len = causal_mask.shape[-2]
            k_len = causal_mask.shape[-1]
            if labels_len == q_len and labels_len == k_len:
                token_segments_for_plot = token_segments_for_causal
        _plot_causal_mask(causal_mask, output_dir, step=step, token_segments=token_segments_for_plot)
    if pixel_values is not None and attn_weights is not None:
        patch_size = cfg.get("patch_size")
        visualize_image_attention(
            attn_weights=attn_weights,
            action_expert_attn_weights=action_expert_attn_weights,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_token_id=image_token_id,
            sep_token_id=sep_token_id,
            n_actions=n_actions,
            output_dir=output_dir,
            step=step,
            patch_size=patch_size,
        )


if __name__ == "__main__":
    main()
