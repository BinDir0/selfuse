"""
Deep analysis of action expert attention weights.

Builds on test 9.9 results to answer:
  1. Which text tokens receive the most attention? (instruction vs template)
  2. Are there attention sinks (few tokens dominating)?
  3. Is visual token attention uniform or concentrated on specific patches?
  4. How does attention distribution change across ODE steps?

Usage:
    python -m src.tests.full_chain_verification.attention_deep_analysis \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --vla-shard /path/to/shard.tar \
        --normalizer-path /path/to/normalizer.pkl \
        --checkpoint-path /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.policy.legendvla_inference import infer_flow_action
from src.tests.full_chain_verification.part9_input_ablation import (
    build_vla_dataset,
    clone_batch,
    collect_samples,
    collate_to_device,
)
from src.tests.full_chain_verification.utils import get_output_dir, safe_import_plt

OmegaConf.register_new_resolver("eval", eval, replace=True)


def decode_prefix_tokens(tokenizer, input_ids: torch.Tensor, answer_start_idx: int) -> list[str]:
    """Decode each prefix token to its string representation."""
    prefix_ids = input_ids[0, :answer_start_idx].tolist()
    return [tokenizer.decode([tid]) for tid in prefix_ids]


def run_analysis(
    config_path: str,
    vla_shard: str,
    normalizer_path: str,
    checkpoint_path: str,
):
    out_dir = get_output_dir("part9_attention_deep")
    device = "cuda"

    # Build model
    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator, load_hydra_config,
    )
    cfg = load_hydra_config(config_path)
    model, collator = build_model_and_collator(config_path, device)

    print(f"Loading checkpoint: {checkpoint_path}")
    from src.utils.checkpoint_util import load_checkpoint
    load_checkpoint(model, checkpoint_path)
    model.eval()

    # Load data
    dataset = build_vla_dataset(cfg, normalizer_path, vla_shard)
    samples = collect_samples(dataset, 2)
    batch = collate_to_device(collator, samples[:2], device)
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")

    # Get token info
    answer_start_idx = int(batch["answer_start_idx"][0].item())
    backbone = model.backbone
    tokenizer = backbone.tokenizer

    # Decode prefix tokens
    prefix_tokens = decode_prefix_tokens(tokenizer, batch["input_ids"], answer_start_idx)

    # Classify token types
    input_ids = batch["input_ids"][0, :answer_start_idx]
    visual_mask = (input_ids == backbone.video_token_id).cpu().numpy()
    state_mask = (input_ids == backbone.state_token_id).cpu().numpy()
    text_mask = ~(visual_mask | state_mask)

    n_visual = int(visual_mask.sum())
    n_state = int(state_mask.sum())
    n_text = int(text_mask.sum())
    print(f"\nPrefix: {answer_start_idx} tokens = {n_visual} visual + {n_state} state + {n_text} text")

    # Run inference with attention weights (eager mode)
    expert = model.flow_expert
    original_impl = expert.config._attn_implementation
    expert.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            torch.manual_seed(42)
            result = infer_flow_action(model, clone_batch(batch), output_attentions=True)
    finally:
        expert.config._attn_implementation = original_impl

    expert_attn = result["expert_attention_weights"]
    n_steps = len(expert_attn)
    n_layers = len(expert_attn[0])
    print(f"ODE steps: {n_steps}, layers: {n_layers}")

    # ── Analysis 1: Per-token attention in prefix (last ODE step) ──────────

    # Average attention to each prefix position across layers and heads (sample 0)
    # Shape per layer: [B, H, action_len, kv_len]
    last_step = expert_attn[-1]
    prefix_len = answer_start_idx

    # Aggregate: mean over heads and action positions -> per-kv-position attention
    per_pos_attn_by_layer = []
    for layer_idx in range(n_layers):
        w = last_step[layer_idx]
        if w is None:
            per_pos_attn_by_layer.append(None)
            continue
        # w[0]: [H, action_len, kv_len]
        w_prefix = w[0].float()[:, :, :prefix_len]  # [H, action_len, prefix_len]
        # Mean over heads and action positions -> [prefix_len]
        per_pos = w_prefix.mean(dim=(0, 1))
        per_pos_attn_by_layer.append(per_pos.cpu().numpy())

    # Average across all layers
    valid_layers = [l for l in per_pos_attn_by_layer if l is not None]
    avg_per_pos = np.mean(valid_layers, axis=0)  # [prefix_len]

    # ── Analysis 1a: Top attended text tokens ──────────────────────────────

    text_indices = np.where(text_mask)[0]
    text_attn = avg_per_pos[text_indices]
    top_k = min(30, len(text_indices))
    top_text_order = np.argsort(text_attn)[::-1][:top_k]

    print(f"\n{'='*80}")
    print(f"Top {top_k} most-attended TEXT tokens (averaged across layers, last ODE step):")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Pos':<6} {'Token':<30} {'Attn':>10} {'% of text':>10}")
    print(f"{'-'*61}")
    total_text_attn = text_attn.sum()
    for rank, idx in enumerate(top_text_order):
        pos = text_indices[idx]
        token_str = prefix_tokens[pos].replace('\n', '\\n')
        attn_val = text_attn[idx]
        pct = attn_val / total_text_attn * 100
        print(f"{rank+1:<5} {pos:<6} {token_str:<30} {attn_val:>10.6f} {pct:>9.1f}%")

    # ── Analysis 1b: Attention concentration (Gini coefficient) ────────────

    def gini(arr):
        arr = np.sort(arr)
        n = len(arr)
        idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1) / n

    text_gini = gini(text_attn)
    visual_attn = avg_per_pos[visual_mask]
    visual_gini = gini(visual_attn) if n_visual > 0 else 0
    state_attn = avg_per_pos[state_mask]
    state_gini = gini(state_attn) if n_state > 0 else 0

    print(f"\nAttention concentration (Gini coefficient, 0=uniform, 1=concentrated):")
    print(f"  Text:   {text_gini:.4f} (n={n_text})")
    print(f"  Visual: {visual_gini:.4f} (n={n_visual})")
    print(f"  State:  {state_gini:.4f} (n={n_state})")

    # ── Analysis 1c: Attention sink detection ──────────────────────────────

    top1_pct = avg_per_pos.max() / avg_per_pos.sum() * 100
    top5_pct = np.sort(avg_per_pos)[-5:].sum() / avg_per_pos.sum() * 100
    top10_pct = np.sort(avg_per_pos)[-10:].sum() / avg_per_pos.sum() * 100
    top5_pos = np.argsort(avg_per_pos)[-5:][::-1]

    print(f"\nAttention sink detection:")
    print(f"  Top 1 token:  {top1_pct:.1f}% of total prefix attention")
    print(f"  Top 5 tokens: {top5_pct:.1f}%")
    print(f"  Top 10 tokens: {top10_pct:.1f}%")
    print(f"  Top 5 positions and tokens:")
    for pos in top5_pos:
        token_str = prefix_tokens[pos].replace('\n', '\\n')
        ttype = "VISUAL" if visual_mask[pos] else ("STATE" if state_mask[pos] else "TEXT")
        print(f"    pos={pos}, type={ttype}, token='{token_str}', attn={avg_per_pos[pos]:.6f}")

    # ── Analysis 2: Per-layer top attended positions ───────────────────────

    print(f"\n{'='*80}")
    print(f"Per-layer analysis: top 3 prefix positions")
    print(f"{'='*80}")
    for layer_idx in range(n_layers):
        if per_pos_attn_by_layer[layer_idx] is None:
            continue
        layer_attn = per_pos_attn_by_layer[layer_idx]
        top3 = np.argsort(layer_attn)[-3:][::-1]
        desc = []
        for p in top3:
            ttype = "V" if visual_mask[p] else ("S" if state_mask[p] else "T")
            tok = prefix_tokens[p].replace('\n', '\\n')[:15]
            desc.append(f"pos={p}({ttype},'{tok}')={layer_attn[p]:.4f}")
        print(f"  Layer {layer_idx:2d}: {', '.join(desc)}")

    # ── Analysis 3: ODE step evolution ─────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"Attention evolution across ODE steps (t=0 → t=1)")
    print(f"{'='*80}")
    step_summaries = []
    for step_idx in range(n_steps):
        step_weights = expert_attn[step_idx]
        visual_total = 0.0
        state_total = 0.0
        text_total = 0.0
        action_total = 0.0
        count = 0
        for layer_idx in range(n_layers):
            w = step_weights[layer_idx]
            if w is None:
                continue
            w_mean = w[0].float().mean(dim=0)  # [action_len, kv_len]
            prefix_w = w_mean[:, :prefix_len]
            visual_total += prefix_w[:, torch.tensor(visual_mask)].sum().item() / w_mean.shape[0]
            state_total += prefix_w[:, torch.tensor(state_mask)].sum().item() / w_mean.shape[0]
            text_total += prefix_w[:, torch.tensor(text_mask)].sum().item() / w_mean.shape[0]
            action_total += w_mean[:, prefix_len:].sum().item() / w_mean.shape[0]
            count += 1
        if count > 0:
            s = {"step": step_idx, "t": (step_idx + 1) / n_steps,
                 "visual": visual_total / count, "state": state_total / count,
                 "text": text_total / count, "action": action_total / count}
            step_summaries.append(s)
            print(f"  Step {step_idx:2d} (t≈{s['t']:.2f}): "
                  f"visual={s['visual']:.4f}, state={s['state']:.4f}, "
                  f"text={s['text']:.4f}, action={s['action']:.4f}")

    # ── Analysis 4: Visual tokens internal distribution ────────────────────

    if n_visual > 0:
        print(f"\n{'='*80}")
        print(f"Visual token internal attention distribution")
        print(f"{'='*80}")
        # Check if visual attention is uniform or concentrated on specific patches
        visual_indices = np.where(visual_mask)[0]
        visual_attn_vals = avg_per_pos[visual_indices]

        # Top and bottom visual tokens
        top5_vis = np.argsort(visual_attn_vals)[-5:][::-1]
        bot5_vis = np.argsort(visual_attn_vals)[:5]

        print(f"  Mean visual attn: {visual_attn_vals.mean():.6f}")
        print(f"  Std visual attn:  {visual_attn_vals.std():.6f}")
        print(f"  Max/Min ratio:    {visual_attn_vals.max() / max(visual_attn_vals.min(), 1e-10):.1f}x")
        print(f"  Top 5 visual positions (relative): {visual_indices[top5_vis].tolist()}")
        print(f"  Bottom 5 visual positions (relative): {visual_indices[bot5_vis].tolist()}")

    # ── Visualization ──────────────────────────────────────────────────────

    plt = safe_import_plt()
    if plt is not None:
        # Plot 1: Per-position attention heatmap
        fig, ax = plt.subplots(figsize=(20, 4))
        colors = np.zeros((prefix_len, 3))
        for i in range(prefix_len):
            if visual_mask[i]:
                colors[i] = [0.2, 0.7, 0.2]   # green for visual
            elif state_mask[i]:
                colors[i] = [1.0, 0.6, 0.0]   # orange for state
            else:
                colors[i] = [0.5, 0.3, 0.8]   # purple for text
        ax.bar(range(prefix_len), avg_per_pos, color=colors, width=1.0, edgecolor='none')
        ax.set_xlabel("Prefix position")
        ax.set_ylabel("Mean attention (across layers & heads)")
        ax.set_title(f"Per-Position Prefix Attention (green=visual, orange=state, purple=text)")
        ax.set_xlim(-1, prefix_len + 1)
        fig.tight_layout()
        fig.savefig(out_dir / "per_position_attention.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved per_position_attention.png")

        # Plot 2: ODE step evolution
        if step_summaries:
            fig, ax = plt.subplots(figsize=(10, 5))
            steps = [s["t"] for s in step_summaries]
            ax.plot(steps, [s["visual"] for s in step_summaries], 'g-o', label="visual", linewidth=2)
            ax.plot(steps, [s["state"] for s in step_summaries], '-o', color="orange", label="state", linewidth=2)
            ax.plot(steps, [s["text"] for s in step_summaries], 'purple', marker='o', label="text", linewidth=2)
            ax.plot(steps, [s["action"] for s in step_summaries], 'r-o', label="action (self-attn)", linewidth=2)
            ax.set_xlabel("ODE time t")
            ax.set_ylabel("Mean attention share")
            ax.set_title("Attention Distribution Evolution Across ODE Steps")
            ax.legend()
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(out_dir / "ode_step_evolution.png", dpi=150)
            plt.close(fig)
            print(f"Saved ode_step_evolution.png")

        # Plot 3: Text token attention ranked
        if n_text > 0:
            sorted_text_attn = np.sort(text_attn)[::-1]
            cumulative = np.cumsum(sorted_text_attn) / sorted_text_attn.sum()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.bar(range(len(sorted_text_attn)), sorted_text_attn, color="purple", alpha=0.7)
            ax1.set_xlabel("Text token rank")
            ax1.set_ylabel("Attention value")
            ax1.set_title(f"Text Token Attention Distribution (n={n_text})")

            ax2.plot(range(len(cumulative)), cumulative, 'purple', linewidth=2)
            ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label="80%")
            ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label="50%")
            n_for_80 = np.searchsorted(cumulative, 0.8) + 1
            n_for_50 = np.searchsorted(cumulative, 0.5) + 1
            ax2.set_xlabel("Number of text tokens (sorted by attention)")
            ax2.set_ylabel("Cumulative attention share")
            ax2.set_title(f"Cumulative: {n_for_50} tokens for 50%, {n_for_80} tokens for 80%")
            ax2.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "text_attention_distribution.png", dpi=150)
            plt.close(fig)
            print(f"Saved text_attention_distribution.png")

        # Plot 4: Layer-wise attention heatmap over prefix positions (sampled)
        # Sample every 4th position to keep heatmap readable
        sample_step = max(1, prefix_len // 128)
        sampled_positions = list(range(0, prefix_len, sample_step))
        heatmap_data = []
        for layer_idx in range(n_layers):
            if per_pos_attn_by_layer[layer_idx] is not None:
                heatmap_data.append(per_pos_attn_by_layer[layer_idx][sampled_positions])
            else:
                heatmap_data.append(np.zeros(len(sampled_positions)))
        heatmap_data = np.array(heatmap_data)

        fig, ax = plt.subplots(figsize=(20, 8))
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis',
                       interpolation='nearest')
        ax.set_xlabel("Sampled prefix position")
        ax.set_ylabel("Layer")
        ax.set_title(f"Per-Layer Attention Heatmap (sampled every {sample_step} positions)")

        # Mark token type boundaries
        vis_start = np.where(visual_mask)[0][0] // sample_step if n_visual > 0 else 0
        vis_end = np.where(visual_mask)[0][-1] // sample_step if n_visual > 0 else 0
        state_start = np.where(state_mask)[0][0] // sample_step if n_state > 0 else 0
        state_end = np.where(state_mask)[0][-1] // sample_step if n_state > 0 else 0
        ax.axvline(x=vis_start - 0.5, color='lime', linewidth=1, linestyle='--', label='visual region')
        ax.axvline(x=vis_end + 0.5, color='lime', linewidth=1, linestyle='--')
        ax.axvline(x=state_start - 0.5, color='orange', linewidth=1, linestyle='--', label='state region')
        ax.axvline(x=state_end + 0.5, color='orange', linewidth=1, linestyle='--')
        ax.legend(loc='upper right')
        fig.colorbar(im, ax=ax, label='Attention weight')
        fig.tight_layout()
        fig.savefig(out_dir / "layer_position_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"Saved layer_position_heatmap.png")

    # Save raw data
    report = {
        "prefix_len": prefix_len,
        "n_visual": n_visual,
        "n_state": n_state,
        "n_text": n_text,
        "text_gini": float(text_gini),
        "visual_gini": float(visual_gini),
        "state_gini": float(state_gini),
        "top1_pct": float(top1_pct),
        "top5_pct": float(top5_pct),
        "top10_pct": float(top10_pct),
        "top_text_tokens": [
            {"pos": int(text_indices[idx]), "token": prefix_tokens[text_indices[idx]], "attn": float(text_attn[idx])}
            for idx in top_text_order
        ],
        "step_evolution": step_summaries,
    }
    with open(out_dir / "deep_analysis.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved deep_analysis.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--vla-shard", type=str, required=True)
    parser.add_argument("--normalizer-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    args = parser.parse_args()
    run_analysis(
        config_path=args.config_path,
        vla_shard=args.vla_shard,
        normalizer_path=args.normalizer_path,
        checkpoint_path=args.checkpoint_path,
    )
