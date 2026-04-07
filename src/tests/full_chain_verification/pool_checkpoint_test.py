"""
Test visual token pooling on trained checkpoint.

Loads the 100k checkpoint and compares:
  1. Attention distribution: baseline vs pooled visual tokens
  2. Action output difference: how much does pooling change predictions?
  3. Ablation: does zeroing visual tokens matter MORE after pooling?

Usage:
    python -m src.tests.full_chain_verification.pool_checkpoint_test \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --vla-shard /path/to/shard.tar \
        --normalizer-path /path/to/normalizer.pkl \
        --checkpoint-path /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from omegaconf import OmegaConf

from src.model.vlm.prefix_cache import PrefixKVCache
from src.policy.legendvla_inference import infer_flow_action
from src.tests.full_chain_verification.part9_input_ablation import (
    build_vla_dataset,
    clone_batch,
    collect_samples,
    collate_to_device,
    action_mse,
    _VisualAblationContext,
)
from src.tests.full_chain_verification.utils import get_output_dir, safe_import_plt
from src.tests.full_chain_verification.visual_fix_experiments import (
    pool_visual_kv,
    get_attention_stats,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--vla-shard", type=str, required=True)
    parser.add_argument("--normalizer-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--pool-sizes", type=str, default="4,8,16,32",
                        help="Comma-separated pool sizes to test")
    args = parser.parse_args()

    out_dir = get_output_dir("pool_checkpoint_test")
    device = "cuda"
    pool_sizes = [int(x) for x in args.pool_sizes.split(",")]

    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator, load_hydra_config,
    )
    cfg = load_hydra_config(args.config_path)
    model, collator = build_model_and_collator(args.config_path, device)

    print(f"Loading checkpoint: {args.checkpoint_path}")
    from src.utils.checkpoint_util import load_checkpoint
    load_checkpoint(model, args.checkpoint_path)
    model.eval()

    dataset = build_vla_dataset(cfg, args.normalizer_path, args.vla_shard)
    samples = collect_samples(dataset, 4)
    batch = collate_to_device(collator, samples[:4], device)

    answer_start_idx = int(batch["answer_start_idx"][0].item())
    input_ids = batch["input_ids"][0, :answer_start_idx]
    backbone = model.backbone
    visual_mask = (input_ids == backbone.video_token_id).cpu().numpy()
    state_mask = (input_ids == backbone.state_token_id).cpu().numpy()
    text_mask = ~(visual_mask | state_mask)
    prefix_len = answer_start_idx
    n_vis = int(visual_mask.sum())

    print(f"Prefix: {prefix_len} = {n_vis} visual + {state_mask.sum()} state + {text_mask.sum()} text")

    # Baseline inference
    print("\n--- Baseline (no pooling) ---")
    with torch.no_grad():
        torch.manual_seed(42)
        baseline_result = infer_flow_action(model, clone_batch(batch))
    baseline_actions = (baseline_result["generated_actions"] if isinstance(baseline_result, dict) else baseline_result).clone()

    # Baseline attention
    baseline_attn = get_attention_stats(model, batch, visual_mask, state_mask, text_mask, prefix_len)
    print(f"  Attn: visual={baseline_attn['visual']:.4f}, state={baseline_attn['state']:.4f}, "
          f"text={baseline_attn['text']:.4f}, sink={baseline_attn['sink_pos0']:.4f}")

    # Baseline visual ablation
    with _VisualAblationContext(model):
        with torch.no_grad():
            torch.manual_seed(42)
            abl_result = infer_flow_action(model, clone_batch(batch))
    abl_actions = (abl_result["generated_actions"] if isinstance(abl_result, dict) else abl_result)
    baseline_vis_abl_mse = action_mse(baseline_actions, abl_actions)
    print(f"  Visual ablation MSE: {baseline_vis_abl_mse:.6f}")

    results = [{
        "pool_size": 0,
        "n_visual_tokens": n_vis,
        "label": "baseline",
        "attention": baseline_attn,
        "action_mse_vs_baseline": 0.0,
        "visual_ablation_mse": baseline_vis_abl_mse,
    }]

    # Test each pool size
    for ps in pool_sizes:
        n_pooled = (n_vis + ps - 1) // ps
        label = f"pool {n_vis}→{n_pooled} (ps={ps})"
        print(f"\n--- {label} ---")

        # Hook to apply pooling
        original_fbs = model.forward_backbone_stream

        def make_patched_fbs(pool_size):
            def patched(batch_arg, slot_embeds, output_attentions=False):
                output = original_fbs(batch_arg, slot_embeds, output_attentions=output_attentions)
                if output.prefix_cache is not None:
                    output.prefix_cache = pool_visual_kv(output.prefix_cache, visual_mask, pool_size=pool_size)
                return output
            return patched

        model.forward_backbone_stream = make_patched_fbs(ps)

        # Inference with pooling
        with torch.no_grad():
            torch.manual_seed(42)
            pooled_result = infer_flow_action(model, clone_batch(batch))
        pooled_actions = (pooled_result["generated_actions"] if isinstance(pooled_result, dict) else pooled_result)
        mse_vs_baseline = action_mse(baseline_actions, pooled_actions)
        print(f"  Action MSE vs baseline: {mse_vs_baseline:.6f}")

        # Attention with pooling
        # Build pooled masks for attention analysis
        vis_indices = np.where(visual_mask)[0]
        non_vis_indices = np.where(~visual_mask)[0]
        first_vis = vis_indices[0] if len(vis_indices) > 0 else 0
        n_non_vis_before = int((non_vis_indices < first_vis).sum())
        new_len = n_non_vis_before + n_pooled + len(non_vis_indices) - n_non_vis_before
        new_vis = np.zeros(new_len, dtype=bool)
        new_vis[n_non_vis_before:n_non_vis_before + n_pooled] = True
        new_state = np.zeros(new_len, dtype=bool)
        new_text = np.zeros(new_len, dtype=bool)
        for i, orig_idx in enumerate(non_vis_indices):
            new_idx = i if orig_idx < first_vis else n_pooled + i
            if new_idx < new_len:
                new_state[new_idx] = state_mask[orig_idx]
                new_text[new_idx] = text_mask[orig_idx]

        pooled_attn = get_attention_stats(model, batch, new_vis, new_state, new_text, new_len)
        print(f"  Attn: visual={pooled_attn['visual']:.4f}, state={pooled_attn['state']:.4f}, "
              f"text={pooled_attn['text']:.4f}")

        # Visual ablation with pooling
        with _VisualAblationContext(model):
            with torch.no_grad():
                torch.manual_seed(42)
                abl_result = infer_flow_action(model, clone_batch(batch))
        abl_actions = (abl_result["generated_actions"] if isinstance(abl_result, dict) else abl_result)
        vis_abl_mse = action_mse(pooled_actions, abl_actions)
        print(f"  Visual ablation MSE: {vis_abl_mse:.6f} (baseline: {baseline_vis_abl_mse:.6f})")

        model.forward_backbone_stream = original_fbs

        results.append({
            "pool_size": ps,
            "n_visual_tokens": n_pooled,
            "label": label,
            "attention": pooled_attn,
            "action_mse_vs_baseline": mse_vs_baseline,
            "visual_ablation_mse": vis_abl_mse,
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'n_vis':>6} {'visual%':>8} {'state%':>8} {'text%':>8} {'vis_abl_MSE':>12} {'vs_baseline':>12}")
    print("-" * 90)
    for r in results:
        a = r["attention"]
        print(f"{r['label']:<25} {r['n_visual_tokens']:>6} {a['visual']:>7.4f} {a['state']:>7.4f} "
              f"{a['text']:>7.4f} {r['visual_ablation_mse']:>12.6f} {r['action_mse_vs_baseline']:>12.6f}")

    # Visualization
    plt = safe_import_plt()
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        labels = [r["label"] for r in results]
        x = range(len(results))

        # Attention breakdown
        ax = axes[0]
        vis_vals = [r["attention"]["visual"] for r in results]
        st_vals = [r["attention"]["state"] for r in results]
        txt_vals = [r["attention"]["text"] for r in results]
        ax.bar(x, vis_vals, label="visual", color="green", alpha=0.8)
        ax.bar(x, st_vals, bottom=vis_vals, label="state", color="orange", alpha=0.8)
        bottoms = [v + s for v, s in zip(vis_vals, st_vals)]
        ax.bar(x, txt_vals, bottom=bottoms, label="text", color="purple", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7, rotation=20)
        ax.set_ylabel("Attention share")
        ax.set_title("Attention Breakdown (100k checkpoint)")
        ax.legend()

        # Visual ablation MSE
        ax = axes[1]
        mse_vals = [r["visual_ablation_mse"] for r in results]
        bars = ax.bar(x, mse_vals, color="steelblue")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7, rotation=20)
        ax.set_ylabel("MSE")
        ax.set_title("Visual Ablation MSE (higher = visual matters more)")
        for bar, val in zip(bars, mse_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.5f}", ha="center", va="bottom", fontsize=7)

        # Action MSE vs baseline
        ax = axes[2]
        mse_vals = [r["action_mse_vs_baseline"] for r in results]
        ax.bar(x, mse_vals, color="coral")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7, rotation=20)
        ax.set_ylabel("MSE")
        ax.set_title("Action MSE vs Baseline (pooling distortion)")

        fig.tight_layout()
        fig.savefig(out_dir / "pool_checkpoint_results.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved pool_checkpoint_results.png")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results.json")


if __name__ == "__main__":
    main()
