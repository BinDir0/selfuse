"""
Quick experiments to fix visual token attention.

Tests several interventions on a single-batch overfit to see which
changes the attention distribution most:

  A. Baseline (knowledge_insulation=True, as-is)
  B. Sink masking: mask position 0 in prefix cache
  C. Visual pooling: average-pool visual K/V from 432 → 27 tokens (4x4)
  D. Visual pooling + sink masking
  E. Visual temperature: scale visual K by 1/sqrt(tau) to amplify logits

Each runs 100-step overfit from random init, measuring attention at
steps 0, 25, 50, 75, 99.

Usage:
    python -m src.tests.full_chain_verification.visual_fix_experiments \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --vla-shard /path/to/shard.tar \
        --normalizer-path /path/to/normalizer.pkl \
        --steps 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.model.vlm.prefix_cache import PrefixKVCache
from src.policy.legendvla_inference import infer_flow_action
from src.tests.full_chain_verification.part9_input_ablation import (
    build_vla_dataset,
    clone_batch,
    collect_samples,
    collate_to_device,
)
from src.tests.full_chain_verification.utils import get_output_dir, safe_import_plt

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ── Prefix cache manipulation ────────────────────────────────────────────────

def pool_visual_kv(cache: PrefixKVCache, visual_mask: np.ndarray, pool_size: int = 16) -> PrefixKVCache:
    """Average-pool visual K/V tokens, keeping text/state tokens intact.

    Replaces 432 visual tokens with ceil(432/pool_size) pooled tokens.
    The mask and non-visual positions are preserved.
    """
    visual_indices = np.where(visual_mask)[0]
    non_visual_indices = np.where(~visual_mask)[0]
    n_vis = len(visual_indices)
    if n_vis == 0 or pool_size <= 1:
        return cache

    # Pool visual tokens
    n_pooled = (n_vis + pool_size - 1) // pool_size
    # keys: [num_layers, B, num_kv_heads, kv_seq_len, head_dim]
    device = cache.keys.device
    dtype = cache.keys.dtype
    L, B, H, S, D = cache.keys.shape

    vis_idx = torch.tensor(visual_indices, device=device)
    non_vis_idx = torch.tensor(non_visual_indices, device=device)

    # Extract visual and non-visual
    vis_keys = cache.keys[:, :, :, vis_idx, :]      # [L, B, H, n_vis, D]
    vis_values = cache.values[:, :, :, vis_idx, :]
    non_vis_keys = cache.keys[:, :, :, non_vis_idx, :]
    non_vis_values = cache.values[:, :, :, non_vis_idx, :]

    # Average pool visual tokens
    # Pad to multiple of pool_size
    pad_n = n_pooled * pool_size - n_vis
    if pad_n > 0:
        vis_keys = torch.nn.functional.pad(vis_keys, (0, 0, 0, pad_n))
        vis_values = torch.nn.functional.pad(vis_values, (0, 0, 0, pad_n))
    pooled_keys = vis_keys.reshape(L, B, H, n_pooled, pool_size, D).mean(dim=4)
    pooled_values = vis_values.reshape(L, B, H, n_pooled, pool_size, D).mean(dim=4)

    # Reconstruct: [pooled_visual | non_visual] preserving relative order
    # Put pooled visual at the start of where visual was
    new_keys = torch.cat([non_vis_keys[:, :, :, :min(visual_indices[0], len(non_visual_indices)), :],
                          pooled_keys,
                          non_vis_keys[:, :, :, min(visual_indices[0], len(non_visual_indices)):, :]], dim=3)
    new_values = torch.cat([non_vis_values[:, :, :, :min(visual_indices[0], len(non_visual_indices)), :],
                            pooled_values,
                            non_vis_values[:, :, :, min(visual_indices[0], len(non_visual_indices)):, :]], dim=3)

    new_seq_len = new_keys.shape[3]
    # Rebuild mask: all True for the new positions that are valid
    new_mask = torch.ones(B, new_seq_len, dtype=torch.bool, device=device)
    new_lengths = torch.full((B,), new_seq_len, dtype=torch.long, device=device)

    return PrefixKVCache(keys=new_keys, values=new_values, mask=new_mask, lengths=new_lengths)


def mask_sink_token(cache: PrefixKVCache) -> PrefixKVCache:
    """Mask position 0 (attention sink) in prefix cache."""
    new_mask = cache.mask.clone()
    new_mask[:, 0] = False
    return PrefixKVCache(keys=cache.keys, values=cache.values, mask=new_mask,
                         lengths=cache.lengths)


def scale_visual_keys(cache: PrefixKVCache, visual_mask: np.ndarray, tau: float = 0.5) -> PrefixKVCache:
    """Scale visual keys by 1/sqrt(tau) to amplify attention logits (TACA-style)."""
    vis_idx = torch.tensor(np.where(visual_mask)[0], device=cache.keys.device)
    new_keys = cache.keys.clone()
    # Q @ K^T / sqrt(d) → scaling K by 1/sqrt(tau) amplifies logits by 1/sqrt(tau)
    new_keys[:, :, :, vis_idx, :] = new_keys[:, :, :, vis_idx, :] / (tau ** 0.5)
    return PrefixKVCache(keys=new_keys, values=cache.values, mask=cache.mask,
                         lengths=cache.lengths)


# ── Experiment runner ─────────────────────────────────────────────────────────

def get_attention_stats(model, batch, visual_mask_for_attn, state_mask_for_attn, text_mask_for_attn, prefix_len):
    """Get attention distribution using eager mode."""
    expert = model.flow_expert
    orig = expert.config._attn_implementation
    expert.config._attn_implementation = "eager"
    try:
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            result = infer_flow_action(model, clone_batch(batch), output_attentions=True)
    finally:
        expert.config._attn_implementation = orig

    last_step = result["expert_attention_weights"][-1]
    vis_shares, st_shares, txt_shares, act_shares, sink_shares = [], [], [], [], []

    for w in last_step:
        if w is None:
            continue
        w_mean = w[0].float().mean(dim=0)  # [action_len, kv_len]
        kv_len = w_mean.shape[1]
        actual_prefix = min(prefix_len, kv_len)
        prefix_w = w_mean[:, :actual_prefix]

        v_mask = torch.tensor(visual_mask_for_attn[:actual_prefix]) if len(visual_mask_for_attn) >= actual_prefix else torch.zeros(actual_prefix, dtype=torch.bool)
        s_mask = torch.tensor(state_mask_for_attn[:actual_prefix]) if len(state_mask_for_attn) >= actual_prefix else torch.zeros(actual_prefix, dtype=torch.bool)
        t_mask = torch.tensor(text_mask_for_attn[:actual_prefix]) if len(text_mask_for_attn) >= actual_prefix else torch.zeros(actual_prefix, dtype=torch.bool)

        if v_mask.any():
            vis_shares.append(prefix_w[:, v_mask].sum(dim=-1).mean().item())
        else:
            vis_shares.append(0.0)
        if s_mask.any():
            st_shares.append(prefix_w[:, s_mask].sum(dim=-1).mean().item())
        else:
            st_shares.append(0.0)
        if t_mask.any():
            txt_shares.append(prefix_w[:, t_mask].sum(dim=-1).mean().item())
        else:
            txt_shares.append(0.0)
        act_shares.append(w_mean[:, actual_prefix:].sum(dim=-1).mean().item())
        if actual_prefix > 0:
            sink_shares.append(prefix_w[:, 0].mean().item())

    return {
        "visual": float(np.mean(vis_shares)) if vis_shares else 0,
        "state": float(np.mean(st_shares)) if st_shares else 0,
        "text": float(np.mean(txt_shares)) if txt_shares else 0,
        "action": float(np.mean(act_shares)) if act_shares else 0,
        "sink_pos0": float(np.mean(sink_shares)) if sink_shares else 0,
    }


def run_experiment(
    config_path: str,
    batch: dict,
    original_visual_mask: np.ndarray,
    original_state_mask: np.ndarray,
    original_text_mask: np.ndarray,
    prefix_len: int,
    transform_fn,
    label: str,
    steps: int = 100,
    lr: float = 3e-4,
):
    """Run overfit with a prefix cache transformation applied at each forward."""
    device = batch["input_ids"].device

    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator,
    )
    model, _ = build_model_and_collator(config_path, device)
    model.knowledge_insulation = True
    model.train()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Hook into forward_backbone_stream to apply transformation
    original_fbs = model.forward_backbone_stream

    # We need to track the transformed prefix masks for attention analysis
    current_masks = {
        "visual": original_visual_mask,
        "state": original_state_mask,
        "text": original_text_mask,
        "prefix_len": prefix_len,
    }

    def patched_forward_backbone_stream(batch_arg, slot_embeds, output_attentions=False):
        output = original_fbs(batch_arg, slot_embeds, output_attentions=output_attentions)
        if output.prefix_cache is not None:
            result = transform_fn(output.prefix_cache, original_visual_mask)
            if isinstance(result, tuple):
                output.prefix_cache, new_visual, new_state, new_text, new_plen = result
                current_masks["visual"] = new_visual
                current_masks["state"] = new_state
                current_masks["text"] = new_text
                current_masks["prefix_len"] = new_plen
            else:
                output.prefix_cache = result
        return output

    model.forward_backbone_stream = patched_forward_backbone_stream

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    losses_history = []
    attn_snapshots = []

    for step in range(steps):
        optimizer.zero_grad()
        loss_dict = model.compute_loss(clone_batch(batch))
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        record = {k: float(v.item()) for k, v in loss_dict.items()}
        losses_history.append(record)

        if (step + 1) % 20 == 0 or step == 0:
            print(f"  Step {step+1:3d}: total={record['total_loss']:.4f}, "
                  f"flow={record['flow_loss']:.4f}, "
                  f"diff={record.get('diffusion_loss', 0):.4f}")

        if step in (0, steps // 4, steps // 2, 3 * steps // 4, steps - 1):
            model.eval()
            attn = get_attention_stats(
                model, batch,
                current_masks["visual"], current_masks["state"],
                current_masks["text"], current_masks["prefix_len"],
            )
            attn["step"] = step
            attn_snapshots.append(attn)
            print(f"    Attn @ step {step}: visual={attn['visual']:.4f}, "
                  f"state={attn['state']:.4f}, text={attn['text']:.4f}, "
                  f"sink={attn['sink_pos0']:.4f}")
            model.train()

    # Restore
    model.forward_backbone_stream = original_fbs

    return {
        "label": label,
        "losses": losses_history,
        "attention_snapshots": attn_snapshots,
    }


def plot_all(results: list[dict], out_dir: Path):
    plt = safe_import_plt()
    if plt is None:
        return

    n_exp = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Loss curves
    ax = axes[0, 0]
    for r in results:
        steps = range(1, len(r["losses"]) + 1)
        ax.plot(steps, [l["flow_loss"] for l in r["losses"]], label=r["label"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Flow Loss")
    ax.set_title("Flow Loss Convergence")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # Visual attention evolution
    ax = axes[0, 1]
    for r in results:
        snaps = r["attention_snapshots"]
        ax.plot([s["step"] for s in snaps], [s["visual"] for s in snaps],
                '-o', label=r["label"], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Visual attention share")
    ax.set_title("Visual Token Attention During Overfit")
    ax.legend(fontsize=8)

    # Sink attention
    ax = axes[1, 0]
    for r in results:
        snaps = r["attention_snapshots"]
        ax.plot([s["step"] for s in snaps], [s["sink_pos0"] for s in snaps],
                '-o', label=r["label"], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Sink (pos 0) attention share")
    ax.set_title("Attention Sink Evolution")
    ax.legend(fontsize=8)

    # Final breakdown
    ax = axes[1, 1]
    x = np.arange(n_exp)
    width = 0.18
    final_snaps = [r["attention_snapshots"][-1] for r in results]
    labels = [r["label"][:20] for r in results]
    types = ["visual", "state", "text", "action"]
    colors = ["green", "orange", "purple", "coral"]
    for i, (t, c) in enumerate(zip(types, colors)):
        vals = [s[t] for s in final_snaps]
        ax.bar(x + i * width, vals, width, label=t, color=c, alpha=0.8)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(labels, fontsize=7, rotation=15)
    ax.set_ylabel("Attention share")
    ax.set_title("Final Attention Breakdown (step 99)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "visual_fix_experiments.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved visual_fix_experiments.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--vla-shard", type=str, required=True)
    parser.add_argument("--normalizer-path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    out_dir = get_output_dir("visual_fix_experiments")
    device = "cuda"

    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator, load_hydra_config,
    )
    cfg = load_hydra_config(args.config_path)
    tmp_model, collator = build_model_and_collator(args.config_path, device)
    backbone = tmp_model.backbone

    dataset = build_vla_dataset(cfg, args.normalizer_path, args.vla_shard)
    samples = collect_samples(dataset, 2)
    batch = collate_to_device(collator, samples[:2], device)

    answer_start_idx = int(batch["answer_start_idx"][0].item())
    input_ids = batch["input_ids"][0, :answer_start_idx]
    visual_mask = (input_ids == backbone.video_token_id).cpu().numpy()
    state_mask = (input_ids == backbone.state_token_id).cpu().numpy()
    text_mask = ~(visual_mask | state_mask)
    prefix_len = answer_start_idx

    n_vis = int(visual_mask.sum())
    print(f"Prefix: {prefix_len} = {n_vis} visual + {state_mask.sum()} state + {text_mask.sum()} text")

    del tmp_model
    torch.cuda.empty_cache()

    results = []

    # A. Baseline
    def no_transform(cache, vis_mask):
        return cache
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        no_transform, "A: baseline", args.steps,
    ))
    torch.cuda.empty_cache()

    # B. Sink masking
    def sink_transform(cache, vis_mask):
        return mask_sink_token(cache)
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        sink_transform, "B: mask sink", args.steps,
    ))
    torch.cuda.empty_cache()

    # C. Visual pooling (432 → ~27 tokens)
    pool_sz = 16
    def pool_transform(cache, vis_mask):
        new_cache = pool_visual_kv(cache, vis_mask, pool_size=pool_sz)
        # Build new masks for the pooled sequence
        vis_indices = np.where(vis_mask)[0]
        non_vis_indices = np.where(~vis_mask)[0]
        n_pooled = (len(vis_indices) + pool_sz - 1) // pool_sz
        first_vis = vis_indices[0] if len(vis_indices) > 0 else 0
        n_non_vis_before = int((non_vis_indices < first_vis).sum())
        new_len = n_non_vis_before + n_pooled + len(non_vis_indices) - n_non_vis_before
        new_vis = np.zeros(new_len, dtype=bool)
        new_vis[n_non_vis_before:n_non_vis_before + n_pooled] = True
        new_state = np.zeros(new_len, dtype=bool)
        new_text = np.zeros(new_len, dtype=bool)
        # Map non-visual positions
        for i, orig_idx in enumerate(non_vis_indices):
            new_idx = i if orig_idx < first_vis else n_pooled + i
            if new_idx < new_len:
                new_state[new_idx] = state_mask[orig_idx]
                new_text[new_idx] = text_mask[orig_idx]
        return new_cache, new_vis, new_state, new_text, new_len
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        pool_transform, f"C: pool {n_vis}→{(n_vis + pool_sz - 1) // pool_sz}", args.steps,
    ))
    torch.cuda.empty_cache()

    # D. Visual temperature (tau=0.25, amplify visual K by 2x)
    def temp_transform(cache, vis_mask):
        return scale_visual_keys(cache, vis_mask, tau=0.25)
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        temp_transform, "D: visual temp τ=0.25", args.steps,
    ))
    torch.cuda.empty_cache()

    # E. Sink masking + visual temperature
    def sink_temp_transform(cache, vis_mask):
        cache = mask_sink_token(cache)
        return scale_visual_keys(cache, vis_mask, tau=0.25)
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        sink_temp_transform, "E: sink mask + temp", args.steps,
    ))
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in results:
        init_a = r["attention_snapshots"][0]
        final_a = r["attention_snapshots"][-1]
        final_l = r["losses"][-1]
        print(f"\n{r['label']}:")
        print(f"  Flow loss: {r['losses'][0]['flow_loss']:.4f} → {final_l['flow_loss']:.4f}")
        print(f"  Visual:  {init_a['visual']:.4f} → {final_a['visual']:.4f}")
        print(f"  State:   {init_a['state']:.4f} → {final_a['state']:.4f}")
        print(f"  Sink:    {init_a['sink_pos0']:.4f} → {final_a['sink_pos0']:.4f}")

    plot_all(results, out_dir)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results.json")


if __name__ == "__main__":
    main()
