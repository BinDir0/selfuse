"""
Test learnable sink token + visual pooling combination.

Based on "One Token Is Enough" (arXiv:2601.19657):
Add a learnable sink token to the prefix cache that absorbs excess attention,
freeing up attention budget for meaningful tokens (visual, state, text).

Combined with visual pooling to address both:
  - Attention sink problem (dedicated sink absorbs garbage attention)
  - Visual token dilution (432 → 27 pooled tokens)

Usage:
    python -m src.tests.full_chain_verification.sink_token_experiment \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --vla-shard /path/to/shard.tar \
        --normalizer-path /path/to/normalizer.pkl \
        --steps 100
"""

from __future__ import annotations

import argparse
import json

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
from src.tests.full_chain_verification.visual_fix_experiments import (
    pool_visual_kv,
    get_attention_stats,
)
from src.tests.full_chain_verification.utils import get_output_dir, safe_import_plt

OmegaConf.register_new_resolver("eval", eval, replace=True)


class LearnableSinkToken(nn.Module):
    """A single learnable sink token prepended to the prefix cache.

    Reference: "One Token Is Enough" (arXiv:2601.19657)
    The sink token absorbs excess attention that would otherwise go to
    meaningless positions (like <|im_start|>). All other tokens can
    attend to it, but it only attends to itself.
    """
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, dtype=torch.bfloat16):
        super().__init__()
        # Learnable K/V for the sink, shared across batch
        self.sink_key = nn.Parameter(torch.randn(num_layers, 1, num_kv_heads, 1, head_dim, dtype=dtype) * 0.02)
        self.sink_value = nn.Parameter(torch.zeros(num_layers, 1, num_kv_heads, 1, head_dim, dtype=dtype))

    def prepend_to_cache(self, cache: PrefixKVCache) -> PrefixKVCache:
        """Prepend sink token to prefix cache."""
        B = cache.keys.shape[1]
        sink_k = self.sink_key.expand(-1, B, -1, -1, -1)
        sink_v = self.sink_value.expand(-1, B, -1, -1, -1)

        new_keys = torch.cat([sink_k, cache.keys], dim=3)
        new_values = torch.cat([sink_v, cache.values], dim=3)

        # Mask: sink is True (visible), then original mask
        sink_mask = torch.ones(B, 1, dtype=torch.bool, device=cache.mask.device)
        new_mask = torch.cat([sink_mask, cache.mask], dim=1)
        new_lengths = cache.lengths + 1

        return PrefixKVCache(keys=new_keys, values=new_values, mask=new_mask, lengths=new_lengths)


def run_experiment(
    config_path: str,
    batch: dict,
    visual_mask: np.ndarray,
    state_mask: np.ndarray,
    text_mask: np.ndarray,
    prefix_len: int,
    label: str,
    use_sink: bool = False,
    use_pool: bool = False,
    pool_size: int = 16,
    steps: int = 100,
    lr: float = 3e-4,
):
    device = batch["input_ids"].device

    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator,
    )
    model, _ = build_model_and_collator(config_path, device)
    # Detach prefix KV in all experts to prevent backbone gradient flow.
    model.flow_expert.detach_prefix_kv = True
    if getattr(model, "world_model_expert", None) is not None:
        model.world_model_expert.detach_prefix_kv = True
    model.train()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Create sink token module if needed
    sink_module = None
    if use_sink:
        expert = model.flow_expert
        num_layers = expert.num_layers
        num_kv_heads = expert.config.num_key_value_heads
        head_dim = expert.config.head_dim
        sink_module = LearnableSinkToken(num_layers, num_kv_heads, head_dim).to(device)
        print(f"  Sink token params: {sum(p.numel() for p in sink_module.parameters())}")

    # Track current masks for attention analysis
    current_masks = {
        "visual": visual_mask.copy(),
        "state": state_mask.copy(),
        "text": text_mask.copy(),
        "prefix_len": prefix_len,
    }

    original_fbs = model.forward_backbone_stream

    def patched_forward_backbone_stream(batch_arg, slot_embeds, output_attentions=False):
        output = original_fbs(batch_arg, slot_embeds, output_attentions=output_attentions)
        if output.prefix_cache is None:
            return output

        cache = output.prefix_cache

        if use_pool:
            cache = pool_visual_kv(cache, visual_mask, pool_size=pool_size)
            # Update masks for pooled version
            vis_indices = np.where(visual_mask)[0]
            non_vis_indices = np.where(~visual_mask)[0]
            n_pooled = (len(vis_indices) + pool_size - 1) // pool_size
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
            current_masks["visual"] = new_vis
            current_masks["state"] = new_state
            current_masks["text"] = new_text
            current_masks["prefix_len"] = new_len

        if use_sink and sink_module is not None:
            cache = sink_module.prepend_to_cache(cache)
            # Shift masks: add sink at position 0
            plen = current_masks["prefix_len"]
            new_vis = np.zeros(plen + 1, dtype=bool)
            new_vis[1:] = current_masks["visual"][:plen]
            new_state = np.zeros(plen + 1, dtype=bool)
            new_state[1:] = current_masks["state"][:plen]
            new_text = np.zeros(plen + 1, dtype=bool)
            new_text[1:] = current_masks["text"][:plen]
            # Position 0 is the sink — mark as text for simplicity
            current_masks["visual"] = new_vis
            current_masks["state"] = new_state
            current_masks["text"] = new_text
            current_masks["prefix_len"] = plen + 1

        output.prefix_cache = cache
        return output

    model.forward_backbone_stream = patched_forward_backbone_stream

    # Collect trainable params (include sink if present)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if sink_module is not None:
        trainable.extend(sink_module.parameters())
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
                  f"flow={record['flow_loss']:.4f}")

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

    model.forward_backbone_stream = original_fbs

    return {
        "label": label,
        "losses": losses_history,
        "attention_snapshots": attn_snapshots,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--vla-shard", type=str, required=True)
    parser.add_argument("--normalizer-path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    out_dir = get_output_dir("sink_token_experiment")
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

    # A: Baseline
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "A: baseline", steps=args.steps,
    ))
    torch.cuda.empty_cache()

    # B: Pool only
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "B: pool 432→27", use_pool=True, pool_size=16, steps=args.steps,
    ))
    torch.cuda.empty_cache()

    # C: Sink token only
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "C: sink token", use_sink=True, steps=args.steps,
    ))
    torch.cuda.empty_cache()

    # D: Pool + Sink token
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "D: pool + sink", use_pool=True, pool_size=16, use_sink=True, steps=args.steps,
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

    # Plot
    plt = safe_import_plt()
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Visual attention
        ax = axes[0]
        for r in results:
            snaps = r["attention_snapshots"]
            ax.plot([s["step"] for s in snaps], [s["visual"] for s in snaps],
                    '-o', label=r["label"], linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Visual attention share")
        ax.set_title("Visual Token Attention")
        ax.legend(fontsize=8)

        # Final breakdown
        ax = axes[1]
        x = np.arange(len(results))
        width = 0.2
        final_snaps = [r["attention_snapshots"][-1] for r in results]
        labels_short = [r["label"][:15] for r in results]
        for i, (t, c) in enumerate(zip(["visual", "state", "text", "action"],
                                        ["green", "orange", "purple", "coral"])):
            vals = [s[t] for s in final_snaps]
            ax.bar(x + i * width, vals, width, label=t, color=c, alpha=0.8)
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(labels_short, fontsize=8)
        ax.set_ylabel("Attention share")
        ax.set_title("Final Attention Breakdown")
        ax.legend()

        # Loss curves
        ax = axes[2]
        for r in results:
            ax.plot([l["flow_loss"] for l in r["losses"]], label=r["label"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Flow Loss")
        ax.set_title("Flow Loss Convergence")
        ax.legend(fontsize=8)
        ax.set_yscale("log")

        fig.tight_layout()
        fig.savefig(out_dir / "sink_token_results.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved sink_token_results.png")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results.json")


if __name__ == "__main__":
    main()
