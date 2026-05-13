"""
Long overfit test: 500 steps with best configurations.

Tests the top configurations from prior experiments with 5x longer training
to verify convergence behavior and whether attention patterns stabilize.

Experiments:
  A. Baseline (500 steps)
  B. KV projection + temperature (best from kv_proj_temp_experiment)
  C. Pool 432→27 + sink token (best from sink_token_experiment)
  D. KV projection + pool (good loss + high state attention)
  E. KV projection + pool + temperature (combine all)

Usage:
    python -m src.tests.full_chain_verification.long_overfit_experiment \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --vla-shard /path/to/shard.tar \
        --normalizer-path /path/to/normalizer.pkl \
        --steps 500
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
from src.tests.full_chain_verification.sink_token_experiment import (
    LearnableSinkToken,
)
from src.tests.full_chain_verification.kv_proj_temp_experiment import (
    PerHeadTemperature,
)
from src.tests.full_chain_verification.utils import get_output_dir, safe_import_plt

OmegaConf.register_new_resolver("eval", eval, replace=True)


def run_experiment(
    config_path: str,
    batch: dict,
    visual_mask: np.ndarray,
    state_mask: np.ndarray,
    text_mask: np.ndarray,
    prefix_len: int,
    label: str,
    use_kv_proj: bool = False,
    use_temp: bool = False,
    use_pool: bool = False,
    use_sink: bool = False,
    pool_size: int = 16,
    steps: int = 500,
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

    if use_kv_proj:
        for layer in model.flow_expert.layers:
            if hasattr(layer, 'prefix_key_proj') and isinstance(layer.prefix_key_proj, nn.Identity):
                head_dim = model.flow_expert.config.head_dim
                layer.prefix_key_proj = nn.Linear(head_dim, head_dim, bias=False).to(device=device, dtype=torch.bfloat16)
                layer.prefix_value_proj = nn.Linear(head_dim, head_dim, bias=False).to(device=device, dtype=torch.bfloat16)
                nn.init.eye_(layer.prefix_key_proj.weight)
                nn.init.eye_(layer.prefix_value_proj.weight)

    model.train()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    temp_module = None
    if use_temp:
        expert = model.flow_expert
        temp_module = PerHeadTemperature(expert.num_layers, expert.config.num_attention_heads).to(device)
        print(f"  Temperature params: {sum(p.numel() for p in temp_module.parameters())}")

    sink_module = None
    if use_sink:
        expert = model.flow_expert
        sink_module = LearnableSinkToken(
            expert.num_layers, expert.config.num_key_value_heads, expert.config.head_dim
        ).to(device)
        print(f"  Sink token params: {sum(p.numel() for p in sink_module.parameters())}")

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
        current_masks["visual"] = visual_mask.copy()
        current_masks["state"] = state_mask.copy()
        current_masks["text"] = text_mask.copy()
        current_masks["prefix_len"] = prefix_len

        if use_pool:
            cache = pool_visual_kv(cache, visual_mask, pool_size=pool_size)
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
            plen = current_masks["prefix_len"]
            for key in ("visual", "state", "text"):
                new_arr = np.zeros(plen + 1, dtype=bool)
                new_arr[1:] = current_masks[key][:plen]
                current_masks[key] = new_arr
            current_masks["prefix_len"] = plen + 1

        if use_temp and temp_module is not None:
            scaled_layers = []
            for layer_idx in range(cache.num_layers):
                scale = temp_module.get_key_scale(layer_idx)
                num_kv_heads = cache.keys.shape[2]
                num_heads = scale.shape[0]
                if num_heads != num_kv_heads:
                    groups = num_heads // num_kv_heads
                    scale = scale.view(num_kv_heads, groups).mean(dim=1)
                scale = scale.to(device=cache.keys.device, dtype=cache.keys.dtype)
                scaled_layers.append(cache.keys[layer_idx] * scale[None, :, None, None])
            new_keys = torch.stack(scaled_layers, dim=0)
            cache = PrefixKVCache(keys=new_keys, values=cache.values,
                                  mask=cache.mask, lengths=cache.lengths)

        output.prefix_cache = cache
        return output

    model.forward_backbone_stream = patched_forward_backbone_stream

    trainable = [p for p in model.parameters() if p.requires_grad]
    if temp_module is not None:
        trainable.extend(temp_module.parameters())
    if sink_module is not None:
        trainable.extend(sink_module.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    losses_history = []
    attn_snapshots = []

    # Measure attention at more frequent intervals for 500 steps
    attn_steps = set([0, 49, 99, 149, 199, 249, 299, 349, 399, 449, 499])
    attn_steps = {s for s in attn_steps if s < steps}

    for step in range(steps):
        optimizer.zero_grad()
        loss_dict = model.compute_loss(clone_batch(batch))
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        record = {k: float(v.item()) for k, v in loss_dict.items()}
        losses_history.append(record)

        if (step + 1) % 50 == 0 or step == 0:
            print(f"  Step {step+1:3d}: total={record['total_loss']:.4f}, "
                  f"flow={record['flow_loss']:.4f}")

        if step in attn_steps:
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
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    out_dir = get_output_dir("long_overfit_experiment")
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

    # B: KV proj + temp (best absolute loss)
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "B: kvproj+temp", use_kv_proj=True, use_temp=True, steps=args.steps,
    ))
    torch.cuda.empty_cache()

    # C: Pool + sink (best in sink experiment)
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "C: pool+sink", use_pool=True, use_sink=True, steps=args.steps,
    ))
    torch.cuda.empty_cache()

    # D: KV proj + pool (good loss + high state attention)
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "D: kvproj+pool", use_kv_proj=True, use_pool=True, steps=args.steps,
    ))
    torch.cuda.empty_cache()

    # E: All combined
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "E: all combined", use_kv_proj=True, use_temp=True, use_pool=True, use_sink=True,
        steps=args.steps,
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
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Flow loss convergence (log scale)
        ax = axes[0, 0]
        for r in results:
            ax.plot([l["flow_loss"] for l in r["losses"]], label=r["label"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Flow Loss")
        ax.set_title("Flow Loss Convergence (500 steps)")
        ax.legend(fontsize=7)
        ax.set_yscale("log")

        # Visual attention evolution
        ax = axes[0, 1]
        for r in results:
            snaps = r["attention_snapshots"]
            ax.plot([s["step"] for s in snaps], [s["visual"] for s in snaps],
                    '-o', label=r["label"], linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Visual attention share")
        ax.set_title("Visual Token Attention (500 steps)")
        ax.legend(fontsize=7)

        # State attention evolution
        ax = axes[1, 0]
        for r in results:
            snaps = r["attention_snapshots"]
            ax.plot([s["step"] for s in snaps], [s["state"] for s in snaps],
                    '-o', label=r["label"], linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("State attention share")
        ax.set_title("State Token Attention (500 steps)")
        ax.legend(fontsize=7)

        # Final breakdown
        ax = axes[1, 1]
        x = np.arange(len(results))
        width = 0.18
        final_snaps = [r["attention_snapshots"][-1] for r in results]
        labels_short = [r["label"][:18] for r in results]
        for i, (t, c) in enumerate(zip(["visual", "state", "text", "action"],
                                        ["green", "orange", "purple", "coral"])):
            vals = [s[t] for s in final_snaps]
            ax.bar(x + i * width, vals, width, label=t, color=c, alpha=0.8)
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(labels_short, fontsize=7, rotation=20)
        ax.set_ylabel("Attention share")
        ax.set_title("Final Attention Breakdown (step 499)")
        ax.legend()

        fig.tight_layout()
        fig.savefig(out_dir / "long_overfit_results.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved long_overfit_results.png")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results.json")


if __name__ == "__main__":
    main()
