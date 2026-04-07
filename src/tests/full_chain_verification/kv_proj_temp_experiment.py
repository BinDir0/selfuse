"""
Test KV projection and per-head learnable temperature for attention alignment.

Experiments:
  A. Baseline (knowledge_insulation=True, as-is)
  B. use_kv_projection=True (identity-initialized linear projection on prefix K/V)
  C. Per-head learnable temperature on action expert attention
  D. KV projection + learnable temperature
  E. KV projection + pool 432→27
  F. KV projection + pool + sink token

Usage:
    python -m src.tests.full_chain_verification.kv_proj_temp_experiment \
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
from src.tests.full_chain_verification.sink_token_experiment import (
    LearnableSinkToken,
)
from src.tests.full_chain_verification.utils import get_output_dir, safe_import_plt

OmegaConf.register_new_resolver("eval", eval, replace=True)


class PerHeadTemperature(nn.Module):
    """Per-head learnable temperature for action expert attention.

    Reference: https://nickcdryan.com/2024/08/02/introducing-a-learnable-temperature-value-into-the-self-attention-scores/
    Scales attention logits by 1/temp per head. Initialized at 1.0 (identity).
    """
    def __init__(self, num_layers: int, num_heads: int, init_temp: float = 1.0):
        super().__init__()
        # log-space to keep temperature positive
        self.log_temps = nn.Parameter(
            torch.full((num_layers, num_heads), float(np.log(init_temp)))
        )

    def get_key_scale(self, layer_idx: int) -> torch.Tensor:
        """Return per-head K scale factor: 1/sqrt(temp) for each head.

        Scaling K by 1/sqrt(temp) is equivalent to dividing attention logits
        by temp (since QK^T scales linearly with K).
        """
        temps = self.log_temps[layer_idx].exp()  # [num_heads]
        return 1.0 / temps.sqrt()


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
    steps: int = 100,
    lr: float = 3e-4,
):
    device = batch["input_ids"].device

    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator,
    )
    model, _ = build_model_and_collator(config_path, device)
    model.knowledge_insulation = True

    # Enable KV projection if requested
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

    # Create auxiliary modules
    temp_module = None
    if use_temp:
        expert = model.flow_expert
        num_layers = expert.num_layers
        num_heads = expert.config.num_attention_heads
        temp_module = PerHeadTemperature(num_layers, num_heads).to(device)
        print(f"  Temperature params: {sum(p.numel() for p in temp_module.parameters())}")

    sink_module = None
    if use_sink:
        expert = model.flow_expert
        num_layers = expert.num_layers
        num_kv_heads = expert.config.num_key_value_heads
        head_dim = expert.config.head_dim
        sink_module = LearnableSinkToken(num_layers, num_kv_heads, head_dim).to(device)
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

        # Reset masks
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
            new_vis = np.zeros(plen + 1, dtype=bool)
            new_vis[1:] = current_masks["visual"][:plen]
            new_state = np.zeros(plen + 1, dtype=bool)
            new_state[1:] = current_masks["state"][:plen]
            new_text = np.zeros(plen + 1, dtype=bool)
            new_text[1:] = current_masks["text"][:plen]
            current_masks["visual"] = new_vis
            current_masks["state"] = new_state
            current_masks["text"] = new_text
            current_masks["prefix_len"] = plen + 1

        if use_temp and temp_module is not None:
            # Scale prefix K by per-head temperature factor
            # keys: [num_layers, B, num_kv_heads, kv_seq_len, head_dim]
            scaled_layers = []
            for layer_idx in range(cache.num_layers):
                scale = temp_module.get_key_scale(layer_idx)  # [num_heads]
                # GQA: num_heads may differ from num_kv_heads
                num_kv_heads = cache.keys.shape[2]
                num_heads = scale.shape[0]
                if num_heads != num_kv_heads:
                    groups = num_heads // num_kv_heads
                    scale = scale.view(num_kv_heads, groups).mean(dim=1)
                # scale: [num_kv_heads] -> [1, num_kv_heads, 1, 1]
                scale = scale.to(device=cache.keys.device, dtype=cache.keys.dtype)
                scaled_layers.append(cache.keys[layer_idx] * scale[None, :, None, None])
            new_keys = torch.stack(scaled_layers, dim=0)
            cache = PrefixKVCache(keys=new_keys, values=cache.values,
                                  mask=cache.mask, lengths=cache.lengths)

        output.prefix_cache = cache
        return output

    model.forward_backbone_stream = patched_forward_backbone_stream

    # Collect trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    if temp_module is not None:
        trainable.extend(temp_module.parameters())
    if sink_module is not None:
        trainable.extend(sink_module.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    losses_history = []
    attn_snapshots = []
    temp_snapshots = []

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

            if temp_module is not None:
                temps = temp_module.log_temps.exp().detach().cpu()
                temp_snap = {
                    "step": step,
                    "mean": float(temps.mean()),
                    "min": float(temps.min()),
                    "max": float(temps.max()),
                    "per_layer_mean": temps.mean(dim=1).tolist(),
                }
                temp_snapshots.append(temp_snap)
                print(f"    Temp @ step {step}: mean={temp_snap['mean']:.3f}, "
                      f"min={temp_snap['min']:.3f}, max={temp_snap['max']:.3f}")

            model.train()

    model.forward_backbone_stream = original_fbs

    return {
        "label": label,
        "losses": losses_history,
        "attention_snapshots": attn_snapshots,
        "temperature_snapshots": temp_snapshots,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--vla-shard", type=str, required=True)
    parser.add_argument("--normalizer-path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    out_dir = get_output_dir("kv_proj_temp_experiment")
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

    # B: KV projection only
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "B: kv_projection", use_kv_proj=True, steps=args.steps,
    ))
    torch.cuda.empty_cache()

    # C: Per-head learnable temperature only
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "C: learnable_temp", use_temp=True, steps=args.steps,
    ))
    torch.cuda.empty_cache()

    # D: KV projection + temperature
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "D: kvproj+temp", use_kv_proj=True, use_temp=True, steps=args.steps,
    ))
    torch.cuda.empty_cache()

    # E: KV projection + pool
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "E: kvproj+pool", use_kv_proj=True, use_pool=True, steps=args.steps,
    ))
    torch.cuda.empty_cache()

    # F: KV projection + pool + sink
    results.append(run_experiment(
        args.config_path, batch, visual_mask, state_mask, text_mask, prefix_len,
        "F: kvproj+pool+sink", use_kv_proj=True, use_pool=True, use_sink=True, steps=args.steps,
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
        if r["temperature_snapshots"]:
            t = r["temperature_snapshots"][-1]
            print(f"  Temp:    mean={t['mean']:.3f}, min={t['min']:.3f}, max={t['max']:.3f}")

    # Plot
    plt = safe_import_plt()
    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Flow loss convergence
        ax = axes[0, 0]
        for r in results:
            ax.plot([l["flow_loss"] for l in r["losses"]], label=r["label"])
        ax.set_xlabel("Step")
        ax.set_ylabel("Flow Loss")
        ax.set_title("Flow Loss Convergence")
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
        ax.set_title("Visual Token Attention")
        ax.legend(fontsize=7)

        # Final attention breakdown
        ax = axes[1, 0]
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
        ax.set_title("Final Attention Breakdown")
        ax.legend()

        # Temperature evolution (for experiments that have it)
        ax = axes[1, 1]
        has_temp = False
        for r in results:
            if r["temperature_snapshots"]:
                has_temp = True
                snaps = r["temperature_snapshots"]
                ax.plot([s["step"] for s in snaps], [s["mean"] for s in snaps],
                        '-o', label=f"{r['label']} mean", linewidth=2)
                ax.fill_between(
                    [s["step"] for s in snaps],
                    [s["min"] for s in snaps],
                    [s["max"] for s in snaps],
                    alpha=0.2,
                )
        if has_temp:
            ax.set_xlabel("Step")
            ax.set_ylabel("Temperature")
            ax.set_title("Learned Temperature (min/mean/max)")
            ax.legend(fontsize=7)
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No temperature experiments", ha='center', va='center',
                    transform=ax.transAxes)

        fig.tight_layout()
        fig.savefig(out_dir / "kv_proj_temp_results.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved kv_proj_temp_results.png")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results.json")


if __name__ == "__main__":
    main()
