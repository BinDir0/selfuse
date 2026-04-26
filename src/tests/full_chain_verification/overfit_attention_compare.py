"""
Quick overfit experiment: detach_prefix_kv ON vs OFF.

Overfits a single batch for N steps, then compares:
  1. Loss convergence (total, flow, diffusion)
  2. Attention weight distribution (visual vs text vs state)
  3. AdaLN gate magnitudes

This tells us whether allowing gradients to flow through prefix_cache
changes the attention pattern on visual tokens.

Usage:
    python -m src.tests.full_chain_verification.overfit_attention_compare \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --vla-shard /path/to/shard.tar \
        --normalizer-path /path/to/normalizer.pkl \
        --steps 100
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tests.full_chain_verification.part9_input_ablation import (
    build_vla_dataset,
    clone_batch,
    collect_samples,
    collate_to_device,
)
from src.tests.full_chain_verification.utils import get_output_dir, safe_import_plt
from src.policy.legendvla_inference import infer_flow_action

OmegaConf.register_new_resolver("eval", eval, replace=True)


def get_attention_stats(model, batch, prefix_len, visual_mask, state_mask, text_mask):
    """Run inference with eager attention and return per-type attention shares."""
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

    # Use last ODE step, average across layers
    last_step = result["expert_attention_weights"][-1]
    visual_shares, state_shares, text_shares, action_shares = [], [], [], []
    sink_shares = []

    for w in last_step:
        if w is None:
            continue
        w_mean = w[0].float().mean(dim=0)  # [action_len, kv_len]
        prefix_w = w_mean[:, :prefix_len]
        visual_shares.append(prefix_w[:, torch.tensor(visual_mask)].sum(dim=-1).mean().item())
        state_shares.append(prefix_w[:, torch.tensor(state_mask)].sum(dim=-1).mean().item())
        text_shares.append(prefix_w[:, torch.tensor(text_mask)].sum(dim=-1).mean().item())
        action_shares.append(w_mean[:, prefix_len:].sum(dim=-1).mean().item())
        # Attention to position 0 (sink)
        sink_shares.append(prefix_w[:, 0].mean().item())

    return {
        "visual": float(np.mean(visual_shares)),
        "state": float(np.mean(state_shares)),
        "text": float(np.mean(text_shares)),
        "action": float(np.mean(action_shares)),
        "sink_pos0": float(np.mean(sink_shares)),
    }


def get_adaln_gate_stats(model):
    """Get mean absolute gate values from AdaLN-Zero layers."""
    gate_norms = []
    for layer in model.flow_expert.layers:
        # AdaLN modulation weight: last 1/3 of output channels = gate
        w = layer.attn_adaln.modulation.weight.detach()
        b = layer.attn_adaln.modulation.bias.detach()
        dim = w.shape[1]
        # Gate is the last chunk of 3
        gate_w = w[2 * dim:]
        gate_b = b[2 * dim:]
        gate_norms.append({
            "weight_norm": float(gate_w.norm().item()),
            "bias_mean": float(gate_b.mean().item()),
            "bias_abs_mean": float(gate_b.abs().mean().item()),
        })
    return gate_norms


def overfit_one_config(
    config_path: str,
    batch: dict,
    prefix_len: int,
    visual_mask,
    state_mask,
    text_mask,
    detach_prefix_kv,
    steps: int,
    lr: float = 3e-4,
    label: str = "",
):
    """Overfit a single batch with given detach_prefix_kv setting."""
    device = batch["input_ids"].device

    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator, load_hydra_config,
    )
    model, _ = build_model_and_collator(config_path, device)
    # Detach prefix KV in all experts to prevent backbone gradient flow.
    model.flow_expert.detach_prefix_kv = detach_prefix_kv
    if getattr(model, "world_model_expert", None) is not None:
        model.world_model_expert.detach_prefix_kv = detach_prefix_kv
    model.train()

    print(f"\n{'='*60}")
    print(f"  {label}: detach_prefix_kv={detach_prefix_kv}")
    print(f"  Steps={steps}, lr={lr}")
    print(f"{'='*60}")

    # Collect trainable params
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

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step+1:3d}: total={record['total_loss']:.4f}, "
                  f"flow={record['flow_loss']:.4f}, "
                  f"diff={record.get('diffusion_loss', 0):.4f}")

        # Attention snapshots at step 0, 25, 50, 75, 100
        if step in (0, steps // 4, steps // 2, 3 * steps // 4, steps - 1):
            model.eval()
            attn = get_attention_stats(model, batch, prefix_len, visual_mask, state_mask, text_mask)
            attn["step"] = step
            attn_snapshots.append(attn)
            print(f"    Attn @ step {step}: visual={attn['visual']:.4f}, "
                  f"state={attn['state']:.4f}, text={attn['text']:.4f}, "
                  f"sink={attn['sink_pos0']:.4f}")
            model.train()

    # Final gate stats
    gate_stats = get_adaln_gate_stats(model)
    mean_gate_bias = float(np.mean([g["bias_abs_mean"] for g in gate_stats]))
    print(f"  Final AdaLN gate |bias| mean: {mean_gate_bias:.6f}")

    return {
        "label": label,
        "detach_prefix_kv": detach_prefix_kv,
        "losses": losses_history,
        "attention_snapshots": attn_snapshots,
        "gate_stats": gate_stats,
    }


def plot_comparison(results: list[dict], out_dir: Path):
    plt = safe_import_plt()
    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Loss curves
    ax = axes[0, 0]
    for r in results:
        steps = range(1, len(r["losses"]) + 1)
        ax.plot(steps, [l["total_loss"] for l in r["losses"]], label=f'{r["label"]} total')
        ax.plot(steps, [l["flow_loss"] for l in r["losses"]], '--', label=f'{r["label"]} flow', alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Convergence")
    ax.legend()
    ax.set_yscale("log")

    # Plot 2: Visual attention evolution
    ax = axes[0, 1]
    for r in results:
        snaps = r["attention_snapshots"]
        ax.plot([s["step"] for s in snaps], [s["visual"] for s in snaps],
                '-o', label=f'{r["label"]}', linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Visual attention share")
    ax.set_title("Visual Token Attention During Overfit")
    ax.legend()

    # Plot 3: Text attention evolution
    ax = axes[1, 0]
    for r in results:
        snaps = r["attention_snapshots"]
        ax.plot([s["step"] for s in snaps], [s["text"] for s in snaps],
                '-o', label=f'{r["label"]} text', linewidth=2)
        ax.plot([s["step"] for s in snaps], [s["sink_pos0"] for s in snaps],
                '--s', label=f'{r["label"]} sink(pos0)', linewidth=1, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Attention share")
    ax.set_title("Text & Sink Attention During Overfit")
    ax.legend()

    # Plot 4: Final attention breakdown comparison
    ax = axes[1, 1]
    x = np.arange(len(results))
    width = 0.15
    final_snaps = [r["attention_snapshots"][-1] for r in results]
    labels = [r["label"] for r in results]
    types = ["visual", "state", "text", "action", "sink_pos0"]
    colors = ["green", "orange", "purple", "coral", "gray"]
    for i, (t, c) in enumerate(zip(types, colors)):
        vals = [s[t] for s in final_snaps]
        ax.bar(x + i * width, vals, width, label=t, color=c, alpha=0.8)
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Attention share")
    ax.set_title("Final Attention Breakdown")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "overfit_attention_compare.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved overfit_attention_compare.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--vla-shard", type=str, required=True)
    parser.add_argument("--normalizer-path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    out_dir = get_output_dir("overfit_attention_compare")
    device = "cuda"

    # Build dataset and batch
    from src.tests.full_chain_verification.part3_backbone_prefix_cache import (
        build_model_and_collator, load_hydra_config,
    )
    cfg = load_hydra_config(args.config_path)

    # We need a temporary model just to get the collator and tokenizer
    tmp_model, collator = build_model_and_collator(args.config_path, device)
    backbone = tmp_model.backbone

    dataset = build_vla_dataset(cfg, args.normalizer_path, args.vla_shard)
    samples = collect_samples(dataset, 2)
    batch = collate_to_device(collator, samples[:2], device)

    # Token classification
    answer_start_idx = int(batch["answer_start_idx"][0].item())
    input_ids = batch["input_ids"][0, :answer_start_idx]
    visual_mask = (input_ids == backbone.video_token_id).cpu().numpy()
    state_mask = (input_ids == backbone.state_token_id).cpu().numpy()
    text_mask = ~(visual_mask | state_mask)
    prefix_len = answer_start_idx

    print(f"Prefix: {prefix_len} = {visual_mask.sum()} visual + {state_mask.sum()} state + {text_mask.sum()} text")

    del tmp_model
    torch.cuda.empty_cache()

    # Run experiments
    results = []

    # Experiment 1: detach_prefix_kv=True (current config)
    r1 = overfit_one_config(
        args.config_path, batch, prefix_len, visual_mask, state_mask, text_mask,
        detach_prefix_kv=True, steps=args.steps, lr=args.lr,
        label="detach=True",
    )
    results.append(r1)
    torch.cuda.empty_cache()

    # Experiment 2: detach_prefix_kv=False
    r2 = overfit_one_config(
        args.config_path, batch, prefix_len, visual_mask, state_mask, text_mask,
        detach_prefix_kv=False, steps=args.steps, lr=args.lr,
        label="detach=False",
    )
    results.append(r2)
    torch.cuda.empty_cache()

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    for r in results:
        final_attn = r["attention_snapshots"][-1]
        init_attn = r["attention_snapshots"][0]
        final_loss = r["losses"][-1]
        print(f"\n{r['label']}:")
        print(f"  Final loss: total={final_loss['total_loss']:.4f}, "
              f"flow={final_loss['flow_loss']:.4f}")
        print(f"  Visual attn: {init_attn['visual']:.4f} → {final_attn['visual']:.4f} "
              f"(change: {final_attn['visual'] - init_attn['visual']:+.4f})")
        print(f"  Sink attn:   {init_attn['sink_pos0']:.4f} → {final_attn['sink_pos0']:.4f}")

    plot_comparison(results, out_dir)

    # Save raw data
    for r in results:
        for snap in r["attention_snapshots"]:
            for k, v in snap.items():
                if isinstance(v, np.floating):
                    snap[k] = float(v)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results.json")


if __name__ == "__main__":
    main()
