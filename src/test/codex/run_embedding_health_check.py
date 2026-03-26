#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import hydra
import numpy as np
import torch

from src.policy.legendvla import LegendVLA
from src.test.debug_dataloader_batch import build_dataloader, load_config, prepare_dataset
from src.utils.embedding_analysis import (
    analyze_embedding_distribution,
    plot_embedding_l2norm_by_position,
    plot_tsne_from_npzs,
)
from src.utils.pytorch_util import dict_apply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding health checks on real LegendVLA batches.")
    parser.add_argument("--config", type=str, default="src/config/experiment/legendvla_qwen3_vl.yaml")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_count", type=int, default=2)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--dataset_kind", type=str, default="vla", choices=["vla", "unified"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--normalizer_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=1)
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    def move(value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        if value.dtype.is_floating_point:
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)

    return dict_apply(batch, move)


def summarize_sequence_embeddings(name: str, embeddings: torch.Tensor) -> dict[str, Any]:
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    norms = torch.linalg.norm(embeddings, dim=-1)
    cosine_adjacent = None
    if embeddings.ndim == 3 and embeddings.shape[1] > 1:
        left = embeddings[:, :-1]
        right = embeddings[:, 1:]
        cosine_adjacent = torch.nn.functional.cosine_similarity(left, right, dim=-1).mean().item()

    singular_values = torch.linalg.svdvals(flat.float())
    probs = singular_values / singular_values.sum().clamp(min=1e-12)
    effective_rank = float(torch.exp(-(probs * torch.log(probs.clamp(min=1e-12))).sum()).item())

    return {
        "name": name,
        "shape": list(embeddings.shape),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "finite": bool(torch.isfinite(embeddings).all().item()),
        "adjacent_cosine_mean": cosine_adjacent,
        "effective_rank": effective_rank,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(pathlib.Path(args.config))
    dataset, _, _, _ = prepare_dataset(cfg, args)
    dataloader = build_dataloader(cfg, dataset, args)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model: LegendVLA = hydra.utils.instantiate(cfg.policy)
    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        for key in ["model", "module", "model_state_dict"]:
            if key in state_dict:
                state_dict = state_dict[key]
                break
        model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
    model.eval()

    npz_paths: list[pathlib.Path] = []
    report = {"batches": []}

    for batch_index, batch in enumerate(dataloader):
        if batch_index >= args.max_batches:
            break

        batch = move_batch_to_device(batch, device=device, dtype=dtype)
        slot_embeds = model.build_slot_embeddings(batch, add_action_noise=False)
        backbone_output = model.forward_backbone_stream(batch, slot_embeds)

        embedding_map = {
            "state_slot_embeds": slot_embeds["state"].detach().float().cpu(),
            "action_slot_embeds": slot_embeds["action"].detach().float().cpu(),
            "last_hidden_states": backbone_output.last_hidden_states.detach().float().cpu(),
        }

        batch_dir = output_dir / f"batch_{batch_index:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        npz_path = batch_dir / "embeddings.npz"
        np.savez(npz_path, **{key: value.numpy() for key, value in embedding_map.items()})
        npz_paths.append(npz_path)

        batch_report = {}
        for name, value in embedding_map.items():
            plot_embedding_l2norm_by_position(value, save_dir=batch_dir / f"{name}_norms")
            stats = analyze_embedding_distribution(value.reshape(-1, value.shape[-1]), plot=True, save_path=str(batch_dir / f"{name}_distribution.png"))
            batch_report[name] = {
                "summary": summarize_sequence_embeddings(name, value),
                "distribution": stats,
            }

        report["batches"].append(batch_report)

    if npz_paths:
        plot_tsne_from_npzs(
            npz_paths=npz_paths,
            keys=["state_slot_embeds", "action_slot_embeds", "last_hidden_states"],
            sample_per_npz=128,
            save_path=output_dir / "embedding_tsne.png",
            show=False,
        )

    (output_dir / "embedding_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

