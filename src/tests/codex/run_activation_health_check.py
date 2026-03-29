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
from src.tests.debug_dataloader_batch import build_dataloader, load_config, prepare_dataset
from src.tests.test_legendvla_hooks import run_with_hooks, visualize_stats
from src.utils.pytorch_util import dict_apply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run activation and parameter health checks on real LegendVLA batches.")
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


def forward_train(model: LegendVLA, batch: dict[str, Any]) -> Any:
    return model("train", batch)


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

    aggregate = {"batches": []}
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= args.max_batches:
            break

        batch = move_batch_to_device(batch, device=device, dtype=dtype)
        batch_dir = output_dir / f"batch_{batch_index:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        stats = run_with_hooks(
            model=model,
            batch=batch,
            forward_fn=forward_train,
            activations_output_path=str(batch_dir / "activations.json"),
            parameters_output_path=str(batch_dir / "parameters.json"),
        )
        visualize_stats(stats, str(batch_dir / "plots"))
        aggregate["batches"].append({"batch_index": batch_index, "output_dir": str(batch_dir)})

    (output_dir / "activation_report.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
