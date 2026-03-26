#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import torch

from src.test.debug_dataloader_batch import (
    build_dataloader,
    build_sample_export,
    load_config,
    prepare_dataset,
    select_normalizer_fields,
    write_html_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one real dataloader batch with contract checks.")
    parser.add_argument("--config", type=str, default="src/config/experiment/legendvla_qwen3_vl.yaml")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_count", type=int, default=4)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--dataset_kind", type=str, default="vla", choices=["vla", "unified"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--normalizer_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def tensor_to_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: tensor_to_list(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_to_list(item) for item in value]
    return value


def validate_batch_contract(batch: dict[str, Any], collator) -> dict[str, Any]:
    tokenizer = collator.batch_processor.tokenizer
    ignore_index = collator.ignore_index
    state_token_id = int(tokenizer.convert_tokens_to_ids(collator.formatter.state_token))
    action_token_id = int(tokenizer.convert_tokens_to_ids(collator.formatter.action_token))

    results = []
    failures = []
    batch_size = int(batch["input_ids"].shape[0])

    for sample_idx in range(batch_size):
        attention_mask = batch["attention_mask"][sample_idx].bool()
        valid_input_ids = batch["input_ids"][sample_idx][attention_mask]
        labels = batch["labels"][sample_idx][attention_mask]
        n_states = int(batch["n_states"][sample_idx].item())
        n_actions = int(batch["n_actions"][sample_idx].item())
        answer_start_idx = int(batch["answer_start_idx"][sample_idx].item())
        is_vla_data = bool(batch["is_vla_data"][sample_idx].item())
        state_token_count = int((valid_input_ids == state_token_id).sum().item())
        action_token_count = int((valid_input_ids == action_token_id).sum().item())
        non_ignore_labels = int((labels != ignore_index).sum().item())

        sample_checks = {
            "sample_index": sample_idx,
            "is_vla_data": is_vla_data,
            "seq_len": int(valid_input_ids.numel()),
            "answer_start_idx": answer_start_idx,
            "n_states": n_states,
            "n_actions": n_actions,
            "state_token_count": state_token_count,
            "action_token_count": action_token_count,
            "non_ignore_labels": non_ignore_labels,
            "finite_states": bool(torch.isfinite(batch["states"][sample_idx]).all().item()),
            "finite_actions": bool(torch.isfinite(batch["actions"][sample_idx]).all().item()),
        }

        sample_failures = []
        if answer_start_idx > sample_checks["seq_len"]:
            sample_failures.append("answer_start_idx exceeds valid token length")
        if state_token_count != n_states:
            sample_failures.append("state token count does not match n_states")
        if is_vla_data and action_token_count != n_actions:
            sample_failures.append("action token count does not match n_actions for VLA sample")
        if is_vla_data and non_ignore_labels != 0:
            sample_failures.append("VLA labels should stay fully masked")
        if not is_vla_data and non_ignore_labels <= 0:
            sample_failures.append("VLM sample should keep assistant supervision labels")
        if not sample_checks["finite_states"]:
            sample_failures.append("states contain non-finite values")
        if not sample_checks["finite_actions"]:
            sample_failures.append("actions contain non-finite values")

        sample_checks["failures"] = sample_failures
        results.append(sample_checks)
        failures.extend(f"sample {sample_idx}: {message}" for message in sample_failures)

    return {
        "batch_size": batch_size,
        "samples": results,
        "failures": failures,
        "passed": len(failures) == 0,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(pathlib.Path(args.config))
    dataset, data_collator, normalizer, normalizer_path = prepare_dataset(cfg, args)
    dataloader = build_dataloader(cfg, dataset, args)
    batch = next(iter(dataloader))

    motion_type = dataset.motion_type if args.dataset_kind == "vla" else dataset.vla_dataset.motion_type
    use_relative_action = dataset.use_relative_action if args.dataset_kind == "vla" else dataset.vla_dataset.use_relative_action
    state_normalizer, action_normalizer = select_normalizer_fields(normalizer, use_relative_action)
    tokenizer = data_collator.batch_processor.tokenizer

    sample_entries = []
    for sample_index in range(int(batch["input_ids"].shape[0])):
        sample_entries.append(
            build_sample_export(
                sample_index=sample_index,
                batch=batch,
                tokenizer=tokenizer,
                state_normalizer=state_normalizer,
                action_normalizer=action_normalizer,
                use_relative_action=use_relative_action,
                motion_type=motion_type,
                output_dir=output_dir,
            )
        )

    contract_report = validate_batch_contract(batch, data_collator)
    manifest = {
        "config": args.config,
        "split": args.split,
        "dataset_kind": args.dataset_kind,
        "sample_count": args.sample_count,
        "normalizer_path": normalizer_path,
        "contract_report": contract_report,
    }
    (output_dir / "contract_report.json").write_text(
        json.dumps(tensor_to_list(contract_report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(tensor_to_list(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_html_report(output_dir, manifest, sample_entries)

    if not contract_report["passed"]:
        raise SystemExit("Real batch contract check failed. See contract_report.json for details.")


if __name__ == "__main__":
    main()

