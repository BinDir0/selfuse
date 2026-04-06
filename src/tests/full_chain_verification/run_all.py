"""
Run all full-chain verification parts.

Usage:
    # CPU-only parts (1, 2, 4):
    python -m src.tests.full_chain_verification.run_all \
        --model-path /path/to/Qwen3-VL-2B-Instruct \
        --normalizer-path /path/to/normalizer.pkl \
        --parts 1 2 4

    # GPU parts (3, 5, 6, 7):
    python -m src.tests.full_chain_verification.run_all \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --parts 3 5 6 7

    # All parts:
    python -m src.tests.full_chain_verification.run_all \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --model-path /path/to/Qwen3-VL-2B-Instruct \
        --normalizer-path /path/to/normalizer.pkl \
        [--vla-shard '/data/shards/taco/{000..001}.tar'] \
        [--checkpoint-path /path/to/checkpoint.pt]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PARTS = {
    0: {
        "name": "Part 0: Per-Dataset Audit",
        "module": "src.tests.full_chain_verification.part0_dataset_audit",
        "needs_gpu": False,
    },
    1: {
        "name": "Part 1: Data Pipeline + Sample Level",
        "module": "src.tests.full_chain_verification.part1_data_pipeline",
        "needs_gpu": False,
    },
    2: {
        "name": "Part 2: Collator / Tokenization / Mask",
        "module": "src.tests.full_chain_verification.part2_collator_tokenization",
        "needs_gpu": False,
    },
    3: {
        "name": "Part 3: Backbone Embedding + Prefix Cache",
        "module": "src.tests.full_chain_verification.part3_backbone_prefix_cache",
        "needs_gpu": True,
    },
    4: {
        "name": "Part 4: Flow Matching Math + RTC Logic",
        "module": "src.tests.full_chain_verification.part4_flow_rtc_math",
        "needs_gpu": False,
    },
    5: {
        "name": "Part 5: Action Expert + DiffLoss + Loss",
        "module": "src.tests.full_chain_verification.part5_expert_diffloss_loss",
        "needs_gpu": True,
    },
    6: {
        "name": "Part 6: Inference + Train-Infer Parity",
        "module": "src.tests.full_chain_verification.part6_inference_parity",
        "needs_gpu": True,
    },
    7: {
        "name": "Part 7: E2E Training + Checkpoint",
        "module": "src.tests.full_chain_verification.part7_e2e_training",
        "needs_gpu": True,
    },
    8: {
        "name": "Part 8: Accelerate Training Verification",
        "module": "src.tests.full_chain_verification.part8_accelerate_training",
        "needs_gpu": True,
    },
}


def run_part(part_id: int, part_info: dict, args: argparse.Namespace) -> bool:
    """Run one part as a subprocess. Returns True if passed."""
    cmd = [sys.executable, "-m", part_info["module"]]

    if args.skip_visual:
        cmd.append("--skip-visual")

    # Route CLI args to the appropriate parts
    if part_id == 0:
        if args.data_root:
            cmd.extend(["--data-root", args.data_root])
        else:
            cmd.append("--auto")
        if args.normalizer_path:
            cmd.extend(["--normalizer-path", args.normalizer_path])
        if args.model_path:
            cmd.extend(["--model-path", args.model_path])
    elif part_id == 1:
        if args.normalizer_path:
            cmd.extend(["--normalizer-path", args.normalizer_path])
        if args.vla_shard:
            cmd.extend(["--vla-shard", args.vla_shard])
        if args.vlm_shard:
            cmd.extend(["--vlm-shard", args.vlm_shard])
    elif part_id == 2:
        if args.model_path:
            cmd.extend(["--model-path", args.model_path])
        else:
            print(f"  SKIP {part_info['name']} (requires --model-path)")
            return True
    elif part_id in (3, 5, 6, 8):
        if args.config_path:
            cmd.extend(["--config-path", args.config_path])
        else:
            print(f"  SKIP {part_info['name']} (requires --config-path)")
            return True
    elif part_id == 7:
        if args.config_path:
            cmd.extend(["--config-path", args.config_path])
        else:
            print(f"  SKIP {part_info['name']} (requires --config-path)")
            return True
        if args.normalizer_path:
            cmd.extend(["--normalizer-path", args.normalizer_path])
        if args.checkpoint_path:
            cmd.extend(["--checkpoint-path", args.checkpoint_path])
        if args.vla_shard:
            cmd.extend(["--vla-shard", args.vla_shard])
    # Part 4 needs no extra args

    print(f"\n{'=' * 60}")
    print(f"  Running {part_info['name']}...")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[3]))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run full-chain verification")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Hydra experiment config (for GPU parts 3,5,6,7)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Data root with vla/ and vlm/ sub-dirs (for Part 0)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Qwen3-VL-2B-Instruct path (for Part 0, 2)")
    parser.add_argument("--normalizer-path", type=str, default=None,
                        help="normalizer.pkl path (for Parts 1, 7)")
    parser.add_argument("--vla-shard", type=str, default=None,
                        help="VLA WebDataset shard pattern (for Part 1)")
    parser.add_argument("--vlm-shard", type=str, default=None,
                        help="VLM WebDataset shard pattern (for Part 1)")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Trained checkpoint path (for Part 7)")
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--parts", type=int, nargs="+", default=None,
                        help="Run specific parts only (e.g. --parts 4 2 1)")
    args = parser.parse_args()

    part_ids = args.parts if args.parts else sorted(PARTS.keys())

    results = {}
    for pid in part_ids:
        if pid not in PARTS:
            print(f"  Unknown part: {pid}")
            continue
        passed = run_part(pid, PARTS[pid], args)
        results[pid] = passed

    # Summary
    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 60}")
    all_passed = True
    for pid in sorted(results.keys()):
        status = "PASS" if results[pid] else "FAIL"
        print(f"  [{status}] {PARTS[pid]['name']}")
        if not results[pid]:
            all_passed = False

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed}/{total} passed")
    print(f"{'=' * 60}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
