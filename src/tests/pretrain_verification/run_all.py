"""
Run all pretrain verification phases sequentially.

Usage:
    # Run phases that don't need real model weights (2,3,4,5+6,7,8,9):
    python -m src.tests.pretrain_verification.run_all

    # Run all phases including data pipeline:
    python -m src.tests.pretrain_verification.run_all \
        --config-path src/config/experiment/legendvla_qwen3_vl.yaml \
        --normalizer-path /path/to/normalizer.pkl

    # Skip visualizations:
    python -m src.tests.pretrain_verification.run_all --skip-visual

    # Run specific phases only:
    python -m src.tests.pretrain_verification.run_all --phases 4 5 6 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PHASES = {
    0: {
        "name": "Phase 0: Normalizer",
        "module": "src.tests.pretrain_verification.phase0_normalizer",
        "requires_data": True,
        "args_key": "normalizer_path",
    },
    1: {
        "name": "Phase 1: Data Pipeline",
        "module": "src.tests.pretrain_verification.phase1_data_pipeline",
        "requires_data": True,
        "args_key": "config_path",
    },
    2: {
        "name": "Phase 2: Embeddings",
        "module": "src.tests.pretrain_verification.phase2_embeddings",
        "requires_data": False,
    },
    3: {
        "name": "Phase 3: Forward Stability",
        "module": "src.tests.pretrain_verification.phase3_forward_stability",
        "requires_data": False,
    },
    4: {
        "name": "Phase 4: Loss Verification",
        "module": "src.tests.pretrain_verification.phase4_loss_verification",
        "requires_data": False,
    },
    56: {
        "name": "Phase 5+6: RTC + Flow Matching",
        "module": "src.tests.pretrain_verification.phase5_rtc_and_phase6_flow",
        "requires_data": False,
    },
    7: {
        "name": "Phase 7: DiffLoss",
        "module": "src.tests.pretrain_verification.phase7_diffloss",
        "requires_data": False,
    },
    8: {
        "name": "Phase 8: Training Loop",
        "module": "src.tests.pretrain_verification.phase8_training_loop",
        "requires_data": False,
    },
    9: {
        "name": "Phase 9: Inference",
        "module": "src.tests.pretrain_verification.phase9_inference",
        "requires_data": False,
    },
}


def run_phase(phase_id: int, phase_info: dict, args: argparse.Namespace) -> bool:
    """Run one phase as a subprocess. Returns True if passed."""
    cmd = [sys.executable, "-m", phase_info["module"]]

    if args.skip_visual:
        cmd.append("--skip-visual")

    # Pass data-specific args
    if phase_info.get("args_key") == "normalizer_path" and args.normalizer_path:
        cmd.extend(["--normalizer-path", args.normalizer_path])
    elif phase_info.get("args_key") == "config_path" and args.config_path:
        cmd.extend(["--config-path", args.config_path])
    elif phase_info.get("requires_data"):
        print(f"  SKIP {phase_info['name']} (requires --normalizer-path or --config-path)")
        return True  # Not a failure, just skipped

    print(f"\n{'='*60}")
    print(f"  Running {phase_info['name']}...")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[3]))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all pretrain verification phases")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Experiment config path (for Phase 0, 1)")
    parser.add_argument("--normalizer-path", type=str, default=None,
                        help="Normalizer pickle path (for Phase 0)")
    parser.add_argument("--skip-visual", action="store_true",
                        help="Skip all visualization generation")
    parser.add_argument("--phases", type=int, nargs="+", default=None,
                        help="Run only specific phases (e.g., --phases 4 8 9)")
    args = parser.parse_args()

    # Determine which phases to run
    if args.phases:
        phase_ids = args.phases
    else:
        phase_ids = sorted(PHASES.keys())

    results = {}
    for pid in phase_ids:
        if pid not in PHASES:
            # Handle combined phases (5 or 6 -> 56)
            if pid in (5, 6) and 56 in PHASES:
                pid = 56
            else:
                print(f"  Unknown phase: {pid}")
                continue

        if pid in results:
            continue  # Already ran (e.g., 5 and 6 both map to 56)

        passed = run_phase(pid, PHASES[pid], args)
        results[pid] = passed

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for pid in sorted(results.keys()):
        status = "PASS" if results[pid] else "FAIL"
        print(f"  [{status}] {PHASES[pid]['name']}")
        if not results[pid]:
            all_passed = False

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  Total: {passed}/{total} passed")
    print(f"{'='*60}\n")

    # Generate aggregated HTML report
    from src.tests.pretrain_verification.utils import generate_html_report, get_output_dir
    output_root = get_output_dir("").parent  # outputs/pretrain_verification/
    html_path = generate_html_report(output_root)
    print(f"  View report: python -m http.server -d {output_root}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
