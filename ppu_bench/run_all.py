"""PPU bench orchestrator - Layers 0-2.

    python run_all.py                 # everything except distributed
    python run_all.py --only ops compile
    PPU_PEAK_TFLOPS=148 python run_all.py

Distributed (Layer 1) needs torchrun and is run separately:
    torchrun --nproc_per_node=2 layer1_distributed.py
"""

from __future__ import annotations

import argparse
import collections
import os

import layer0_env
import layer1_compile
import layer1_gemm
import layer1_membw
import layer1_models
import layer1_ops
import feature_matrix
import opt_ablation
from common import banner, dump, results

LAYERS = {
    "gemm": layer1_gemm.run,
    "membw": layer1_membw.run,
    "ops": layer1_ops.run,
    "compile": layer1_compile.run,
    "models": layer1_models.run,
    "matrix": feature_matrix.run,
    "ablation": opt_ablation.run,
}


def summary() -> None:
    banner("SUMMARY")
    counts = collections.Counter(r.status for r in results())
    for r in results():
        print(r.line())
    print()
    order = ["PASS", "FALLBACK", "SLOW", "SKIP", "FAIL"]
    print("  " + "  ".join(f"{s}={counts.get(s, 0)}" for s in order))
    if counts.get("FAIL"):
        print("\n  ⚠️  FAILs are the headline: a feature that errored or returned wrong numbers.")
    if counts.get("SLOW") or counts.get("FALLBACK"):
        print("  ⚠️  SLOW/FALLBACK = ran but degraded (likely silent eager lowering) - investigate.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(LAYERS), help="run a subset")
    ap.add_argument("--out", default="ppu_bench_report.json")
    args = ap.parse_args()

    env = layer0_env.main()
    chosen = args.only or list(LAYERS)
    for name in chosen:
        try:
            LAYERS[name]()
        except Exception as exc:  # noqa: BLE001 - a whole layer blowing up shouldn't kill the run
            print(f"  [layer '{name}' crashed: {type(exc).__name__}: {exc}]")

    summary()
    dump(args.out, extra={"env": env})
    print("\nRun distributed separately:  torchrun --nproc_per_node=<N> layer1_distributed.py")


if __name__ == "__main__":
    main()
