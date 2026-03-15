#!/usr/bin/env python3
"""Simple test to verify torch profiler works"""

import torch
from torch.profiler import profile, ProfilerActivity, schedule
from pathlib import Path

def dummy_work():
    """Simulate some GPU work"""
    x = torch.randn(1000, 1000, device='cuda')
    for _ in range(10):
        x = torch.mm(x, x)
    return x

def test_profiler():
    output_dir = Path("./test_profiler_output")
    output_dir.mkdir(exist_ok=True)

    print(f"[TEST] Starting profiler test")
    print(f"[TEST] Output dir: {output_dir}")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=0, active=2, repeat=1),
        on_trace_ready=lambda p: (
            print(f"[TEST] Trace ready! Exporting..."),
            p.export_chrome_trace(str(output_dir / "test_trace.json"))
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        print("[TEST] Step 1")
        prof.step()
        dummy_work()

        print("[TEST] Step 2")
        prof.step()
        dummy_work()

        print("[TEST] Step 3 (should trigger trace)")
        prof.step()

    print("[TEST] Profiler context exited")

    # Check if file was created
    trace_file = output_dir / "test_trace.json"
    if trace_file.exists():
        print(f"[TEST] SUCCESS! Trace file created: {trace_file}")
        print(f"[TEST] File size: {trace_file.stat().st_size} bytes")
    else:
        print(f"[TEST] FAILED! Trace file not found: {trace_file}")

if __name__ == "__main__":
    test_profiler()
