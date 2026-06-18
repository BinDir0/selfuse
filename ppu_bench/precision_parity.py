"""Worry #1 - cross-platform precision parity (GPU golden vs PPU replay).

Two phases. Run capture on the NVIDIA box where the model was trained, copy the
golden file to the PPU, then run compare there:

    # on NVIDIA GPU
    python precision_parity.py capture --out golden.pt
    # on PPU
    python precision_parity.py compare --golden golden.pt

The point is to separate two errors:
  * PPU_bf16  vs  GPU_bf16   -> pure platform/kernel divergence
  * PPU_bf16  vs  GPU_fp32   -> total error
and compare the latter against GPU's *own* bf16-vs-fp32 error (stored in the
golden file). Verdict: if PPU's total error is in the same ballpark as GPU's
own bf16 error, precision is fine; if it's much larger, the PPU diverges.

It also reports per-layer divergence *growth* (does error stay bounded with
depth?) and run-to-run determinism on the PPU.

To test the *real* EgoVLA model instead of the representative one, import this
module and pass your own (model, input) into `capture_tensors` / `replay`.
"""

from __future__ import annotations

import argparse

import torch

from common import IS_ACCEL, banner, rel_err
from repmodel import RepModel

LAYERS, D, HEADS, FFN = 8, 2048, 16, 6144
B, S = 2, 512


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    if IS_ACCEL:
        torch.cuda.manual_seed_all(s)


def _build():
    _seed(0)
    return RepModel(LAYERS, D, HEADS, FFN)


def capture_tensors(model: torch.nn.Module, x: torch.Tensor) -> dict:
    """Run a forward pass with per-block hooks; return {layer_idx: output(cpu fp32)}."""
    outs: dict[int, torch.Tensor] = {}
    handles = []
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(
            lambda m, inp, out, i=i: outs.__setitem__(i, out.detach().float().cpu())
        ))
    with torch.no_grad():
        final = model(x)
    for h in handles:
        h.remove()
    outs["final"] = final.detach().float().cpu()
    outs["loss"] = final.float().pow(2).mean().detach().cpu()
    return outs


def capture(out_path: str) -> None:
    banner("precision parity: CAPTURE (run on NVIDIA GPU)")
    device = torch.device("cuda" if IS_ACCEL else "cpu")
    model = _build().to(device)
    _seed(1)
    x = torch.randn(B, S, D, device=device)

    fp32 = capture_tensors(model.float(), x.float())
    bf16 = capture_tensors(model.to(torch.bfloat16), x.to(torch.bfloat16))

    # GPU's own bf16-vs-fp32 error per layer = the "acceptable" baseline.
    gpu_bf16_err = {k: rel_err(bf16[k], fp32[k]) for k in fp32 if k != "loss"}

    torch.save({
        "meta": {"torch": torch.__version__, "device": torch.cuda.get_device_name(0) if IS_ACCEL else "cpu",
                 "dims": [LAYERS, D, HEADS, FFN, B, S]},
        "state_dict": {k: v.float().cpu() for k, v in model.state_dict().items()},
        "input": x.float().cpu(),
        "fp32": fp32,
        "bf16": bf16,
        "gpu_bf16_err": gpu_bf16_err,
    }, out_path)
    print(f"  saved golden -> {out_path}")
    print(f"  reference device: {torch.cuda.get_device_name(0) if IS_ACCEL else 'cpu'}")
    for k in sorted(gpu_bf16_err, key=lambda z: (z != 'final', z)):
        print(f"    GPU bf16-vs-fp32 rel_err @ {str(k):<6}: {gpu_bf16_err[k]:.4f}")


def replay(model: torch.nn.Module, x: torch.Tensor) -> dict:
    model.set_backend("sdpa")
    return capture_tensors(model, x)


def compare(golden_path: str) -> None:
    banner("precision parity: COMPARE (run on PPU)")
    g = torch.load(golden_path, map_location="cpu", weights_only=False)
    print(f"  golden from: {g['meta']['device']} (torch {g['meta']['torch']})")
    device = torch.device("cuda" if IS_ACCEL else "cpu")

    model = _build().to(device)
    model.load_state_dict({k: v for k, v in g["state_dict"].items()})
    model = model.to(torch.bfloat16)
    x = g["input"].to(device).to(torch.bfloat16)

    ppu = replay(model, x)
    ppu2 = replay(model, x)  # determinism check

    keys = [k for k in g["fp32"] if k != "loss"]
    print(f"\n  {'layer':<7} {'PPUbf16-GPUbf16':>16} {'PPUbf16-GPUfp32':>16} "
          f"{'GPUbf16-GPUfp32':>16} {'nondet':>9}")
    totals = []
    for k in sorted(keys, key=lambda z: (z != 'final', z)):
        e_plat = rel_err(ppu[k], g["bf16"][k])
        e_total = rel_err(ppu[k], g["fp32"][k])
        e_gpu = g["gpu_bf16_err"][k]
        e_nondet = rel_err(ppu[k], ppu2[k])
        totals.append((e_total, e_gpu))
        print(f"  {str(k):<7} {e_plat:>16.4f} {e_total:>16.4f} {e_gpu:>16.4f} {e_nondet:>9.1e}")

    # Verdict: PPU total error vs GPU's own bf16 error at the final output.
    final_total = rel_err(ppu["final"], g["fp32"]["final"])
    final_gpu = g["gpu_bf16_err"]["final"]
    ratio = final_total / max(final_gpu, 1e-9)
    print(f"\n  final-output total error / GPU bf16 baseline = {ratio:.2f}x")
    if ratio <= 2.0:
        print("  ✅ PPU precision in the same ballpark as GPU bf16 - looks fine.")
    elif ratio <= 5.0:
        print("  ⚠️  PPU error 2-5x GPU bf16 - acceptable for inference, watch long training.")
    else:
        print("  ❌ PPU error >5x GPU bf16 - real precision divergence, investigate kernels.")
    nondet = rel_err(ppu["final"], ppu2["final"])
    print(f"  run-to-run nondeterminism (final): {nondet:.2e}"
          + ("  (bit-stable)" if nondet == 0 else "  (nondeterministic)"))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--out", default="golden.pt")
    p = sub.add_parser("compare")
    p.add_argument("--golden", default="golden.pt")
    args = ap.parse_args()
    if args.cmd == "capture":
        capture(args.out)
    else:
        compare(args.golden)


if __name__ == "__main__":
    main()
