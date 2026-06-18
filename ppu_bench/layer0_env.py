"""Layer 0 - capture the environment we are benchmarking.

The numbers from every other layer are meaningless without this, so run it
first and keep the JSON alongside the results.
"""

from __future__ import annotations

import importlib
import platform
import subprocess

import torch

from common import DEVICE, IS_ACCEL, PEAK_TFLOPS, banner


def _ver(mod: str) -> str:
    try:
        return importlib.import_module(mod).__version__
    except Exception as exc:  # noqa: BLE001
        return f"<absent: {type(exc).__name__}>"


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__}>"


def collect() -> dict:
    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if IS_ACCEL else 0,
        "peak_tflops_assumed": PEAK_TFLOPS,
        "versions": {m: _ver(m) for m in (
            "transformers", "flash_attn", "triton", "deepspeed",
            "accelerate", "diffusers", "xformers", "bitsandbytes",
        )},
    }
    if IS_ACCEL:
        info["devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_mem_gb": round(props.total_memory / 1024**3, 1),
                "multi_processor_count": getattr(props, "multi_processor_count", None),
                "capability": f"{props.major}.{props.minor}",
            })
        # cuDNN / TF32 knobs that matter for the PPU compat layer.
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["allow_tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        info["allow_tf32_cudnn"] = torch.backends.cudnn.allow_tf32
        info["sdpa_backends"] = _probe_sdpa_backends()
    return info


def _probe_sdpa_backends() -> dict:
    """Which SDPA kernels does the compat layer expose?"""
    out = {}
    try:
        from torch.backends.cuda import (
            can_use_efficient_attention,
            can_use_flash_attention,
        )  # noqa: F401
        out["flash_sdp_enabled"] = torch.backends.cuda.flash_sdp_enabled()
        out["mem_efficient_sdp_enabled"] = torch.backends.cuda.mem_efficient_sdp_enabled()
        out["math_sdp_enabled"] = torch.backends.cuda.math_sdp_enabled()
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> dict:
    banner("Layer 0: environment")
    info = collect()
    print(f"  device          : {DEVICE}")
    print(f"  torch           : {info['torch']} (cuda build {info['torch_cuda_build']})")
    print(f"  device_count    : {info['device_count']}")
    for d in info.get("devices", []):
        print(f"    [{d['index']}] {d['name']}  {d['total_mem_gb']} GB  cc {d['capability']}")
    print("  key versions:")
    for mod, ver in info["versions"].items():
        print(f"    {mod:<14}: {ver}")
    if IS_ACCEL:
        print(f"  sdpa backends   : {info['sdpa_backends']}")
        print(f"  tf32 matmul/cudnn: {info['allow_tf32_matmul']}/{info['allow_tf32_cudnn']}")
    return info


if __name__ == "__main__":
    main()
