"""Layer 1 - multi-card distributed: collective bandwidth + FSDP2 + DeepSpeed.

Launch with torchrun, e.g.:
    torchrun --nproc_per_node=2 layer1_distributed.py

EgoVLA's production path is FSDP2 (fully_shard) over NCCL. On the PPU, NCCL is
the vendor's CCL shim, so this probes (1) collective bandwidth, (2) that
fully_shard wraps + steps without hanging, (3) optional DeepSpeed ZeRO.

NOTE: do NOT export EgoVLA's debug_start.sh NCCL_IB_* env here - those mlx5
HCA names are NVIDIA-fabric specific and will hang on PPU. Let the vendor
CCL pick its own defaults.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from common import banner, device_time, guard, record, Result


def _is_dist() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def run() -> None:
    banner("Layer 1: distributed (collectives + FSDP2 + DeepSpeed)")
    if not _is_dist():
        print("  Not under torchrun. Launch with:")
        print("    torchrun --nproc_per_node=<N> layer1_distributed.py")
        return

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")  # PPU exposes a NCCL-compatible CCL

    def rec(r: Result) -> None:
        if rank == 0:
            record(r)

    # ---- all_reduce bandwidth ----
    r = Result("all_reduce busbw (256 MB)")
    try:
        n = 64 * 1024 * 1024  # 256 MB fp32
        t = torch.randn(n, device="cuda")
        secs = device_time(lambda: dist.all_reduce(t), iters=20, warmup=5)
        size_gb = t.numel() * t.element_size() / 1e9
        busbw = 2 * (world - 1) / world * size_gb / secs
        r.metrics = {"busbw_GB/s": round(busbw, 1), "world": world}
        r.detail = f"{busbw:.1f} GB/s busbw across {world} cards"
    except Exception as exc:  # noqa: BLE001
        r.status, r.error = "FAIL", f"{type(exc).__name__}: {exc}"
    rec(r)

    # ---- FSDP2 (fully_shard) one train step ----
    r = Result("FSDP2 fully_shard train step")
    try:
        from torch.distributed.fsdp import fully_shard

        model = nn.Sequential(
            nn.Linear(2048, 4096), nn.GELU(), nn.Linear(4096, 2048)
        ).cuda().to(torch.bfloat16)
        for m in model:
            if isinstance(m, nn.Linear):
                fully_shard(m)
        fully_shard(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)
        loss0 = None
        for step in range(3):
            opt.zero_grad()
            out = model(x)
            loss = out.float().pow(2).mean()
            loss.backward()
            opt.step()
            if step == 0:
                loss0 = loss.item()
        r.metrics = {"loss0": round(loss0, 5), "loss_last": round(loss.item(), 5)}
        r.detail = f"3 steps ok, loss {loss0:.4f} -> {loss.item():.4f}"
        if not torch.isfinite(torch.tensor(loss.item())):
            r.status = "FAIL"
    except Exception as exc:  # noqa: BLE001
        r.status, r.error = "FAIL", f"{type(exc).__name__}: {exc}"
    rec(r)

    # ---- DeepSpeed ZeRO (optional - PPU SDK ships deepspeed) ----
    r = Result("DeepSpeed ZeRO-2 train step")
    try:
        import deepspeed  # noqa: F401

        model = nn.Sequential(nn.Linear(2048, 4096), nn.GELU(), nn.Linear(4096, 2048))
        ds_config = {
            "train_micro_batch_size_per_gpu": 8,
            "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 2},
        }
        engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), config=ds_config
        )
        x = torch.randn(8, 2048, device=engine.device, dtype=torch.bfloat16)
        out = engine(x)
        loss = out.float().pow(2).mean()
        engine.backward(loss)
        engine.step()
        r.metrics = {"loss": round(loss.item(), 5)}
        r.detail = f"loss={loss.item():.4f}"
    except ImportError:
        r.status, r.detail = "SKIP", "deepspeed not installed"
    except Exception as exc:  # noqa: BLE001
        r.status, r.error = "FAIL", f"{type(exc).__name__}: {exc}"
    rec(r)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
