"""Re-pin dataloader workers to the pool published by the launch wrapper.

Main processes are pinned to a few dedicated cores by the launch wrapper
(scripts/numa_bind_wrapper.sh) so NCCL/FSDP can run uninterrupted. Workers
fork from main and inherit that narrow affinity; this widens them back to
the per-NUMA worker pool so dataloader CPU work doesn't starve main.

Hardware constants live in the wrapper; this module just reads
DATALOADER_WORKER_CPUS.
"""
from __future__ import annotations

import logging
import os

import psutil

log = logging.getLogger(__name__)


def parse_cpu_list(spec: str) -> list[int]:
    """Parse a Linux cpu list like '8-31,72-95' into a flat list of ints."""
    cpus: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def dataloader_worker_init_fn(worker_id: int) -> None:
    pool_spec = os.environ.get("DATALOADER_WORKER_CPUS")
    if not pool_spec:
        # Wrapper not used (single-card debug etc.); leave inherited affinity alone.
        return
    pool = parse_cpu_list(pool_spec)
    psutil.Process().cpu_affinity(pool)
    if worker_id == 0:
        log.info("dataloader worker pinned to %d cores: %s", len(pool), pool_spec)
