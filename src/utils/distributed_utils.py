from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

log = logging.getLogger(__name__)


@dataclass
class DistributedContext:
    """Holds process group topology and device assignment."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    mesh: DeviceMesh | None  # None for single-node plain FSDP


def init_distributed(
    backend: str = "nccl",
    timeout_sec: int = 3600,
) -> DistributedContext:
    """Initialize the distributed process group and optionally create an HSDP mesh.

    When running on multiple nodes (detected via LOCAL_WORLD_SIZE from
    torchrun), a 2D DeviceMesh is created automatically:
      - dim 0 ("replicate"): across nodes — gradient all-reduce only
      - dim 1 ("shard"):     within a node — all-gather / reduce-scatter via NVLink

    On a single node the mesh is left as None for plain 1D FSDP sharding.

    Environment variables read (all set by torchrun):
      LOCAL_RANK, LOCAL_WORLD_SIZE
    Reference: torch/distributed/elastic/agent/server/local_elastic_agent.py:305-326

    HSDP mesh dim convention (dim0=replicate, dim1=shard):
    Reference: torch/distributed/fsdp/_fully_shard/_fsdp_init.py:60-69

    Returns:
        DistributedContext with rank, device, and optional HSDP mesh.
    """
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_sec))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Auto-detect multi-node topology for HSDP
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    num_nodes = world_size // gpus_per_node

    mesh = None
    if num_nodes > 1:
        # 2D mesh: dim 0 = replicate (across nodes), dim 1 = shard (within node)
        # Reference: torch/distributed/device_mesh.py:1460-1469
        mesh = init_device_mesh(
            "cuda",
            mesh_shape=(num_nodes, gpus_per_node),
            mesh_dim_names=("replicate", "shard"),
        )

    if rank == 0:
        if mesh is not None:
            log.info(
                "Distributed init: backend=%s, world_size=%d, "
                "HSDP mesh=%d nodes x %d GPUs/node (shard within node via NVLink)",
                backend, world_size, num_nodes, gpus_per_node,
            )
        else:
            log.info(
                "Distributed init: backend=%s, world_size=%d, plain FSDP (single node)",
                backend, world_size,
            )

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        mesh=mesh,
    )


def build_mixed_precision_policy(use_bf16: bool) -> MixedPrecisionPolicy | None:
    """Build the project's standard mixed-precision policy.

    The recipe: fp32 master params with bf16 compute + bf16 output.
    reduce_dtype stays fp32 for numerically safe gradient all-reduce.

    Returns None when use_bf16 is False (full fp32 training).

    Reference: torch/distributed/fsdp/_fully_shard/_fsdp_api.py:14-53
    """
    if not use_bf16:
        return None
    return MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )


def apply_fsdp2(
    model: torch.nn.Module,
    wrap_classes: tuple[type, ...],
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool = False,
    mp_policy: MixedPrecisionPolicy | None = None,
) -> None:
    """Apply FSDP2 sharding to a model by wrapping matching sub-modules, then root.

    Sub-modules whose type matches any entry in wrap_classes are sharded
    first (leaf-to-root order from module iteration). The root module is
    always sharded last.

    With a 2D mesh this becomes HSDP: parameters are sharded on dim 1
    and replicated on dim 0. With mesh=None, this is plain 1D FSDP.

    Reference: torch/distributed/fsdp/_fully_shard/_fully_shard.py:90-99

    Args:
        model: The model to shard. Modified in place.
        wrap_classes: Tuple of module types to individually shard before root.
        mesh: 2D DeviceMesh for HSDP, or None for plain FSDP.
        reshard_after_forward: Whether to reshard parameters after each
            forward pass. False keeps params unsharded between forward and
            backward, trading memory for speed.
        mp_policy: Mixed precision policy. When None, all computation
            stays in the model's current dtype.
    """
    fsdp_kwargs: dict = {"reshard_after_forward": reshard_after_forward}
    if mesh is not None:
        fsdp_kwargs["mesh"] = mesh
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy

    for module in model.modules():
        if isinstance(module, wrap_classes):
            fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
