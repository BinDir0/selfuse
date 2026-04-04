from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap_module(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m


def module_state_dict(m: nn.Module) -> dict[str, Any]:
    return unwrap_module(m).state_dict()


def maybe_save_step_checkpoint(
    ckpt_dir: str | None, optim_steps: int, model: nn.Module, *, is_rank0: bool
) -> None:
    if not ckpt_dir or not is_rank0:
        return
    sd = module_state_dict(model)
    path = os.path.join(ckpt_dir, f"step_{optim_steps:07d}.pt")
    torch.save(sd, path)
    torch.save(sd, os.path.join(ckpt_dir, "latest.pt"))
