# Current approach: unshard/reshard per update step, shadow params on CPU.
# For large-scale EMA where per-step all-gather becomes a bottleneck, consider
# the sharded DTensor EMA approach (requires a second fully_shard-wrapped model):
# https://github.com/NVlabs/rcm/blob/main/rcm/utils/dtensor_helper.py

import logging
from contextlib import contextmanager
from typing import Dict, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def _collect_fsdp_modules(module: nn.Module) -> list:
    """Collect all FSDPModule instances in the module tree."""
    # Import here to avoid hard dependency when FSDP is not used.
    from torch.distributed.fsdp import FSDPModule
    return [m for m in module.modules() if isinstance(m, FSDPModule)]


def _is_fsdp_wrapped(module: nn.Module) -> bool:
    return len(_collect_fsdp_modules(module)) > 0


@contextmanager
def _unshard_params(module: nn.Module):
    """
    Temporarily unshard all FSDP submodules so that parameters become
    plain torch.Tensor (full, unsharded) on their original device.
    Non-FSDP modules pass through as a no-op.

    This is the FSDP2 equivalent of FSDP1's summon_full_params.
    See: https://github.com/pytorch/pytorch/issues/137129
    """
    fsdp_modules = _collect_fsdp_modules(module)
    for m in fsdp_modules:
        m.unshard()
    try:
        yield
    finally:
        for m in fsdp_modules:
            m.reshard()


class ModelAveraging:
    """
    Lightweight EMA / SWA that works with both plain and FSDP2-wrapped modules.

    Shadow parameters are stored as plain tensors on *device* (typically CPU),
    fully detached from the training graph.  For FSDP2 models the implementation
    uses unshard()/reshard() to read full parameters — no deepcopy needed.
    """

    def __init__(self, model: nn.Module, cfg, device):
        self.use_ema = cfg.ema.enabled
        self.use_swa = cfg.swa.enabled
        assert not (self.use_ema and self.use_swa), (
            "Cannot use both EMA and SWA at once"
        )

        self.model = model
        self.device = device
        self.shadow: Optional[Dict[str, torch.Tensor]] = None

        if self.use_ema:
            self.start = cfg.ema.start
            self.decay = cfg.ema.decay
            self.freq = cfg.ema.freq
            self.avg_device = torch.device(cfg.ema.device or "cpu")

        if self.use_swa:
            self.start = cfg.swa.start
            self.freq = cfg.swa.freq
            self.avg_device = torch.device(cfg.swa.device or "cpu")
            self.n_averaged = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def maybe_initialize(self, cnt_update: int):
        if self.shadow is not None:
            return
        if not (self.use_ema or self.use_swa):
            return
        if cnt_update < self.start:
            return

        with torch.no_grad(), _unshard_params(self.model):
            self.shadow = {
                name: param.data.float().to(self.avg_device).clone()
                for name, param in self.model.named_parameters()
            }
        if self.use_swa:
            self.n_averaged = 1
        tag = "EMA" if self.use_ema else "SWA"
        log.info(f"Starting {tag} at update step {cnt_update} on {self.avg_device}")

    def maybe_update(self, cnt_update: int):
        if self.shadow is None:
            return
        if cnt_update % self.freq != 0:
            return
        if self.use_ema:
            self._ema_update()
        elif self.use_swa:
            self._swa_update()

    # ------------------------------------------------------------------
    # Internal update routines
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ema_update(self):
        with _unshard_params(self.model):
            for name, param in self.model.named_parameters():
                shadow = self.shadow[name]
                # shadow = decay * shadow + (1 - decay) * param
                shadow.lerp_(param.data.float().to(self.avg_device), 1.0 - self.decay)

    @torch.no_grad()
    def _swa_update(self):
        self.n_averaged += 1
        with _unshard_params(self.model):
            for name, param in self.model.named_parameters():
                shadow = self.shadow[name]
                # running mean: shadow += (param - shadow) / n
                shadow.add_(
                    (param.data.float().to(self.avg_device) - shadow) / self.n_averaged
                )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @contextmanager
    def use_averaged_params(self):
        """Temporarily swap model parameters with the averaged (EMA/SWA) shadow copy.

        Caller must ensure FSDP parameters are already unsharded before
        entering this context (e.g. via eval_with_averaged_model which
        handles unshard/reshard independently).
        """
        if self.shadow is None:
            yield
            return

        originals: Dict[str, torch.Tensor] = {}
        try:
            for name, param in self.model.named_parameters():
                originals[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device, param.dtype))
            yield
        finally:
            for name, param in self.model.named_parameters():
                param.data.copy_(originals[name])

    def averaged_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a copy of the averaged parameters (always full, on avg_device)."""
        if self.shadow is not None:
            return {k: v.clone() for k, v in self.shadow.items()}
        # Fallback: return current model parameters.
        with torch.no_grad(), _unshard_params(self.model):
            return {
                name: param.data.float().to(self.avg_device).clone()
                for name, param in self.model.named_parameters()
            }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        if self.shadow is not None:
            result = {
                "shadow": self.shadow,
                "model_type": "ema" if self.use_ema else "swa",
            }
            if self.use_swa:
                result["n_averaged"] = self.n_averaged
            return result
        return {}

    def load_state_dict(self, state_dict: dict, **kwargs):
        if not state_dict:
            self.shadow = None
            return
        model_type = state_dict.get("model_type")
        if model_type == "ema" and self.use_ema:
            self.shadow = state_dict["shadow"]
            log.info("Loaded EMA shadow parameters")
        elif model_type == "swa" and self.use_swa:
            self.shadow = state_dict["shadow"]
            self.n_averaged = state_dict.get("n_averaged", 1)
            log.info(f"Loaded SWA shadow parameters (n_averaged={self.n_averaged})")
        else:
            log.warning(
                f"Model averaging type mismatch: saved={model_type}, "
                f"current=ema:{self.use_ema}/swa:{self.use_swa}"
            )
            self.shadow = None
