from typing import Optional, Dict
import logging
import os
import pathlib
import shutil
import sys
import types

import torch

log = logging.getLogger(__name__)


def enable_pathlib_local_pickle_compat() -> None:
    """Register a runtime shim for Python 3.13 pathlib pickle payloads.

    Python 3.13 stores concrete Path classes under pathlib._local, while
    Python 3.10 still exposes them from pathlib.py. Torch distributed
    checkpoint metadata is read with pickle.load, so checkpoints written in a
    newer environment can fail to deserialize in older runtimes unless that
    module path is mapped.

    Reference:
    - CPython 3.13 pathlib package layout: Lib/pathlib/_local.py
    - PyTorch FileSystemReader.read_metadata uses pickle.load on .metadata
    """
    if "pathlib._local" in sys.modules:
        return

    shim = types.ModuleType("pathlib._local")
    for name in [
        "Path",
        "PosixPath",
        "WindowsPath",
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
    ]:
        value = getattr(pathlib, name, None)
        if value is not None:
            setattr(shim, name, value)
    sys.modules["pathlib._local"] = shim

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:04d}-train_loss={train_loss:.4f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def get_ckpt_path(self, accelerator, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            # only main process mkdir and delete the checkpoint
            if accelerator.is_main_process:
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)

                if os.path.exists(delete_path):
                    if os.path.isfile(delete_path): 
                        os.remove(delete_path)
                    else:
                        shutil.rmtree(delete_path)
            return ckpt_path


def load_checkpoint(model: torch.nn.Module, path: str | pathlib.Path) -> None:
    """Load model weights from a checkpoint file or Accelerate directory.

    Supports three formats:
    1. Single file (.pt / .ckpt) — torch.load with key probing.
    2. Accelerate FSDP2 sharded dir (contains pytorch_model_fsdp_0/).
       Uses torch.distributed.checkpoint with no_dist=True (PyTorch 2.3+).
    3. Accelerate safetensors dir — falls back to load_checkpoint_in_model.

    All paths use strict loading: any missing or unexpected keys will raise
    an error instead of silently producing a partially loaded model.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Single-file checkpoint
    if path.is_file():
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        for key in ["model", "module", "model_state_dict"]:
            if key in state_dict:
                state_dict = state_dict[key]
                break
        model.load_state_dict(state_dict, strict=True)
        log.info("Loaded single-file checkpoint from %s", path)
        return

    # Accelerate FSDP2 sharded checkpoint
    # dcp.load supports no_dist=True since PyTorch 2.3+ (fix: pytorch#115660),
    # and auto-infers it when dist is not initialized since 2.4+ (pytorch#118554).
    fsdp_dir = path / "pytorch_model_fsdp_0"
    if fsdp_dir.exists():
        import torch.distributed.checkpoint as dcp

        enable_pathlib_local_pickle_compat()
        state_dict = {"model": model.state_dict()}
        dcp.load(
            state_dict=state_dict,
            storage_reader=dcp.FileSystemReader(str(fsdp_dir)),
            no_dist=True,
        )
        model.load_state_dict(state_dict["model"], strict=True)
        log.info("Loaded FSDP sharded checkpoint from %s", fsdp_dir)
        return

    # Safetensors / other Accelerate format
    # Source: accelerate.utils.load_checkpoint_in_model
    # load_checkpoint_in_model does not support strict mode natively;
    # load into a temporary state_dict and use strict load_state_dict instead.
    from safetensors.torch import load_file

    safetensor_files = sorted(path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {path}")
    state_dict = {}
    for sf in safetensor_files:
        state_dict.update(load_file(str(sf), device="cpu"))
    model.load_state_dict(state_dict, strict=True)
    log.info("Loaded safetensors checkpoint from %s (%d files)", path, len(safetensor_files))


def load_model_and_collator_from_saved_config(
    train_config_path: str | pathlib.Path,
    device: str,
    collator_mode: str = "train",
    dtype: torch.dtype = torch.bfloat16,
):
    """Instantiate model + collator from a training run's saved .hydra/config.yaml.

    Mirrors the loading path in ``evaluate.py::main`` (lines 119-143): load
    the training config, instantiate ``cfg.policy``, move to device in the
    target dtype, and instantiate the collator. Use this when the checkpoint
    was trained with options (use_kv_projection, world_model,
    intermediate_size, ...) that differ from the current code's default
    experiment config.

    Note: this function does NOT register OmegaConf resolvers. Callers that
    need ``${eval:...}`` interpolations in saved configs must register the
    resolver at module import time via::

        OmegaConf.register_new_resolver("eval", eval, replace=True)

    Args:
        train_config_path: Path to the run's saved ``.hydra/config.yaml``.
        device: Target device string (e.g. ``"cuda"``).
        collator_mode: Mode passed to the collator constructor (default
            ``"train"`` to match the most common analysis use case).
        dtype: Tensor dtype for ``model.to(...)`` (default bfloat16).

    Returns:
        ``(model, collator, train_cfg)``.
    """
    import hydra  # lazy import: only this helper needs hydra
    from omegaconf import OmegaConf

    train_cfg = OmegaConf.load(str(train_config_path))
    log.info("Loaded saved training config from %s", train_config_path)
    model = hydra.utils.instantiate(train_cfg.policy)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    collator = hydra.utils.instantiate(train_cfg.data_collator, mode=collator_mode)
    return model, collator, train_cfg
