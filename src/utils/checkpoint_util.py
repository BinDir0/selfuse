from typing import Optional, Dict
import logging
import os
import pathlib
import shutil

import torch

log = logging.getLogger(__name__)

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
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Single-file checkpoint
    if path.is_file():
        state_dict = torch.load(path, map_location="cpu")
        for key in ["model", "module", "model_state_dict"]:
            if key in state_dict:
                state_dict = state_dict[key]
                break
        model.load_state_dict(state_dict)
        log.info("Loaded single-file checkpoint from %s", path)
        return

    # Accelerate FSDP2 sharded checkpoint
    # dcp.load supports no_dist=True since PyTorch 2.3+ (fix: pytorch#115660),
    # and auto-infers it when dist is not initialized since 2.4+ (pytorch#118554).
    fsdp_dir = path / "pytorch_model_fsdp_0"
    if fsdp_dir.exists():
        import torch.distributed.checkpoint as dcp

        state_dict = {"model": model.state_dict()}
        dcp.load(
            state_dict=state_dict,
            storage_reader=dcp.FileSystemReader(str(fsdp_dir)),
            no_dist=True,
        )
        model.load_state_dict(state_dict["model"])
        log.info("Loaded FSDP sharded checkpoint from %s", fsdp_dir)
        return

    # Safetensors / other Accelerate format
    # Source: accelerate.utils.load_checkpoint_in_model
    from accelerate.utils import load_checkpoint_in_model
    load_checkpoint_in_model(model, str(path))
    log.info("Loaded checkpoint from %s", path)
