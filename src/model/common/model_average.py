import logging

import torch
import torch.nn as nn
import accelerate

log = logging.getLogger(__name__)


class ModelAveraging:
    """Model averaging with EMA and SWA support. Now supports resume from checkpoint."""

    def __init__(self, model, cfg, device):
        self.use_ema = cfg.ema.enabled
        self.use_swa = cfg.swa.enabled
        assert not (self.use_ema and self.use_swa), (
            "Cannot use both EMA and SWA at once"
        )

        self.model_avg = None
        self.model = model
        self.device = device

        # EMA configuration
        if self.use_ema:
            self.ema_start = cfg.ema.start
            self.ema_decay = cfg.ema.decay
            self.ema_freq = cfg.ema.freq
            self.ema_device = cfg.ema.get("device", device)

        # SWA configuration
        if self.use_swa:
            self.swa_start = cfg.swa.start
            self.swa_freq = cfg.swa.freq
            self.swa_device = cfg.swa.get("device", device)

    def maybe_initialize(self, cnt_update):
        if self.use_swa and cnt_update == self.swa_start:
            self.model_avg = torch.optim.swa_utils.AveragedModel(
                self.model, device=self.swa_device
            )
            logging.info("Starting SWA...")

        if self.use_ema and cnt_update == self.ema_start:
            self.model_avg = torch.optim.swa_utils.AveragedModel(
                self.model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.ema_decay),
                device=self.ema_device,
            )
            logging.info(f"Starting EMA with decay {self.ema_decay}...")

    def maybe_update(self, cnt_update):
        if self.model_avg is None:
            return
        if self.use_ema and cnt_update % self.ema_freq == 0:
            self.model_avg.update_parameters(self.model.to(self.ema_device))
        if self.use_swa and cnt_update % self.swa_freq == 0:
            self.model_avg.update_parameters(self.model.to(self.swa_device))

    def get_unwrapped_averaged_model(self) -> nn.Module:
        if self.model_avg:
            return self.model_avg.module
        return self.model

    def state_dict(self) -> dict:
        if self.model_avg:
            return {
                "state_dict": self.model_avg.module.state_dict(),
                "n_averaged": self.model_avg.state_dict().get("n_averaged", 1),
                "model_type": "ema" if self.use_ema else "swa",
            }
        return {}
    
    def load_state_dict(self, state_dict: dict):
        """
        Load state dict for model averaging.
        This enables compatibility with save_checkpoint method.
        """
        if not state_dict:
            # Empty state dict means no model averaging was used
            self.model_avg = None
            return
            
        model_type = state_dict.get("model_type", "normal")
        n_averaged = state_dict.get("n_averaged", 1)
        
        if model_type in ["ema", "swa"] and "state_dict" in state_dict:
            # Create appropriate averaged model based on type
            if model_type == "ema" and self.use_ema:
                self.model_avg = torch.optim.swa_utils.AveragedModel(
                    self.model,
                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.ema_decay),
                    device=self.ema_device,
                )
                logging.info(f"Loaded EMA model with {n_averaged} averaged updates")
            elif model_type == "swa" and self.use_swa:
                self.model_avg = torch.optim.swa_utils.AveragedModel(
                    self.model, device=self.swa_device
                )
                logging.info(f"Loaded SWA model with {n_averaged} averaged updates")
            else:
                logging.warning(f"Model averaging type mismatch: saved={model_type}, current=ema:{self.use_ema}/swa:{self.use_swa}")
                self.model_avg = None
                return
                
            # Load the averaged model weights
            self.model_avg.module.load_state_dict(state_dict["state_dict"])
            
            # Restore n_averaged counter if available
            if hasattr(self.model_avg, 'n_averaged'):
                self.model_avg.n_averaged = torch.tensor(n_averaged)
        else:
            logging.warning("Invalid state dict format for model averaging")
            self.model_avg = None
