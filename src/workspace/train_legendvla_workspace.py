if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
import pickle
from collections import deque
from PIL import Image
import bitsandbytes as bnb
import einops
from transformers import AutoTokenizer

from .base_workspace import BaseWorkspace
from src.policy.legendvla import LegendVLA
from src.dataset.legendvla_dataset import TorchRLDSInterleavedDataset
from src.dataset.paligemma_processing import PaliGemmaVLAProcessor
from src.utils.checkpoint_util import TopKCheckpointManager
from src.utils.json_logger import JsonLogger
from src.model.common.lr_scheduler import get_scheduler
from src.model.common.model_average import ModelAveraging
from src.utils.metric import get_action_accuracy
from src.utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions
from accelerate import Accelerator, DistributedDataParallelKwargs

import wandb

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainLegendVLAWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: LegendVLA
        self.model = LegendVLA(cfg, use_ddp=False)  # Accelerate will handle DDP
        
        # do not save optimizer if resume=False
        if not cfg.resume:
            self.exclude_keys = ['optimizer']

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        # Set GPU device before initializing accelerator
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            log_with='wandb',
            mixed_precision='bf16' if cfg.use_bf16 else 'no',
            device_placement=True,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=cfg.gradient_accumulation_steps
        )

        if accelerator.is_main_process:
            print(f"Using mixed precision: {accelerator.mixed_precision}")
            print(f"Using device: {accelerator.device}")
            print(f"Local rank: {local_rank}")
            if torch.cuda.is_available():
                print(f"CUDA Device: {torch.cuda.get_device_name(local_rank)}")
                print(f"CUDA Capability: {torch.cuda.get_device_capability(local_rank)}")

        # Initialize wandb tracking
        if cfg.wandb:
            wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
            project_name = wandb_cfg.pop('project', 'legendvla-training')
            accelerator.init_trackers(
                project_name=project_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg}
            )

        # Setup model
        if cfg.resume_checkpoint_path:
            self.load_checkpoint(cfg.resume_checkpoint_path)
        elif cfg.load_pretrained_weights:
            self.model.load_pretrained_weights()
        self.model.tie_action_proprio_weights()
        self.model.freeze_unused_weights()
        if cfg.lora:
            self.model.freeze_non_lora_weights_in_vlm()
        
        # Configure optimizers
        self.train_vlm = cfg.train_vlm
        model = self.model  # Get unwrapped model for parameter access
        
        # Action optimizer
        self.action_optimizer = bnb.optim.AdamW8bit(
            model.action_expert_parameters,
            lr=cfg.action_lr,
            weight_decay=cfg.action_weight_decay,
        )
        
        # VLM optimizer (if training VLM)
        if self.train_vlm:
            if cfg.lora:
                vlm_trained_parameters = model.lora_trainable_vlm_parameters
            else:
                vlm_trained_parameters = model.trainable_vlm_parameters
            self.vlm_optimizer = bnb.optim.AdamW8bit(
                vlm_trained_parameters,
                lr=cfg.vlm_lr,
                weight_decay=cfg.vlm_weight_decay,
            )

        # Configure dataset and dataloader
        dataset = TorchRLDSInterleavedDataset(cfg.data.train, train=True).dataset
        
        # Processor for text and image processing
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )
        self.processor = PaliGemmaVLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )
        
        train_dataloader = DataLoader(
            dataset,
            batch_size=cfg.per_device_batch_size,
            pin_memory=True,
        )

        # Configure validation dataset if available
        val_dataloader = None
        if cfg.data.val:
            cfg_data_val = OmegaConf.merge(cfg.data.train, cfg.data.val)
            val_dataset = TorchRLDSInterleavedDataset(cfg_data_val, train=False).dataset
            val_dataloader = DataLoader(
                val_dataset,
                    batch_size=cfg.per_device_batch_size,
                    pin_memory=True,
            )

        # Configure learning rate schedulers
        self.action_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.action_optimizer,
            first_cycle_steps=cfg.action_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.action_lr,
            min_lr=cfg.action_lr_scheduler.min_lr,
            warmup_steps=cfg.action_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        
        if self.train_vlm:
            self.vlm_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.vlm_optimizer,
                first_cycle_steps=cfg.vlm_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.vlm_lr,
                min_lr=cfg.vlm_lr_scheduler.min_lr,
                warmup_steps=cfg.vlm_lr_scheduler.warmup_steps,
                gamma=1.0,
            )

        # Compile model if requested
        if cfg.use_torch_compile:
            self.model = torch.compile(self.model, mode="default")

        # Configure checkpoint manager (if available)
        # topk_manager = None
        # if cfg.checkpoint and cfg.checkpoint.topk:
        #     topk_manager = TopKCheckpointManager(
        #         save_dir=os.path.join(self.output_dir, 'checkpoints'),
        #         **cfg.checkpoint.topk
        #     )

        # Prepare everything with Accelerate
        if self.train_vlm:
            train_dataloader, val_dataloader, self.model, self.action_optimizer, self.vlm_optimizer, self.action_lr_scheduler, self.vlm_lr_scheduler = accelerator.prepare(
                train_dataloader, val_dataloader, self.model, self.action_optimizer, self.vlm_optimizer, self.action_lr_scheduler, self.vlm_lr_scheduler
            )
        else:
            train_dataloader, val_dataloader, self.model, self.action_optimizer, self.action_lr_scheduler = accelerator.prepare(
                train_dataloader, val_dataloader, self.model, self.action_optimizer, self.action_lr_scheduler
            )

        # Flow matching timestep sampling
        self.flow_sampling = cfg.flow_sampling
        if self.flow_sampling == "beta":
            flow_alpha = cfg.flow_alpha if hasattr(cfg, 'flow_alpha') else 1.5
            flow_beta = cfg.flow_beta if hasattr(cfg, 'flow_beta') else 1
            self.flow_t_max = 1 - (cfg.flow_sig_min if hasattr(cfg, 'flow_sig_min') else 0.001)
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

        if cfg.debug:
            cfg.n_epochs = 2
            cfg.max_train_steps = 3
            cfg.max_val_steps = 3

        # Training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.n_epochs):
                self.model.train()
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.tqdm_interval_sec if hasattr(cfg, 'tqdm_interval_sec') else 1.0, 
                        disable=not accelerator.is_main_process) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        with accelerator.accumulate(self.model):
                            # Preprocess batch
                            inputs = self.preprocess_batch(batch, sample_fm_time=True)
                            
                            # Forward pass
                            raw_loss = self.model(**inputs)
                            accelerator.backward(raw_loss)
                            
                            step_log = {}
                            if accelerator.sync_gradients:
                                # Gradient clipping
                                if cfg.max_grad_norm:
                                    total_norm = accelerator.clip_grad_norm_(
                                        self.model.parameters(), 
                                        cfg.max_grad_norm
                                    )
                                    step_log['grad_norm'] = total_norm.item()
                                
                                # Optimizer step
                                self.action_optimizer.step()
                                self.action_lr_scheduler.step()
                                if self.train_vlm:
                                    self.vlm_optimizer.step()
                                    self.vlm_lr_scheduler.step()
                                
                                # Zero gradients
                                self.action_optimizer.zero_grad(set_to_none=True)
                                if self.train_vlm:
                                    self.vlm_optimizer.zero_grad(set_to_none=True)
                                
                                self.global_step += 1
                                
                                # Save model at regular intervals (open-pi-zero style)
                                save_model_freq = cfg.save_model_freq
                                save_model_start = cfg.save_model_start
                                max_updates = cfg.n_updates
                                
                                if ((self.global_step % save_model_freq == 0 and self.global_step > save_model_start) 
                                    or self.global_step >= max_updates):
                                    if accelerator.is_main_process:
                                        # Unwrap model for saving
                                        model_ddp = self.model
                                        self.model = accelerator.unwrap_model(self.model)
                                        
                                        # Save training checkpoint
                                        self.save_training(
                                            cnt_update=self.global_step,
                                            cnt_batch=batch_idx + self.epoch * len(train_dataloader),
                                            main_rank=accelerator.is_main_process
                                        )
                                        
                                        # Restore wrapped model
                                        self.model = model_ddp
                                    
                                    # Wait for all processes
                                    accelerator.wait_for_everyone()
                            
                            # Logging
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log.update({
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'action_lr': self.action_lr_scheduler.get_last_lr()[0],
                            })
                            if self.train_vlm:
                                step_log['vlm_lr'] = self.vlm_lr_scheduler.get_last_lr()[0]

                            is_last_batch = (batch_idx == (len(train_dataloader)-1))
                            if not is_last_batch:
                                accelerator.log(step_log, step=self.global_step)
                                json_logger.log(step_log)

                            if cfg.max_train_steps and batch_idx >= (cfg.max_train_steps-1):
                                break

                # End of epoch processing
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # Validation
                if (self.epoch % cfg.val_every) == 0 and val_dataloader is not None:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.tqdm_interval_sec, 
                                disable=not accelerator.is_main_process) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                inputs = self.preprocess_batch(batch, sample_fm_time=False)
                                loss = self.model(**inputs)
                                val_losses.append(loss)
                                if cfg.max_val_steps and batch_idx >= (cfg.max_val_steps-1):
                                    break
                        
                        if len(val_losses) > 0:
                            val_losses = torch.stack(val_losses)
                            val_losses = accelerator.gather(val_losses)
                            
                            if accelerator.is_main_process:
                                val_loss = torch.mean(val_losses).item()
                                step_log['val_loss'] = val_loss

                # Checkpoint saving
                if (self.epoch % cfg.checkpoint_every) == 0:
                    if accelerator.is_main_process:
                        model_ddp = self.model
                        self.model = accelerator.unwrap_model(self.model)

                        # Save training checkpoint with step-based naming (open-pi-zero style)
                        self.save_training(
                            cnt_update=self.global_step, 
                            cnt_batch=self.global_step,  # Using global_step as batch counter
                            main_rank=accelerator.is_main_process
                        )
                        self.model = model_ddp
                    
                    # Wait for all processes to finish checkpoint saving
                    accelerator.wait_for_everyone()

                # Log final step of epoch
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.epoch += 1

        accelerator.end_training()

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t

    def preprocess_batch(self, batch, sample_fm_time: bool = True):
        """Preprocess batch for training"""
        # Extract data from batch
        images = batch["observation"]["image_primary"]
        proprios = batch["observation"]["proprio"]
        actions = batch["action"].squeeze(1)  # remove the time dimension
        texts = [
            text.decode("utf-8") for text in batch["task"]["language_instruction"]
        ]
        
        # Reshape images
        images = einops.rearrange(
            images, "B T H W C -> B (T C) H W"
        )  # remove cond_steps dimension
        
        # Process with VLA processor
        model_inputs = self.processor(text=texts, images=images)

        # Get unwrapped model for mask building
        model = self.model
        if hasattr(self.model, 'module'):
            model = self.model.module
        
        # Build causal mask and position ids
            causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
                model.build_causal_mask_and_position_ids(
                model_inputs["attention_mask"], torch.bfloat16
            )
            )

            inputs = {
                "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"],
            "causal_mask": causal_mask,
                "vlm_position_ids": vlm_position_ids,
                "proprio_position_ids": proprio_position_ids,
                "action_position_ids": action_position_ids,
            "proprios": proprios,
            "actions": actions,
        }
        
        # Sample flow matching timesteps
        if sample_fm_time:
            inputs["t"] = self.sample_fm_time(len(texts))

        return inputs

    def save_training(self, cnt_update: int, cnt_batch: int):
        """
        Save training state with step-based naming convention.
        Compatible with open-pi-zero training format.
        """
        # Get model averaging state if available
        avg_state = {}
        if hasattr(self, 'model_averaging') and self.model_averaging is not None:
            avg_state = self.model_averaging.state_dict()
        
        model_type = avg_state.get("model_type", "normal")
        n_averaged = avg_state.get("n_averaged", 1)
        
        # Get model weights
        if avg_state and "state_dict" in avg_state:
            weights = avg_state["state_dict"]
        else:
            # Use unwrapped model state dict
            if hasattr(self.model, 'module'):
                weights = self.model.module.state_dict()
            else:
                weights = self.model.state_dict()
        
        # Prepare training data
        data = {
            "cnt_update": cnt_update,
            "cnt_batch": cnt_batch,
            "model": weights,
            "action_optimizer": self.action_optimizer.state_dict(),
            "vlm_optimizer": self.vlm_optimizer.state_dict()
            if self.train_vlm and hasattr(self, 'vlm_optimizer')
            else None,
            "action_lr_scheduler": self.action_lr_scheduler.state_dict(),
            "vlm_lr_scheduler": self.vlm_lr_scheduler.state_dict()
            if self.train_vlm and hasattr(self, 'vlm_lr_scheduler')
            else None,
            "wandb_id": wandb.run.id if hasattr(self, 'use_wandb') and self.use_wandb and wandb.run is not None else None,
            "n_averaged": n_averaged,
        }
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save with step-based naming
        savepath = os.path.join(checkpoint_dir, f"step{cnt_update}.pt")
        torch.save(data, savepath)
        checkpoint_size_in_gb = os.path.getsize(savepath) / (1024**3)
        print(
            f"Saved model to {savepath}, size: {checkpoint_size_in_gb:.2f} GB, type: {model_type}, averaged: {n_averaged}"
        )

    def load_checkpoint(self, path: str):
        """
        Load checkpoint with training state.
        Compatible with open-pi-zero checkpoint format.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load checkpoint data to CPU first
        data = torch.load(path, weights_only=True, map_location="cpu")
        
        # Load training counters
        if "cnt_update" in data:
            self.cnt_update = data["cnt_update"]
        if "cnt_batch" in data:
            self.cnt_batch = data["cnt_batch"]
        if "wandb_id" in data:
            self.wandb_id = data["wandb_id"]
        
        # Handle compiled model keys (remove _orig_mod. prefix)
        model_state_dict = data["model"]
        model_state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()
        }
        
        # Load model state
        self.model.load_state_dict(model_state_dict, strict=True)
        
        print(
            f"Loaded model from {path} at update {getattr(self, 'cnt_update', 'unknown')} batch {getattr(self, 'cnt_batch', 'unknown')}"
        )

    def load_optimizer(self, path: str):
        """
        Load optimizer and scheduler states from checkpoint.
        Compatible with open-pi-zero optimizer loading.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Import optimizer_to utility if available
        try:
            from src.utils.optim import optimizer_to
        except ImportError:
            # Fallback: manual device transfer
            def optimizer_to(optimizer, device):
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
        
        # Load checkpoint data to CPU first
        data = torch.load(path, weights_only=True, map_location="cpu")
        
        # Load action optimizer and scheduler
        if "action_optimizer" in data and data["action_optimizer"] is not None:
            self.action_optimizer.load_state_dict(data["action_optimizer"])
            optimizer_to(self.action_optimizer, self.device)
        
        if "action_lr_scheduler" in data and data["action_lr_scheduler"] is not None:
            self.action_lr_scheduler.load_state_dict(data["action_lr_scheduler"])

        # Load VLM optimizer and scheduler if training VLM
        if self.train_vlm and hasattr(self, 'vlm_optimizer'):
            if "vlm_optimizer" in data and data["vlm_optimizer"] is not None:
                self.vlm_optimizer.load_state_dict(data["vlm_optimizer"])
                optimizer_to(self.vlm_optimizer, self.device)
            
            if "vlm_lr_scheduler" in data and data["vlm_lr_scheduler"] is not None:
                self.vlm_lr_scheduler.load_state_dict(data["vlm_lr_scheduler"])
        
        print(f"Loaded optimizer and scheduler states from {path}")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainLegendVLAWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
