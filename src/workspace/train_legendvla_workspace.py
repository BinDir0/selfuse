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
from typing import Optional
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
from src.dataset.base_dataset import BaseImageDataset
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
    include_keys = ['global_step', 'update_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: LegendVLA
        self.model = hydra.utils.instantiate(cfg.policy)  # Accelerate will handle DDP
        
        # do not save optimizer if resume=False
        if not cfg.resume:
            self.exclude_keys = ['optimizer']

        self.global_step = 0
        self.update_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        # Set GPU device before initializing accelerator
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            log_with='wandb',
            mixed_precision='bf16' if cfg.training.use_bf16 else 'no',
            device_placement=True,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every
        )

        if accelerator.is_main_process:
            print(f"Using mixed precision: {accelerator.mixed_precision}")
            print(f"Using device: {accelerator.device}")
            print(f"Local rank: {local_rank}")
            if torch.cuda.is_available():
                print(f"CUDA Device: {torch.cuda.get_device_name(local_rank)}")
                print(f"CUDA Capability: {torch.cuda.get_device_capability(local_rank)}")

        # Initialize wandb tracking
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        project_name = wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # Setup model
        if cfg.training.resume_checkpoint_path:
            self.load_checkpoint(cfg.training.resume_checkpoint_path)
        elif cfg.training.load_pretrained_weights:
            self.model.load_pretrained_weights()
        self.model.tie_action_proprio_weights()
        self.model.freeze_unused_weights()
        if cfg.lora:
            self.model.freeze_non_lora_weights_in_vlm()
        
        # Configure optimizers
        self.train_vlm = cfg.training.train_vlm
        model = self.model  # Get unwrapped model for parameter access
        
        # Action optimizer
        self.action_optimizer = bnb.optim.AdamW8bit(
            model.human_action_expert_parameters,
            lr=cfg.optimizer.action.lr,
            weight_decay=cfg.optimizer.action.weight_decay,
        )
        
        # VLM optimizer (if training VLM)
        if self.train_vlm:
            if cfg.lora:
                vlm_trained_parameters = model.lora_trainable_vlm_parameters
            else:
                vlm_trained_parameters = model.trainable_vlm_parameters
            self.vlm_optimizer = bnb.optim.AdamW8bit(
                vlm_trained_parameters,
                lr=cfg.optimizer.vlm.lr,
                weight_decay=cfg.optimizer.vlm.weight_decay,
            )

        # Configure dataset and dataloader
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.dataset)
        train_dataloader = DataLoader(dataset, collate_fn=dataset.get_collator(), **cfg.dataloader)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.policy.cfg.pretrained_model_path, padding_side="right"
        )
        self.processor = PaliGemmaVLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.policy.vision_tower.config.num_image_tokens,
            max_seq_len=cfg.policy.cfg.max_image_text_tokens,
            tokenizer_padding=cfg.tokenizer_padding,
        )
        dataset.set_preprocessor(self.processor)
        train_dataloader = DataLoader(dataset, collate_fn=dataset.get_collator(), **cfg.dataloader)
        
        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))
        self.model.set_normalizer(normalizer)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.get_collator(), **cfg.val_dataloader)

        # Configure learning rate schedulers
        self.action_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.action_optimizer,
            first_cycle_steps=cfg.lr_scheduler.action.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.optimizer.action.lr,
            min_lr=cfg.lr_scheduler.action.min_lr,
            warmup_steps=cfg.lr_scheduler.action.warmup_steps,
            gamma=1.0,
        )
        
        if self.train_vlm:
            self.vlm_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.vlm_optimizer,
                first_cycle_steps=cfg.lr_scheduler.vlm.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.optimizer.vlm.lr,
                min_lr=cfg.lr_scheduler.vlm.min_lr,
                warmup_steps=cfg.lr_scheduler.vlm.warmup_steps,
                gamma=1.0,
            )

        # Compile model if requested
        if cfg.training.use_torch_compile:
            # self.model = torch.compile(self.model, mode="max-autotune")
            self.model = torch.compile(self.model, mode="default")

        # Configure checkpoint manager (if available)
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

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
        self.flow_sampling = cfg.flow.sampling
        if self.flow_sampling == "beta":
            flow_alpha = cfg.flow.alpha if hasattr(cfg.flow, 'alpha') else 1.5
            flow_beta = cfg.flow.beta if hasattr(cfg.flow, 'beta') else 1
            self.flow_t_max = 1 - (cfg.flow.sig_min if hasattr(cfg.flow, 'sig_min') else 0.001)
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        self.model_averaging = ModelAveraging(self.model, cfg, accelerator.device)

        # Training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
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
                            # Gradient clipping
                            if accelerator.sync_gradients and cfg.training.clipping.enabled:
                                total_norm = accelerator.clip_grad_norm_(
                                    self.model.parameters(), 
                                    cfg.training.clipping.max_grad_norm
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

                            if accelerator.sync_gradients:
                                self.update_step += 1
                                # initialize model averaging
                                self.model_averaging.maybe_initialize(self.update_step)
                                # update model averaging
                                self.model_averaging.maybe_update(self.update_step)

                            # Logging
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log.update({
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'update_step': self.update_step,
                                'epoch': self.epoch,
                                'action_lr': self.action_lr_scheduler.get_last_lr()[0],
                            })
                            if self.train_vlm:
                                step_log['vlm_lr'] = self.vlm_lr_scheduler.get_last_lr()[0]

                            is_last_batch = (batch_idx == (len(train_dataloader)-1))
                            if not is_last_batch:
                                accelerator.log(step_log, step=self.update_step)
                                json_logger.log(step_log)

                            if cfg.training.max_train_steps and batch_idx >= (cfg.training.max_train_steps-1):
                                break

                # End of epoch processing
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # Validation
                if (self.epoch % cfg.training.val_every) == 0 and val_dataloader is not None:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.tqdm_interval_sec, 
                                disable=not accelerator.is_main_process) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                inputs = self.preprocess_batch(batch, sample_fm_time=False)
                                loss = self.model(**inputs)
                                val_losses.append(loss)
                                if cfg.training.max_val_steps and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        
                        if len(val_losses) > 0:
                            val_losses = torch.stack(val_losses)
                            val_losses = accelerator.gather(val_losses)
                            
                            if accelerator.is_main_process:
                                val_loss = torch.mean(val_losses).item()
                                step_log['val_loss'] = val_loss

                # Checkpoint saving
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    # Need to update_bn when the model contains batch norm layers !!!
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint()

                # Save model at specific epochs without affecting best model saving
                if self.epoch % cfg.training.ckpt_save_interval == 0 and accelerator.is_main_process:
                    save_dir = os.path.join(self.output_dir, 'epoch_checkpoints')
                    os.makedirs(save_dir, exist_ok=True)
                    # Need to update_bn when the model contains batch norm layers !!!
                    self.save_checkpoint()

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
            "human_action_position_ids": action_position_ids,
            "proprios": proprios,
            "human_actions": actions,
        }
        
        # Sample flow matching timesteps
        if sample_fm_time:
            inputs["t"] = self.sample_fm_time(len(texts))

        return inputs




@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainLegendVLAWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
