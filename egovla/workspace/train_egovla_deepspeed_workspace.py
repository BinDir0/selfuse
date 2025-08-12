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

from egovla.utils.pytorch_util import dict_apply
from egovla.workspace.base_workspace import BaseWorkspace
from egovla.policy.egovla import EgoVLA
from egovla.dataset.base_dataset import BaseImageDataset
from egovla.dataset.nvila_preprocessor import NVILAPreprocessor
from egovla.utils.checkpoint_util import TopKCheckpointManager
from egovla.utils.json_logger import JsonLogger
from egovla.model.common.lr_scheduler import get_scheduler
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DummyOptim, DummyScheduler
OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainEgoVLAWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: EgoVLA
        self.model = hydra.utils.instantiate(cfg.policy)
    
        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

        if cfg.policy.start_ckpt_path is not None:
            print(f"Starting from checkpoint {cfg.policy.start_ckpt_path}")
            self.load_checkpoint(path=cfg.policy.start_ckpt_path)

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        accelerator = Accelerator(log_with='wandb')
        dtype_map = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
            'no': torch.float32  
        }

        mixed_precision_dtype = dtype_map.get(accelerator.state.mixed_precision)
        print(f"current mixed precision dtype: {mixed_precision_dtype}")
        self.model = self.model.to(mixed_precision_dtype)

        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        wandb_cfg['mode'] = cfg.logging.mode
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # Create optimizer - use DummyOptim if DeepSpeed is configured
        if (
            accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            # Use model's get_optimizer method for regular training
            self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        else:
            # Use DummyOptim for DeepSpeed training
            # Get optimizer parameters using the same logic as get_optimizer method
            param_dict = {pn: p for pn, p in self.model.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optimizer_grouped_parameters = [
                {'params': decay_params, 'weight_decay': cfg.optimizer.weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            self.optimizer = DummyOptim(optimizer_grouped_parameters, 
                                        lr=cfg.optimizer.lr, 
                                        betas=cfg.optimizer.betas, 
                                        fused=True)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.dataset)
        dataset.set_preprocessor(NVILAPreprocessor(
            image_preprocessor=self.model.vlm.vlm.get_vision_tower().image_processor,
            tokenizer=self.model.vlm.vlm.tokenizer
        ))
        train_dataloader = DataLoader(dataset, collate_fn=dataset.get_collator(), **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        if accelerator.is_main_process:
            # 1. main process compute/get object
            normalizer = dataset.get_normalizer()
            objects_to_broadcast = [normalizer]
        else:
            # 2. other process prepare a placeholder
            objects_to_broadcast = [None]

        # 3. broadcast object from main process (from_process=0) to all processes
        objects_to_broadcast = accelerate.utils.broadcast_object_list(objects_to_broadcast, from_process=0)

        # 4. now all processes have a fully identical object copy
        normalizer = objects_to_broadcast[0]

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, collate_fn=dataset.get_collator(), **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        
        # configure lr scheduler - use DummyScheduler if DeepSpeed is configured
        if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            # Use regular scheduler for normal training
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=(
                    len(train_dataloader) * cfg.training.num_epochs),
                # pytorch assumes stepping LRScheduler every epoch
                # however huggingface diffusers steps it every batch
                last_epoch=self.global_step-1
            )
        else:
            # Use DummyScheduler for DeepSpeed training
            max_train_steps = len(train_dataloader) * cfg.training.num_epochs
            lr_scheduler = DummyScheduler(
                self.optimizer, 
                total_num_steps=max_train_steps, 
                warmup_num_steps=cfg.training.lr_warmup_steps
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=not accelerator.is_main_process) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        with accelerator.accumulate(self.model):
                            # compute loss - let DeepSpeed handle BF16 without autocast
                            raw_loss = self.model(batch)
                            accelerator.backward(raw_loss)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            
                            # logging
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }

                            is_last_batch = (batch_idx == (len(train_dataloader)-1))
                            if not is_last_batch:
                                # log of last step is combined with validation and rollout
                                accelerator.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                                self.global_step += 1

                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                                break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                policy.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec, 
                                disable=not accelerator.is_main_process) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                # Let DeepSpeed handle BF16 without autocast
                                loss = self.model(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        
                        if len(val_losses) > 0:
                            # Collect validation losses from all processes
                            val_losses = torch.stack(val_losses)
                            val_losses = accelerator.gather(val_losses)
                            
                            # Calculate mean loss on main process
                            if accelerator.is_main_process:
                                val_loss = torch.mean(val_losses).item()
                                step_log['val_loss'] = val_loss

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

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
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp

                # Save model at specific epochs without affecting best model saving
                if self.epoch % cfg.training.ckpt_save_interval == 0 and accelerator.is_main_process:
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    save_dir = os.path.join(self.output_dir, 'epoch_checkpoints')
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(path=os.path.join(save_dir, f'epoch={self.epoch}.ckpt'))
                    self.model = model_ddp

                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainEgoVLAWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

