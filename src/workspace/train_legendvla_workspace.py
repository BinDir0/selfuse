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
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import TorchDynamoPlugin

from .base_workspace import BaseWorkspace
from src.policy.legendvla import LegendVLA
from src.model.action.fast_tokenizer import UniversalActionProcessor
from src.model.common.lr_scheduler import get_scheduler
from src.model.common.model_average import ModelAveraging
from src.dataset.base_dataset import BaseImageDataset
from src.dataset.paligemma_processing import PaliGemmaVLAProcessor, PaliGemmaProcessor
from src.utils.checkpoint_util import TopKCheckpointManager
from src.utils.json_logger import JsonLogger
from src.utils.metric import get_action_accuracy
from src.utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions
from src.utils.pytorch_util import dict_apply


import wandb

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainLegendVLAWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'update_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: LegendVLA
        self.model = hydra.utils.instantiate(cfg.policy)  # Accelerate will handle DDP
        
        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

        self.dtype = torch.bfloat16 if cfg.training.use_bf16 else torch.float32
        self.global_step = 0
        self.update_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        # # Configure TorchDynamoPlugin
        if cfg.training.use_torch_compile:
            dynamo_plugin = TorchDynamoPlugin(
                backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
                mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
                fullgraph=False,
                dynamic=False
            )

        # Set GPU device before initializing accelerator
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        accelerator = Accelerator(
            log_with='wandb',
            mixed_precision='bf16' if cfg.training.use_bf16 else 'no',
            device_placement=True,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
            dynamo_plugin=dynamo_plugin if cfg.training.use_torch_compile else None
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.policy.cfg.pretrained_model_path, padding_side="right"
        )
        self.fast_tokenizer = UniversalActionProcessor.from_pretrained(
            cfg.processor.fast_tokenizer_path
        )
        
        self.vla_processor = PaliGemmaVLAProcessor(
            self.tokenizer,
            self.fast_tokenizer,
            num_image_tokens=cfg.policy.vision_tower.config.num_image_tokens,
            max_seq_len=cfg.policy.cfg.max_vlm_tokens,
            ignore_index=cfg.ignore_index,
            image_size=cfg.policy.vision_tower.config.image_size,
            state_vocab_size=cfg.processor.state_vocab_size,
            tokenizer_padding=cfg.tokenizer_padding,
        )
        self.vlm_processor = PaliGemmaProcessor(
            self.tokenizer,
            num_image_tokens=cfg.policy.vision_tower.config.num_image_tokens,
            max_seq_len=cfg.policy.cfg.max_vlm_tokens,
            ignore_index=cfg.ignore_index,
            image_size=cfg.policy.vision_tower.config.image_size,
            tokenizer_padding=cfg.tokenizer_padding,
        )
        dataset.vla_dataset.set_preprocessor(self.vla_processor)
        if dataset.vlm_dataset is not None:
            dataset.vlm_dataset.set_preprocessor(self.vlm_processor)
        
        print("Computing normalizer...")
        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.vla_dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        print("Loading normalizer...")
        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))
        self.model.set_normalizer(normalizer)
        dataset.vla_dataset.set_normalizer(normalizer)

        # configure training dataset
        train_dataloader = DataLoader(dataset, collate_fn=dataset.get_collator(), **cfg.dataloader)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.get_collator(), **cfg.val_dataloader)

        # Configure learning rate schedulers
        self.action_lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.action_optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )
        
        if self.train_vlm:
            self.vlm_lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.vlm_optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # Configure checkpoint manager (if available)
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # # Compile model if requested
        # if cfg.training.use_torch_compile:
        #     # self.model = torch.compile(self.model, mode="max-autotune")
        #     self.model = torch.compile(self.model, mode="default")

        # Prepare everything with Accelerate
        if self.train_vlm:
            train_dataloader, val_dataloader, self.model, self.action_optimizer, self.vlm_optimizer, self.action_lr_scheduler, self.vlm_lr_scheduler = accelerator.prepare(
                train_dataloader, val_dataloader, self.model, self.action_optimizer, self.vlm_optimizer, self.action_lr_scheduler, self.vlm_lr_scheduler
            )
        else:
            train_dataloader, val_dataloader, self.model, self.action_optimizer, self.action_lr_scheduler = accelerator.prepare(
                train_dataloader, val_dataloader, self.model, self.action_optimizer, self.action_lr_scheduler
            )

        # wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
        # if accelerator.is_main_process:
        #     wandb_tracker.watch(accelerator.unwrap_model(self.model), log="all", log_freq=10)

        # Flow matching timestep sampling
        self.flow_sampling = cfg.flow.sampling
        if self.flow_sampling == "beta":
            flow_alpha = cfg.flow.get("alpha", 1.5)
            flow_beta = cfg.flow.get("beta", 1)
            self.flow_t_max = 1 - cfg.flow.get("sig_min", 0.001)
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        self.model_averaging = ModelAveraging(accelerator.unwrap_model(self.model), cfg.training.average, accelerator.device)

        # Training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                self.model.train()
                train_losses = dict()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec, 
                        disable=not accelerator.is_main_process) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        with accelerator.accumulate(self.model):
                            # Preprocess batch
                            inputs = self.preprocess_batch(batch, split_mask=False, sample_fm_time=True)
                            
                            # Forward pass
                            raw_loss = self.model("train", inputs)
                            accelerator.backward(raw_loss["total_loss"])

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
                            raw_loss_cpu = dict_apply(raw_loss, lambda x: x.item())
                            tepoch.set_postfix(refresh=False, **raw_loss_cpu)
                            for key, value in raw_loss_cpu.items():
                                if key not in train_losses:
                                    train_losses[key] = list()
                                train_losses[key].append(value)
                            step_log.update({
                                'global_step': self.global_step,
                                'update_step': self.update_step,
                                'epoch': self.epoch,
                                'action_lr': self.action_lr_scheduler.get_last_lr()[0],
                            })
                            step_log.update(raw_loss_cpu)
                            if self.train_vlm:
                                step_log['vlm_lr'] = self.vlm_lr_scheduler.get_last_lr()[0]

                            is_last_batch = (batch_idx == (len(train_dataloader)-1))
                            if not is_last_batch and accelerator.sync_gradients:
                                accelerator.log(step_log, step=self.update_step)
                                json_logger.log(step_log)

                            if cfg.training.max_train_steps and batch_idx >= (cfg.training.max_train_steps-1):
                                break

                # End of epoch processing
                train_loss = dict_apply(train_losses, lambda x: np.mean(x))
                step_log.update(train_loss)

                # Validation
                if (self.epoch % cfg.training.val_every) == 0 and val_dataloader is not None:
                    with torch.no_grad():
                        val_losses = dict()
                        
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec, 
                                disable=not accelerator.is_main_process) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                inputs = self.preprocess_batch(batch, split_mask=False, sample_fm_time=True)
                                
                                # Compute validation loss
                                loss = self.model("train", inputs)
                                for key, loss in loss.items():
                                    if key not in val_losses:
                                        val_losses[key] = list()
                                    val_losses[key].append(loss)
                                
                                if cfg.training.max_val_steps and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        
                        # Process validation loss
                        if len(val_losses) > 0:
                            for key in val_losses.keys():
                                val_losses[key] = torch.stack(val_losses[key])
                                val_losses[key] = accelerator.gather(val_losses[key])
                            
                            if accelerator.is_main_process:
                                for key in val_losses.keys():
                                    val_losses[key] = torch.mean(val_losses[key]).item()
                                    step_log[key] = val_losses[key]

                # Sampling
                if (self.epoch % cfg.training.sample_every) == 0 and val_dataloader is not None:
                    with torch.no_grad():
                        self.model.eval()
                        # Initialize evaluation metrics
                        eval_thresholds = cfg.training.eval_thresholds
                        eval_accuracy = []
                        eval_l1_loss = []
                        
                        with tqdm.tqdm(val_dataloader, desc=f"Sampling epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec, 
                                disable=not accelerator.is_main_process) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                # Preprocess batch
                                inputs = self.preprocess_batch(batch, split_mask=True, sample_fm_time=False)
                                # Compute action accuracy if actions are available
                                if 'human_actions' in inputs:
                                    gt_actions = inputs['human_actions']
                                    human_actions_valid_mask = inputs['human_actions_valid_mask']
                                    # Get action predictions
                                    pred_actions = self.model.forward("infer_human_action", inputs)
                                    
                                    gt_actions = torch.where(human_actions_valid_mask, gt_actions, torch.zeros_like(gt_actions))
                                    pred_actions = torch.where(human_actions_valid_mask, pred_actions, torch.zeros_like(pred_actions))
                                    
                                    # Compute accuracy metrics
                                    batch_accuracy = get_action_accuracy(
                                        gt_actions,
                                        pred_actions,
                                        eval_thresholds,
                                    )
                                    eval_accuracy.append(batch_accuracy)
                                    
                                    # Compute L1 loss
                                    batch_l1_loss = torch.sum(torch.abs(pred_actions - gt_actions)) / torch.sum(human_actions_valid_mask)
                                    eval_l1_loss.append(batch_l1_loss)
                                
                                if cfg.training.max_val_steps and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        
                        # Process action accuracy metrics
                        if len(eval_accuracy) > 0:
                            # Average over batches
                            eval_accuracy = torch.stack(eval_accuracy)
                            eval_l1_loss = torch.stack(eval_l1_loss)
                            
                            # Gather metrics across all processes
                            eval_accuracy = accelerator.gather(eval_accuracy)
                            eval_l1_loss = accelerator.gather(eval_l1_loss)
                            
                            if accelerator.is_main_process:
                                eval_accuracy = torch.mean(eval_accuracy, dim=0)
                                eval_l1_loss = torch.mean(eval_l1_loss)
                                
                                # Log accuracy metrics
                                step_log['eval_l1_loss'] = eval_l1_loss.item()
                                for i, threshold in enumerate(eval_thresholds):
                                    step_log[f'eval_acc_{threshold}'] = eval_accuracy[i].item()
                                
                                # Create log message
                                log_msg = f"Eval | Epoch {self.epoch} | L1 Loss: {eval_l1_loss.item():.3f} | "
                                log_msg += " | ".join([
                                    f"acc thres {threshold}: {eval_accuracy[i].item():.3f}"
                                    for i, threshold in enumerate(eval_thresholds)
                                ])
                                print(log_msg)

                        self.model.train()

                # Checkpoint saving
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
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
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp

                # Save model at specific epochs without affecting best model saving
                if self.epoch % cfg.training.ckpt_save_interval == 0 and accelerator.is_main_process:
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    save_dir = os.path.join(self.output_dir, 'epoch_checkpoints')
                    os.makedirs(save_dir, exist_ok=True)
                    # Need to update_bn when the model contains batch norm layers !!!
                    self.save_checkpoint(path=os.path.join(save_dir, f'epoch_{self.epoch}.ckpt'))
                    self.model = model_ddp

                # Log final step of epoch
                accelerator.log(step_log, step=self.update_step)
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

    def preprocess_batch(self, batch, split_mask: bool = False, sample_fm_time: bool = True):
        """Preprocess batch for training"""
        # Extract data from batch
        pixel_values = batch["pixel_values"]
        human_actions = batch["human_actions"]
        human_actions_valid_mask = batch["human_actions_valid_mask"]

        input_ids = batch["input_ids"]

        # Get unwrapped model for mask building
        model = self.model
        if hasattr(self.model, 'module'):
            model = self.model.module
        
        # Build causal mask and position ids
        # We need to move the new created tensors to the same device as the input prepared by the accelerate
        # get causal mask by the first unignored index of labels 
        causal_mask, vlm_position_ids, human_action_position_ids = (
            model.build_causal_mask_and_position_ids(   
                batch["attention_mask"], batch["answer_start_idx"], self.dtype
            )
        )

        inputs = {
            "input_ids": input_ids,
            "labels": batch["labels"],
            "pixel_values": pixel_values.to(self.dtype),
            "vlm_position_ids": vlm_position_ids,
            "human_action_position_ids": human_action_position_ids,
            "human_actions": human_actions.to(self.dtype),
            "human_actions_valid_mask": human_actions_valid_mask,
        }
        
        if split_mask:
            vlm_mask, human_action_mask = (
                model.split_full_mask_into_submasks(causal_mask)
            )
            inputs["vlm_mask"] = vlm_mask
            inputs["human_action_mask"] = human_action_mask
        else:
            inputs["causal_mask"] = causal_mask

        # Sample flow matching timesteps
        if sample_fm_time:
            # We need to move the new created tensors to the same device as the input prepared by the accelerate
            inputs["t"] = self.sample_fm_time(len(input_ids)).to(self.dtype).to(input_ids.device)

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
