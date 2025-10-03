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
from contextlib import nullcontext, contextmanager
from torch.utils.data import DataLoader
import copy
import random
import numpy as np
from torch import nn
import pickle
from transformers import AutoTokenizer
import accelerate
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler, ProfileKwargs

from .base_workspace import BaseWorkspace
from src.policy.legendvla import LegendVLA
from src.dataset.base_dataset import BaseImageDataset
from src.dataset.paligemma_processing import PaliGemmaVLAProcessor, PaliGemmaProcessor
from src.utils.checkpoint_util import TopKCheckpointManager
from src.utils.json_logger import JsonLogger
from src.utils.pytorch_util import dict_apply
from src.model.common.model_average import ModelAveraging
from src.model.action.fast_tokenizer import UniversalActionProcessor
from src.utils.metric import get_action_accuracy
from src.utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainingState:
    """A simple class to encapsulate all scalar training states that need to be saved."""
    def __init__(self, epoch: int = 0, update_step: int = 0, global_step: int = 0):
        self.epoch = epoch
        self.update_step = update_step
        self.global_step = global_step

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "update_step": self.update_step,
            "global_step": self.global_step,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.update_step = state_dict["update_step"]
        self.global_step = state_dict["global_step"]


class TrainLegendVLAWorkspace(BaseWorkspace):
    include_keys = ['training_state', 'model_averaging']

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
        self.training_state = TrainingState()
        self.epoch = 0
        self.update_step = 0
        self.global_step = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        def trace_handler(p):
            # sort by GPU time, find GPU bottleneck
            output_gpu = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
            print("--- GPU Bottlenecks ---")
            print(output_gpu)

            # sort by CPU time, find CPU bottleneck
            output_cpu = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
            print("\n--- CPU Bottlenecks ---")
            print(output_cpu)

            '''
            # sort by GPU memory usage, find GPU memory bottleneck
            output_gpu_mem = p.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
            print("\n--- GPU Memory Consumption ---")
            print(output_gpu_mem)
            '''
            
            p.export_chrome_trace(f"{self.output_dir}/trace/trace_step_{p.step_num}.json")

        if cfg.training.profile: 
            profile_kwargs = ProfileKwargs(
                activities=['cpu', 'cuda'],
                schedule_option={"wait": 1, "warmup": 2, "active": 10, "repeat": 3, "skip_first": 50},
                on_trace_ready=trace_handler, 
                # profile_memory=True,  # enable memory analysis
                # with_stack=True
            )
            os.makedirs(f"{self.output_dir}/trace", exist_ok=True)

        accelerator = Accelerator(
            log_with='wandb', 
            kwargs_handlers=[profile_kwargs] if cfg.training.profile else None
        )

        # Initialize wandb tracking
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        project_name = wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # Configure optimizers
        model = self.model  # Get unwrapped model for parameter access

        # Load pretrained weights and freeze non-lora weights in VLM before deepspeed optimizer setup
        # cause deepspeed will back up the parameters, manually load pretrained weights can't affect these parameters
        if cfg.training.load_pretrained_weights:
            model.load_pretrained_weights()
        if cfg.lora:
            model.freeze_non_lora_weights_in_vlm()

        self.model_averaging = ModelAveraging(self.model, cfg.training.average, accelerator.device)
        for key in self.include_keys:
            accelerator.register_for_checkpointing(self.__dict__[key])

        # Action optimizer
        all_trainable_parameters = self.get_grouped_parameters(
            model.human_action_expert_parameters, 
            cfg.optimizer.action, 
        )
        
        # VLM optimizer (if training VLM)
        if cfg.training.train_vlm:
            if cfg.lora:
                vlm_trained_parameters = model.lora_trainable_vlm_parameters
            else:
                vlm_trained_parameters = model.trainable_vlm_parameters
            vlm_trainable_parameters = self.get_grouped_parameters(
                vlm_trained_parameters, 
                cfg.optimizer.vlm, 
            )
            all_trainable_parameters.extend(vlm_trainable_parameters)

        self.optimizer = DummyOptim(
            all_trainable_parameters, 
        )
        
        print("--> Configure dataset and dataloader...................")
        # Configure dataset and dataloader
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.dataset)
        print("--> dataset instantiated")
        accelerator.wait_for_everyone()

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.policy.cfg.pretrained_model_path, padding_side="right"
        )
        self.fast_tokenizer = {
            "states": UniversalActionProcessor.from_pretrained(
                os.path.join(cfg.processor.fast_tokenizer_path, "states")
            ),
            "human_actions": UniversalActionProcessor.from_pretrained(
                os.path.join(cfg.processor.fast_tokenizer_path, "human_actions")
            )
        }
        
        self.vla_processor = PaliGemmaVLAProcessor(
            self.tokenizer,
            self.fast_tokenizer,
            num_image_tokens=cfg.policy.vision_tower.config.num_image_tokens,
            max_seq_len=cfg.policy.cfg.max_vlm_tokens,
            ignore_index=cfg.ignore_index,
            image_size=cfg.policy.vision_tower.config.image_size,
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
        if cfg.training.normalizer_path is not None:
            normalizer = pickle.load(open(cfg.training.normalizer_path, 'rb'))
        else:
            # compute normalizer on the main process and save to disk
            if accelerator.is_main_process:
                # 1. main process compute/get object
                normalizer = dataset.vla_dataset.get_normalizer()
                normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
                pickle.dump(normalizer, open(normalizer_path, 'wb'))
                objects_to_broadcast = [normalizer]
            else:
                # 2. other process prepare a placeholder
                objects_to_broadcast = [None]

            # 3. broadcast object from main process (from_process=0) to all processes
            objects_to_broadcast = accelerate.utils.broadcast_object_list(objects_to_broadcast, from_process=0)
            normalizer = objects_to_broadcast[0]

        # 4. now all processes have a fully identical object copy
        dataset.vla_dataset.set_normalizer(normalizer)

        # configure training dataset
        train_dataloader = DataLoader(dataset, collate_fn=dataset.get_collator(), **cfg.dataloader)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.get_collator(), **cfg.val_dataloader)

        # Configure learning rate schedulers
        max_train_steps = len(train_dataloader) * cfg.training.num_epochs
        self.lr_scheduler = DummyScheduler(
            optimizer=self.optimizer,
            warmup_num_steps=cfg.training.lr_warmup_steps,
            total_num_steps=max_train_steps,
        )

        # Configure checkpoint manager (if available)
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # Prepare everything with Accelerate
        train_dataloader, val_dataloader, self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, self.lr_scheduler
        )

        # resume training from checkpoint after accelerator prepare
        if cfg.training.resume_checkpoint_path:
            accelerator.load_state(cfg.training.resume_checkpoint_path)
            train_dataloader = accelerator.skip_first_batches(train_dataloader, self.global_step % len(train_dataloader))
            self.update_step = self.training_state.update_step
            self.global_step = self.training_state.global_step
            self.epoch = self.training_state.epoch

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

        profile_context = nullcontext()
        if cfg.training.profile and accelerator.is_main_process:
            profile_context = accelerator.profile()

        # Training loop
        if accelerator.is_main_process:
            print(f"Training with {len(train_dataloader)} steps per epoch")
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger, profile_context as prof:
            for _ in range(cfg.training.num_epochs):
                self.model.train()
                step_log = dict()
                train_losses = dict()
                if accelerator.is_main_process:
                    print(f"Training epoch {self.epoch} started")
                for batch_idx, batch in enumerate(train_dataloader):
                    with accelerator.accumulate(self.model):
                        # Preprocess batch
                        inputs = self.preprocess_batch(batch, split_mask=False, sample_fm_time=True)

                        # Forward pass
                        with accelerator.autocast():
                            raw_loss = self.model("train", inputs)
                        accelerator.backward(raw_loss["total_loss"])

                        # Gradient clipping
                        if accelerator.sync_gradients and cfg.training.clipping.enabled:
                            total_norm = accelerator.clip_grad_norm_(
                                self.model.parameters(), 
                                float('inf')
                            )
                            step_log['grad_norm'] = total_norm

                        # Optimizer step
                        self.optimizer.step()
                        self.lr_scheduler.step()

                        # Zero gradients
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        self.global_step += 1

                        if accelerator.sync_gradients:
                            self.update_step += 1
                            # initialize model averaging
                            self.model_averaging.maybe_initialize(self.update_step)
                            # update model averaging
                            self.model_averaging.maybe_update(self.update_step)

                        # Logging
                        raw_loss_cpu = dict_apply(raw_loss, lambda x: x.item())
                        for key, value in raw_loss_cpu.items():
                            if key not in train_losses:
                                train_losses[key] = []
                            train_losses[key].append(value)
                        step_log.update({
                            'global_step': self.global_step,
                            'update_step': self.update_step,
                            'epoch': self.epoch,
                            'lr': self.lr_scheduler.get_last_lr()[0],
                        })
                        step_log.update(raw_loss_cpu)

                        # Validation
                        if (self.update_step % cfg.training.val_every) == 0 and \
                            val_dataloader is not None and accelerator.sync_gradients:
                            self.validation(accelerator, val_dataloader, step_log)

                        if (self.update_step % cfg.training.sample_every) == 0 and \
                            val_dataloader is not None and accelerator.sync_gradients:
                            self.sample(accelerator, val_dataloader, step_log)

                        # Checkpoint saving
                        if (self.update_step % cfg.training.checkpoint_every) == 0 and accelerator.sync_gradients:
                            self.save_topk_ckpt(accelerator, topk_manager, step_log)

                        if self.update_step % cfg.training.ckpt_save_interval == 0 and accelerator.sync_gradients:
                            self.save_interval_ckpt(accelerator)

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch and accelerator.sync_gradients:
                            accelerator.log(step_log, step=self.update_step)
                            json_logger.log(step_log)

                        if cfg.training.max_train_steps and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                        if self.global_step % 100 == 0 and accelerator.is_main_process:
                            print(f"Global step {self.global_step} completed")

                        if cfg.training.profile and accelerator.is_main_process:
                            prof.step()

                # End of epoch processing
                train_loss = dict_apply(train_losses, lambda x: np.mean(x))
                step_log.update(train_loss)

                # Log final step of epoch
                accelerator.log(step_log, step=self.update_step)
                json_logger.log(step_log)
                self.epoch += 1

        accelerator.end_training()

    def validation(self, accelerator, dataloader, step_log): 
        if accelerator.is_main_process:
            print(f"Validation step {self.update_step} started")
        with torch.no_grad(), eval_with_averaged_model(accelerator, self.model, self.model_averaging):
            val_losses = dict()
            
            for batch_idx, batch in enumerate(dataloader):
                inputs = self.preprocess_batch(batch, split_mask=False, sample_fm_time=True)

                # Compute validation loss
                with accelerator.autocast():
                    loss = self.model("train", inputs)
                for key, loss in loss.items():
                    if key not in val_losses:
                        val_losses[key] = list()
                    val_losses[key].append(loss)
                
                if self.cfg.training.max_val_steps and batch_idx >= (self.cfg.training.max_val_steps-1):
                    break
            
            # Process validation loss
            if len(val_losses) > 0:
                for key in val_losses.keys():
                    val_losses[key] = torch.stack(val_losses[key])
                    val_losses[key] = accelerator.gather_for_metrics(val_losses[key])
                
                for key in val_losses.keys():
                    val_losses[key] = torch.mean(val_losses[key]).item()
                    step_log[f'val_{key}'] = val_losses[key]

    def sample(self, accelerator, dataloader, step_log):
        if accelerator.is_main_process:
            print(f"Sampling step {self.update_step} started")
        with torch.no_grad(), eval_with_averaged_model(accelerator, self.model, self.model_averaging):
            # Initialize evaluation metrics
            eval_thresholds = self.cfg.training.eval_thresholds
            eval_accuracy = []
            eval_l1_loss = []
            
            for batch_idx, batch in enumerate(dataloader):
                # Preprocess batch
                inputs = self.preprocess_batch(batch, split_mask=True, sample_fm_time=False)
                # Compute action accuracy if actions are available
                if 'human_actions' in inputs:
                    gt_actions = inputs['human_actions']
                    human_actions_valid_mask = inputs['human_actions_valid_mask']
                    # Get action predictions
                    with accelerator.autocast():
                        pred_actions = self.model("infer_human_action", inputs)
                    
                    # ignore invalid actions
                    B, _, _ = gt_actions.shape
                    eval_sample = torch.any(human_actions_valid_mask.reshape(B, -1), dim=1)
                    if not torch.any(eval_sample):
                        continue
                    human_actions_valid_mask = human_actions_valid_mask[eval_sample]
                    gt_actions = gt_actions[eval_sample]
                    pred_actions = pred_actions[eval_sample]
                    gt_actions = gt_actions * human_actions_valid_mask
                    pred_actions = pred_actions * human_actions_valid_mask
                    
                    # Compute accuracy metrics
                    batch_accuracy = get_action_accuracy(
                        gt_actions,
                        pred_actions,
                        eval_thresholds,
                    )
                    eval_accuracy.append(batch_accuracy)
                    
                    # Compute L1 loss, num should not be 0 here since we have checked eval_sample
                    human_actions_valid_num = torch.sum(human_actions_valid_mask)
                    batch_l1_loss = torch.sum(torch.abs(pred_actions - gt_actions)) / human_actions_valid_num
                    eval_l1_loss.append(batch_l1_loss)
                
                if self.cfg.training.max_val_steps and batch_idx >= (self.cfg.training.max_val_steps-1):
                    break
            
            # fill eval_accuracy and eval_l1_loss to the same length as dataloader
            eval_len, data_len = len(eval_accuracy), len(dataloader)
            while eval_len < data_len: 
                idx = random.randint(0, eval_len-1)
                eval_accuracy.append(eval_accuracy[idx])
                eval_l1_loss.append(eval_l1_loss[idx])
                eval_len += 1
            
            # Process action accuracy metrics
            if len(eval_accuracy) > 0:
                # Average over batches
                eval_accuracy = torch.stack(eval_accuracy)
                eval_l1_loss = torch.stack(eval_l1_loss)
                
                # Gather metrics across all processes
                eval_accuracy = accelerator.gather_for_metrics(eval_accuracy)
                eval_l1_loss = accelerator.gather_for_metrics(eval_l1_loss)
                
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
                if accelerator.is_main_process:
                    print(log_msg)

    def save_checkpoint_accelerator(self, accelerator, path=None, tag='latest'):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}')
        else:
            path = pathlib.Path(path)
        path.parent.mkdir(parents=False, exist_ok=True)
        self.training_state.update_step = self.update_step
        self.training_state.global_step = self.global_step
        self.training_state.epoch = self.epoch
        accelerator.save_state(path)

    def save_topk_ckpt(self, accelerator, topk_manager, step_log): 
        # Need to update_bn when the model contains batch norm layers !!!
        if self.cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint_accelerator(accelerator)

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
            self.save_checkpoint_accelerator(accelerator, path=topk_ckpt_path)

    def save_interval_ckpt(self, accelerator): 
        save_dir = os.path.join(self.output_dir, 'step_checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        # Need to update_bn when the model contains batch norm layers !!!
        self.save_checkpoint_accelerator(accelerator, path=os.path.join(save_dir, f'step_{self.update_step}.ckpt'))

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
            inputs["t"] = self.sample_fm_time(len(input_ids)).to(input_ids.device).to(self.dtype)

        return inputs

    def get_grouped_parameters(self, param_list, cfg):
        '''
        Args:
            param_list: list of parameters from some part of the model
            cfg: config
        Returns:
            optimizer_grouped_parameters: list of parameter groups
        '''
        param_list = [p for p in param_list if p.requires_grad]
        decay_params = [p for p in param_list if p.dim() >= 2]
        nodecay_params = [p for p in param_list if p.dim() < 2]
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': cfg.weight_decay, 'lr': cfg.lr, 'betas': cfg.betas},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': cfg.lr, 'betas': cfg.betas}
        ]
        return optimizer_grouped_parameters


@contextmanager
def eval_with_averaged_model(accelerator, model, averaged_model):
    """
    A context manager to temporarily load averaged weights into the main model during evaluation.
    """
    if averaged_model.model_avg is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Use .clone() to avoid affecting the original dictionary
        # Move to CPU to avoid GPU memory issues
        device = next(iter(unwrapped_model.parameters())).device
        original_state_dict = {k: v.clone().to('cpu') for k, v in unwrapped_model.state_dict().items()}
        
        averaged_state_dict = averaged_model.averaged_model_state_dict() 
        unwrapped_model.load_state_dict(averaged_state_dict)
    model.eval()
    
    try:
        yield
    finally:
        if averaged_model.model_avg is not None:
            unwrapped_model.load_state_dict(original_state_dict)
            unwrapped_model.to(device)
        model.train()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainLegendVLAWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
