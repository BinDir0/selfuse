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
from contextlib import nullcontext, contextmanager
from torch.utils.data import DataLoader
import copy
import random
import numpy as np
import pickle
import time
from datetime import datetime, timedelta
import accelerate
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler, ProfileKwargs, InitProcessGroupKwargs

from .base_workspace import BaseWorkspace
from src.policy.legendvla import LegendVLA
from src.utils.checkpoint_util import TopKCheckpointManager
from src.model.common.model_average import ModelAveraging
from src.utils.metric import get_action_accuracy
from src.utils.training_utils import TrainingState, capture_output_to_training_log, FullMemoryTracker

OmegaConf.register_new_resolver("eval", eval, replace=True)

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
        self.model = hydra.utils.instantiate(cfg.policy) 
        self.tracker = FullMemoryTracker(self.model)
        
        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

        self.dtype = torch.bfloat16 if cfg.training.use_bf16 else torch.float32
        self.training_state = TrainingState()
        self.epoch = 0
        self.update_step = 0
        self.global_step = 0
        if cfg.training.objective is None: 
            self.objective_func = "train"
        else: 
            self.objective_func = "train_" + cfg.training.objective
        print(f"Training with objective function: {self.objective_func}")

    @capture_output_to_training_log
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

        init_process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=3600)
        )
        kwargs_handlers = [init_process_group_kwargs]
        if cfg.training.profile:
            kwargs_handlers.append(profile_kwargs)
        accelerator = Accelerator(
            log_with='wandb', 
            kwargs_handlers=kwargs_handlers
        )

        # Initialize wandb tracking
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        project_name = wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # Broadcast output directory to all processes
        # so that all processes can save checkpoint to the same directory
        if accelerator.is_main_process:
            output_dir = self.output_dir
            objects_to_broadcast = [output_dir]
        else:
            objects_to_broadcast = [None]

        objects_to_broadcast = accelerate.utils.broadcast_object_list(objects_to_broadcast, from_process=0)
        output_dir = objects_to_broadcast[0]
        self._output_dir = output_dir
        accelerator.wait_for_everyone()

        # Configure optimizers
        model = self.model  # Get unwrapped model for parameter access

        # Load pretrained weights and freeze non-lora weights in VLM before deepspeed optimizer setup
        # cause deepspeed will back up the parameters, manually load pretrained weights after setup can't affect these parameters
        if cfg.training.load_pretrained_pi05_weights:
            model.load_pretrained_pi05_weights()
        elif cfg.training.load_pretrained_vlm_weights:
            model.load_pretrained_vlm_weights()
        if cfg.lora:
            model.freeze_non_lora_weights_in_vlm()

        self.model_averaging = ModelAveraging(self.model, cfg.training.average, accelerator.device)
        for key in self.include_keys:
            accelerator.register_for_checkpointing(self.__dict__[key])

        # Action optimizer
        all_trainable_parameters = []
        if self.objective_func != "train_ar":
            all_trainable_parameters = self.get_grouped_parameters(
                model.action_expert_parameters, 
                cfg.optimizer.action, 
            )
        else: 
            model.freeze_non_lora_weights_in_ae()
        
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
        else: 
            model.freeze_non_lora_weights_in_vlm()

        diffloss_trainable_paramters = self.get_grouped_parameters(
            model.diffloss_parameters,
            cfg.optimizer.diffloss,
        )
        all_trainable_parameters.extend(diffloss_trainable_paramters)

        all_trainable_params_list = []
        for params_dict in all_trainable_parameters:
            all_trainable_params_list.extend(params_dict['params'])
        
        trainable_param_ids = {id(p) for p in all_trainable_params_list}

        for i, param in enumerate(all_trainable_params_list):
            assert param.requires_grad, \
                f"Parameter at index {i} is in optimizer groups but requires_grad is False"

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert id(param) in trainable_param_ids, \
                    f"Parameter '{name}' requires grad but is NOT in the optimizer parameters list"
        
        self.optimizer = DummyOptim(all_trainable_parameters)
        
        print("--> Configure dataset and dataloader...................")
        # Configure dataset and dataloader
        dataset = hydra.utils.instantiate(cfg.dataset)
        self.use_relative_action = dataset.vla_dataset.use_relative_action
        print("--> dataset instantiated")
        accelerator.wait_for_everyone()

        self.vla_processor = hydra.utils.instantiate(cfg.vla_processor)
        self.vlm_processor = hydra.utils.instantiate(cfg.vlm_processor)
        dataset.vla_dataset.set_preprocessor(self.vla_processor)
        if dataset.vlm_dataset is not None:
            dataset.vlm_dataset.set_preprocessor(self.vlm_processor)
        # According to the PaliGemma paper, we can initialize the motion token embeddings 
        # to gain better performance.
        # if cfg.training.init_motion_token_embeddings:
        #     self.model.init_motion_token_embeddings(self.vla_processor.total_motion_token_list)
        # Initialize extra token embeddings.
        # self.model.init_motion_token_embeddings([id for id in range(257152, 257216)])

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
        self.normalizer = normalizer

        # configure training dataset
        train_dataloader = DataLoader(
            dataset=dataset, 
            batch_sampler=dataset.get_sampler(**cfg.dataloader.batch_sampler),
            collate_fn=dataset.get_collator(), 
            **cfg.dataloader.loader
        )
        # Accelerate needs to know the batch size. 
        # But Dataloader does not support batch_size argument, when we set batch_sampler, 
        # so we set the batch_size here.
        train_dataloader.__dict__["batch_size"] = cfg.dataloader.batch_sampler.batch_size

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(
            dataset=val_dataset, 
            batch_sampler=val_dataset.get_sampler(**cfg.val_dataloader.batch_sampler),
            collate_fn=val_dataset.get_collator(), 
            **cfg.val_dataloader.loader
        )
        val_dataloader.__dict__["batch_size"] = cfg.val_dataloader.batch_sampler.batch_size

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
            self.update_step = self.training_state.update_step
            self.global_step = self.training_state.global_step
            self.epoch = self.training_state.epoch
            print(f"Skipping {self.global_step % len(train_dataloader)} batches, total batches: {len(train_dataloader)}")
            skipped_dataloader = accelerator.skip_first_batches(train_dataloader, self.global_step % len(train_dataloader))

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
        training_start_time = None  # set at first step
        total_samples_processed = 0
        log_interval = int(getattr(cfg.training, "log_interval", 50))
        with profile_context as prof:
            if accelerator.is_main_process:
                print(f"Training with {len(train_dataloader)} steps per epoch")
            for epoch_idx in range(self.epoch, cfg.training.num_epochs):
                self.model.train()
                if accelerator.is_main_process:
                    print(f"Training epoch {self.epoch} started")
                if epoch_idx == 0 and cfg.training.resume_checkpoint_path: 
                    dataloader = skipped_dataloader
                else:
                    dataloader = train_dataloader
                for batch_idx, batch in enumerate(dataloader):
                    step_perf_start = time.perf_counter()
                    if training_start_time is None:
                        training_start_time = time.time()
                    if cfg.training.profile and torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # Preprocess batch
                    inputs = self.preprocess_batch(batch, split_mask=False, sample_fm_time=self.objective_func != "train_ar")

                    if batch_idx == 10 and accelerator.is_main_process and cfg.training.profile:
                        self.tracker.track()
                    # Forward pass
                    with accelerator.autocast():
                        raw_loss = self.model(self.objective_func, inputs)
                    accelerator.backward(raw_loss["total_loss"])
                    if batch_idx == 10 and accelerator.is_main_process and cfg.training.profile:
                        torch.cuda.empty_cache() 
                        print(torch.cuda.memory_summary())
                        self.tracker.report()
                        self.tracker.stop()

                    # Gradient clipping
                    total_norm = None
                    if accelerator.sync_gradients and cfg.training.clipping.enabled:
                        total_norm = accelerator.clip_grad_norm_(
                            self.model.parameters(), 
                            float('inf')
                        )

                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    # Zero gradients
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    self.global_step += 1

                    if accelerator.sync_gradients:
                        self.update_step += 1
                        total_samples_processed += inputs["input_ids"].shape[0]
                        # initialize model averaging
                        self.model_averaging.maybe_initialize(self.update_step)
                        # update model averaging
                        self.model_averaging.maybe_update(self.update_step)

                    should_record = (
                        accelerator.sync_gradients
                        and (self.update_step % log_interval == 0)
                    )
                    should_eval = (
                        accelerator.sync_gradients
                        and val_dataloader is not None
                        and (self.update_step % cfg.training.eval_every == 0)
                    )
                    should_ckpt = (
                        accelerator.sync_gradients
                        and (self.update_step % cfg.training.checkpoint_every == 0)
                    )
                    should_interval_ckpt = (
                        accelerator.sync_gradients
                        and (self.update_step % cfg.training.ckpt_save_interval == 0)
                    )

                    step_log = None
                    if should_record or should_eval or should_ckpt or should_interval_ckpt:
                        step_log = {
                            'global_step': self.global_step,
                            'update_step': self.update_step,
                            'epoch': self.epoch,
                            'lr': self.lr_scheduler.get_last_lr()[0],
                        }

                    if should_record:
                        # Logging
                        raw_loss_cpu = {}
                        for key, value in raw_loss.items(): 
                            raw_loss_cpu[key] = value.item()
                        step_wall_time = time.time()
                        step_time_sec = time.perf_counter() - step_perf_start
                        batch_size = inputs["input_ids"].shape[0]
                        elapsed_time_sec = step_wall_time - training_start_time
                        step_log.update({
                            'elapsed_time_sec': elapsed_time_sec,
                            'step_time_sec': step_time_sec,
                            'avg_samples_per_sec': total_samples_processed / elapsed_time_sec if elapsed_time_sec > 0 else 0,
                            'samples_per_sec': batch_size / step_time_sec if step_time_sec > 0 else 0,
                        })
                        if total_norm is not None:
                            step_log['grad_norm'] = total_norm
                        step_log.update(raw_loss_cpu)

                    # Evaluation
                    if should_eval:
                        self.evaluation(accelerator, val_dataloader, step_log)

                    # Checkpoint saving
                    if should_ckpt:
                        self.save_topk_ckpt(accelerator, topk_manager, step_log)

                    if should_interval_ckpt:
                        self.save_interval_ckpt(accelerator)

                    if step_log is not None:
                        accelerator.log(step_log, step=self.update_step)
                    
                    if cfg.training.max_train_steps and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                    if self.global_step % 100 == 0 and accelerator.is_main_process:
                        print(f"Global step {self.global_step} completed")

                    if cfg.training.profile and accelerator.is_main_process:
                        prof.step()

                self.epoch += 1

        accelerator.end_training()

    # Combine validation and sampling, so we can process data only once. 
    @torch.compiler.disable() 
    def evaluation(self, accelerator, dataloader, step_log): 
        if accelerator.is_main_process:
            print(f"Evaluation step {self.update_step} started")
        accelerator.wait_for_everyone()
        with torch.no_grad(), eval_with_averaged_model(accelerator, self.model, self.model_averaging):
            val_losses = dict()
            eval_thresholds = self.cfg.training.eval_thresholds
            eval_accuracy = []
            eval_l1_loss = []
            
            # Track min/max loss batches for saving (only store these two)
            min_loss_sample = {
                'loss': float('inf'),
                'attn_weights': None,
                'inputs': None,
                'metadata': None,
            }
            max_loss_sample = {
                'loss': float('-inf'),
                'attn_weights': None,
                'inputs': None,
                'metadata': None,
            }
            for batch_idx, batch in enumerate(dataloader):
                inputs = self.preprocess_batch(batch, split_mask=True, sample_fm_time=True)

                # Compute validation loss
                with accelerator.autocast():
                    loss = self.model(self.objective_func, inputs)
                for key, loss_ in loss.items():
                    if key not in val_losses:
                        val_losses[key] = list()
                    val_losses[key].append(loss_.item())

                if hasattr(self.model, 'module'):
                    model = self.model.module
                else:
                    model = self.model
                full_seq_attn_maps = None 
                if hasattr(model, 'attn_weights') and len(model.attn_weights) > 0:
                    full_seq_attn_maps = torch.stack(model.attn_weights, dim=0) # [num_layers, B, num_heads, seq_len, seq_len]
                # Compute action accuracy if actions are available
                if 'actions' in inputs and self.objective_func != "train_ar":
                    gt_actions = inputs['actions']
                    actions_valid_mask = inputs['actions_valid_mask']
                    # Get action predictions
                    with accelerator.autocast():
                        pred_actions = self.model("infer_action", inputs)
                    
                    # ignore invalid actions
                    B, H, D = gt_actions.shape
                    eval_sample = torch.any(actions_valid_mask.reshape(B, -1), dim=1)
                    if not torch.any(eval_sample):
                        continue
                    actions_valid_mask = actions_valid_mask[eval_sample]
                    if self.use_relative_action: 
                        gt_actions = self.normalizer['actions'].unnormalize(gt_actions[eval_sample])
                        pred_actions = self.normalizer['actions'].unnormalize(pred_actions[eval_sample])
                    else:
                        gt_actions = self.normalizer['motions'].unnormalize(gt_actions[eval_sample])
                        pred_actions = self.normalizer['motions'].unnormalize(pred_actions[eval_sample])
                    gt_actions = gt_actions * actions_valid_mask
                    pred_actions = pred_actions * actions_valid_mask
                    
                    # Compute accuracy metrics
                    batch_accuracy = get_action_accuracy(
                        gt_actions,
                        pred_actions,
                        eval_thresholds,
                    )
                    eval_accuracy.append(batch_accuracy)
                    
                    abs_diff = torch.abs(pred_actions - gt_actions)
                    # Compute L1 loss per sample (not batch average)
                    # Calculate per-sample loss for finding min/max
                    per_sample_l1_loss = torch.sum(
                        abs_diff.flatten(start_dim=1), dim=1
                    ) / torch.sum(actions_valid_mask.flatten(start_dim=1), dim=1)  # [B]
                    
                    # Compute batch-level statistics for logging
                    actions_valid_num = torch.sum(actions_valid_mask)
                    batch_l1_loss = torch.sum(abs_diff) / actions_valid_num
                    eval_l1_loss.append(batch_l1_loss)
                    
                    # Track min/max loss samples for saving
                    if full_seq_attn_maps is not None:
                        # [num_layers, eval_sample, num_heads, seq_len, seq_len]
                        attn_weights = full_seq_attn_maps[:, eval_sample, :, :, :]
                        eval_indices = torch.nonzero(eval_sample, as_tuple=False).squeeze(1)
                        batch_size = inputs["input_ids"].shape[0]

                        def to_numpy(value):
                            if torch.is_tensor(value):
                                return value.detach().float().cpu().numpy()
                            if isinstance(value, np.ndarray):
                                return value
                            return np.array(value)

                        def build_sample_inputs(sample_batch_idx):
                            sample_inputs = {}
                            for key, value in inputs.items():
                                if torch.is_tensor(value) and value.shape[0] == batch_size:
                                    sample_inputs[key] = to_numpy(value[sample_batch_idx])
                                else:
                                    sample_inputs[key] = to_numpy(value)
                            return sample_inputs
                        
                        # Find min/max loss samples in this batch
                        min_idx = torch.argmin(per_sample_l1_loss).item()
                        max_idx = torch.argmax(per_sample_l1_loss).item()

                        min_loss = per_sample_l1_loss[min_idx].item()
                        max_loss = per_sample_l1_loss[max_idx].item()

                        min_batch_idx = eval_indices[min_idx].item()
                        max_batch_idx = eval_indices[max_idx].item()

                        min_metadata = {
                            'batch_idx': batch_idx,
                            'sample_idx': min_batch_idx,
                            'eval_sample_idx': min_idx,
                            'l1_loss': min_loss,
                        }
                        max_metadata = {
                            'batch_idx': batch_idx,
                            'sample_idx': max_batch_idx,
                            'eval_sample_idx': max_idx,
                            'l1_loss': max_loss,
                        }

                        # Update min loss sample
                        if min_loss < min_loss_sample['loss']:
                            min_loss_sample['loss'] = min_loss
                            min_loss_sample['attn_weights'] = attn_weights[:, min_idx, :, :, :].float().cpu()  # [num_layers, num_heads, seq_len, seq_len]
                            min_loss_sample['inputs'] = build_sample_inputs(min_batch_idx)
                            min_loss_sample['metadata'] = min_metadata

                        # Update max loss sample
                        if max_loss > max_loss_sample['loss']:
                            max_loss_sample['loss'] = max_loss
                            max_loss_sample['attn_weights'] = attn_weights[:, max_idx, :, :, :].float().cpu()  # [num_layers, num_heads, seq_len, seq_len]
                            max_loss_sample['inputs'] = build_sample_inputs(max_batch_idx)
                            max_loss_sample['metadata'] = max_metadata
                if self.cfg.training.max_eval_steps and batch_idx >= (self.cfg.training.max_eval_steps-1):
                    break
            
            # Process validation loss
            for key in val_losses.keys():
                num_samples = torch.tensor(len(val_losses[key]), device=accelerator.device)
                if len(val_losses[key]) == 0:
                    val_losses[key] = torch.tensor(0.0, device=accelerator.device)
                else: 
                    val_losses[key] = torch.stack(val_losses[key]).sum()
                total_num_samples = accelerator.reduce(num_samples, reduction='sum')
                val_losses[key] = accelerator.reduce(val_losses[key], reduction='sum') / total_num_samples.clamp(min=1)
                val_losses[key] = val_losses[key].item()
                step_log[f'val_{key}'] = val_losses[key]

            # fill eval_accuracy and eval_l1_loss to the same length as dataloader
            # Note: we assume at least one action dimension is available for evaluation
            eval_len = len(eval_accuracy)
            
            # Process action accuracy metrics
            if eval_len > 0:
                # Average over batches
                sum_eval_accuracy = torch.stack(eval_accuracy).sum(dim=0)
                sum_eval_l1_loss = torch.stack(eval_l1_loss).sum()
            else: 
                sum_eval_accuracy = torch.tensor(0.0, device=accelerator.device)
                sum_eval_l1_loss = torch.tensor(0.0, device=accelerator.device)
            eval_len_tensor = torch.tensor(eval_len, device=accelerator.device)
            # Gather metrics across all processes
            sum_eval_accuracy = accelerator.reduce(sum_eval_accuracy, reduction='sum')
            sum_eval_l1_loss = accelerator.reduce(sum_eval_l1_loss, reduction='sum')
            eval_len_tensor = accelerator.reduce(eval_len_tensor, reduction='sum')
            
            eval_accuracy = sum_eval_accuracy / eval_len_tensor.clamp(min=1)
            eval_l1_loss = sum_eval_l1_loss / eval_len_tensor.clamp(min=1)
            
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
            
            # Save attention weights and inputs for selected samples
            if accelerator.is_main_process and min_loss_sample['attn_weights'] is not None:
                print(f"\nSaving attention weights and inputs for selected samples...")
                
                selected_samples = {
                    'lowest_loss': min_loss_sample,
                    'highest_loss': max_loss_sample,
                }
                
                print(f"Selected samples for visualization:")
                print(f"  Lowest loss: batch {min_loss_sample['metadata']['batch_idx']}, "
                      f"sample {min_loss_sample['metadata']['sample_idx']}, loss={min_loss_sample['loss']:.4f}")
                print(f"  Highest loss: batch {max_loss_sample['metadata']['batch_idx']}, "
                      f"sample {max_loss_sample['metadata']['sample_idx']}, loss={max_loss_sample['loss']:.4f}")
                
                # Visualize each selected sample
                for name, sample_data in selected_samples.items():
                    if sample_data['attn_weights'] is None:
                        continue
                        
                    print(f"\nProcessing {name} sample...")
                    
                    # Create output directory
                    output_dir = os.path.join(
                        self.output_dir, 
                        'attention_visualization',
                        f'step_{self.update_step}',
                        name
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, 'attention_and_inputs.npz')
                    save_payload = dict(sample_data['inputs'])
                    save_payload['attn_weights'] = sample_data['attn_weights'].numpy()
                    save_payload['metadata'] = np.array(sample_data['metadata'], dtype=object)
                    save_payload['update_step'] = np.array(self.update_step)
                    try:
                        np.savez_compressed(output_path, **save_payload)
                        print(f"  Saved to: {output_path}")
                    except Exception as e:
                        print(f"  Error saving attention data: {e}")
                

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
        topk_ckpt_path = topk_manager.get_ckpt_path(accelerator, metric_dict)

        if topk_ckpt_path is not None:
            self.save_checkpoint_accelerator(accelerator, path=topk_ckpt_path)

    def save_interval_ckpt(self, accelerator): 
        save_dir = os.path.join(self.output_dir, 'step_checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        # Need to update_bn when the model contains batch norm layers !!!
        self.save_checkpoint_accelerator(accelerator, path=os.path.join(save_dir, f'update_step_{self.update_step}'))

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
        input_ids = batch["input_ids"]
        # Get unwrapped model for mask building
        model = self.model
        if hasattr(self.model, 'module'):
            model = self.model.module
        
        # Build causal mask and position ids
        # We need to move the new created tensors to the same device as the input prepared by the accelerate
        causal_mask, vlm_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(   
                batch["attention_mask"], batch["answer_start_idx"], batch["n_actions"], self.dtype
            )
        )

        inputs = {
            "input_ids": input_ids,
            "pixel_values": batch["pixel_values"].to(self.dtype),
            "vlm_position_ids": vlm_position_ids,
            "states": batch["states"].to(self.dtype),
            "answer_start_idx": batch["answer_start_idx"],
            "is_vla_data": batch["is_vla_data"],
            "n_states": batch["n_states"],
            "n_actions": batch["n_actions"],
        }
        # Add depth_values if available
        if "depth_values" in batch:
            inputs["depth_values"] = batch["depth_values"].to(self.dtype)
            inputs["has_depth_values"] = batch["has_depth_values"]
        if self.objective_func != "train_ar":
            inputs["action_position_ids"] = action_position_ids
            inputs["actions"] = batch["actions"].to(self.dtype)
            inputs["actions_valid_mask"] = batch["actions_valid_mask"]
        if self.objective_func != "train_flow":
            inputs["labels"] = batch["labels"]
        
        if split_mask:
            max_vlm_tokens = input_ids.shape[-1]
            vlm_mask, action_mask = (
                model.split_full_mask_into_submasks(causal_mask, max_vlm_tokens)
            )
            inputs["vlm_mask"] = vlm_mask
            if self.objective_func != "train_ar":
                inputs["action_mask"] = action_mask
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
