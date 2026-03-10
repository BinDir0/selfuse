"""
WebDataset-based training workspace for LegendVLA.

Inherits from TrainLegendVLAWorkspace and overrides run() to use
IterableDataset (WebDataset) instead of map-style Dataset (Zarr).

Key differences from the parent:
- DataLoader uses batch_size instead of batch_sampler
- Normalizer must be pre-computed (no random access in WebDataset)
- No validation dataloader (streaming doesn't support random split)
- Steps per epoch is configured via cfg instead of len(dataloader)
"""

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
from contextlib import nullcontext
from torch.utils.data import DataLoader
import copy
import random
import numpy as np
import pickle
import time
import math
from datetime import timedelta
from transformers import get_scheduler
import accelerate
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import ProfileKwargs, InitProcessGroupKwargs

from .train_legendvla_deepspeed_workspace import TrainLegendVLAWorkspace
from src.utils.checkpoint_util import TopKCheckpointManager
from src.utils.training_utils import (
    capture_output_to_training_log,
    params_l2_norm,
    DeviceTransferWrapper,
)


class TrainLegendVLAWdsWorkspace(TrainLegendVLAWorkspace):
    """Workspace for WebDataset-based LegendVLA training.

    Overrides run() to replace the dataset/dataloader creation section.
    All other training logic (model setup, optimizer, training loop, eval,
    checkpointing) is inherited from the parent class.
    """

    def evaluation(self, accelerator, dataloader, step_log):
        """Override to add device transfer for val batches not wrapped by accelerate."""
        wrapped = DeviceTransferWrapper(dataloader, accelerator.device)
        super().evaluation(accelerator, wrapped, step_log)

    @capture_output_to_training_log
    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # --- Accelerator setup (identical to parent) ---
        def trace_handler(p):
            output_gpu = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
            print("--- GPU Bottlenecks ---")
            print(output_gpu)
            output_cpu = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
            print("\n--- CPU Bottlenecks ---")
            print(output_cpu)
            p.export_chrome_trace(f"{self.output_dir}/trace/trace_step_{p.step_num}.json")

        if cfg.training.profile:
            profile_kwargs = ProfileKwargs(
                activities=['cpu', 'cuda'],
                schedule_option={"wait": 1, "warmup": 2, "active": 10, "repeat": 3, "skip_first": 50},
                on_trace_ready=trace_handler,
            )
            os.makedirs(f"{self.output_dir}/trace", exist_ok=True)

        init_process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=3600)
        )
        kwargs_handlers = [init_process_group_kwargs]
        if cfg.training.profile:
            kwargs_handlers.append(profile_kwargs)

        self.is_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
        self.is_deepspeed = False

        deepspeed_plugin = None
        if self.is_deepspeed:
            ds_config_file = os.environ.get(
                "ACCELERATE_DEEPSPEED_CONFIG_FILE",
                "src/config/ds_config.json"
            )
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_config_file)

        accelerator = Accelerator(
            log_with='wandb',
            deepspeed_plugin=deepspeed_plugin,
            kwargs_handlers=kwargs_handlers
        )

        if accelerator.is_main_process:
            print("=" * 80)
            print("Accelerator Initialization Info:")
            print(f"  distributed_type: {accelerator.distributed_type}")
            print(f"  mixed_precision: {accelerator.mixed_precision}")
            print(f"  num_processes: {accelerator.num_processes}")
            print(f"  process_index: {accelerator.process_index}")
            print(f"  device: {accelerator.device}")
            print("=" * 80)

        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        project_name = wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        if accelerator.is_main_process:
            output_dir = self.output_dir
            objects_to_broadcast = [output_dir]
        else:
            objects_to_broadcast = [None]
        objects_to_broadcast = accelerate.utils.broadcast_object_list(objects_to_broadcast, from_process=0)
        output_dir = objects_to_broadcast[0]
        self._output_dir = output_dir
        accelerator.wait_for_everyone()

        self.reset_run_seed(accelerator)

        # --- Model & optimizer setup (identical to parent) ---
        model = self.model
        if cfg.training.load_pretrained_pi05_weights:
            model.load_pretrained_pi05_weights()
        elif cfg.training.load_pretrained_vlm_weights:
            model.load_pretrained_vlm_weights()
        if cfg.lora:
            model.freeze_non_lora_weights_in_vlm()

        from src.model.common.model_average import ModelAveraging
        self.model_averaging = ModelAveraging(self.model, cfg.training.average, accelerator.device)
        for key in self.include_keys:
            accelerator.register_for_checkpointing(self.__dict__[key])

        all_trainable_parameters = []
        if self.objective_func != "train_ar":
            all_trainable_parameters = self.get_grouped_parameters(
                model.action_expert_parameters, cfg.optimizer.action)
        else:
            model.freeze_non_lora_weights_in_ae()

        if cfg.training.train_vlm:
            if cfg.lora:
                vlm_trained_parameters = model.lora_trainable_vlm_parameters
            else:
                vlm_trained_parameters = model.trainable_vlm_parameters
            vlm_trainable_parameters = self.get_grouped_parameters(
                vlm_trained_parameters, cfg.optimizer.vlm)
            all_trainable_parameters.extend(vlm_trainable_parameters)
        else:
            model.freeze_non_lora_weights_in_vlm()

        self.grad_stats = {}
        def get_grad_hook(param):
            def hook(grad):
                if grad is not None:
                    self.grad_stats[id(param)] = grad.detach()
                return grad
            return hook
        for _, param in model.named_parameters():
            param.register_hook(get_grad_hook(param))
        diffloss_trainable_paramters = self.get_grouped_parameters(
            model.diffloss_parameters, cfg.optimizer.diffloss)
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

        self.optimizer = torch.optim.AdamW(all_trainable_parameters, fused=True)

        # ============================================================
        # WebDataset-specific: dataset and dataloader creation
        # ============================================================
        print("--> Configure WebDataset dataset and dataloader...")
        dataset = hydra.utils.instantiate(cfg.dataset)
        self.use_relative_action = dataset.vla_dataset.use_relative_action
        print("--> dataset instantiated")
        accelerator.wait_for_everyone()

        self.vla_processor = hydra.utils.instantiate(cfg.vla_processor)
        self.vlm_processor = hydra.utils.instantiate(cfg.vlm_processor)
        dataset.vla_dataset.set_preprocessor(self.vla_processor)
        if dataset.vlm_dataset is not None:
            dataset.vlm_dataset.set_preprocessor(self.vlm_processor)

        # Normalizer must be pre-computed for WebDataset
        print("Loading normalizer...")
        assert cfg.training.normalizer_path is not None, (
            "WebDataset training requires a pre-computed normalizer_path.")
        normalizer = pickle.load(open(cfg.training.normalizer_path, 'rb'))
        dataset.vla_dataset.set_normalizer(normalizer)
        self.normalizer = normalizer

        # Distributed shard splitting: handled at dataset level, not by accelerate
        dataset.distribute(
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )

        # DataLoader for IterableDataset: use batch_size, no batch_sampler
        batch_size = cfg.dataloader.loader.batch_size
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=dataset.get_collator(),
            num_workers=cfg.dataloader.loader.num_workers,
            pin_memory=cfg.dataloader.loader.get("pin_memory", True),
            persistent_workers=cfg.dataloader.loader.get("persistent_workers", True),
            prefetch_factor=cfg.dataloader.loader.get("prefetch_factor", 32),
        )
        train_dataloader.__dict__["batch_size"] = batch_size

        # Validation dataloader (requires val shard URLs in config)
        val_dataset = dataset.get_validation_dataset()
        val_dataset.distribute(
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.get_collator(),
            num_workers=cfg.dataloader.loader.num_workers,
            pin_memory=cfg.dataloader.loader.get("pin_memory", True),
            persistent_workers=False,
        )

        # Steps per epoch: use configured value or default
        steps_per_epoch = cfg.training.get("steps_per_epoch", 100000)
        # ============================================================

        # LR scheduler
        num_update_steps_per_epoch = math.ceil(steps_per_epoch / accelerator.gradient_accumulation_steps)
        max_train_steps = num_update_steps_per_epoch * cfg.training.num_epochs
        max_train_steps = max_train_steps * accelerator.num_processes
        num_warmup_steps = cfg.training.lr_warmup_steps * accelerator.num_processes
        if accelerator.is_main_process:
            print(f"num_warmup_steps: {num_warmup_steps}, max_train_steps: {max_train_steps}")
        self.lr_scheduler = get_scheduler(
            name=cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # Prepare with Accelerate (DataLoader excluded — sharding handled manually)
        self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

        if accelerator.is_main_process:
            print(f"\nTraining with: {'DeepSpeed' if self.is_deepspeed else 'Accelerate (DDP/FSDP)'}")

        # Resume from checkpoint
        if cfg.training.resume_checkpoint_path:
            accelerator.load_state(cfg.training.resume_checkpoint_path)
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
        training_start_time = None
        total_samples_processed = 0
        log_interval = int(getattr(cfg.training, "log_interval", 50))
        with profile_context as prof:
            if accelerator.is_main_process:
                print(f"Training with {steps_per_epoch} steps per epoch (WebDataset streaming)")
            for epoch_idx in range(self.epoch, cfg.training.num_epochs):
                self.model.train()
                if accelerator.is_main_process:
                    print(f"Training epoch {self.epoch} started")
                dataloader = train_dataloader
                for batch_idx, batch in enumerate(dataloader):
                    # Enforce steps_per_epoch limit
                    if batch_idx >= steps_per_epoch:
                        break

                    step_perf_start = time.perf_counter()
                    if training_start_time is None:
                        training_start_time = time.time()
                    if cfg.training.profile and torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # Manual device transfer (DataLoader not wrapped by accelerate)
                    batch = {
                        k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    inputs = self.preprocess_batch(batch, split_mask=False, sample_fm_time=self.objective_func != "train_ar")

                    if batch_idx == 10 and accelerator.is_main_process and cfg.training.profile:
                        self.tracker.track()
                    self.grad_stats.clear()
                    with accelerator.accumulate(self.model):
                        with accelerator.autocast():
                            raw_loss = self.model(self.objective_func, inputs)

                        if self.is_deepspeed:
                            self.model.backward(raw_loss["total_loss"])
                        else:
                            accelerator.backward(raw_loss["total_loss"])
                        if batch_idx == 10 and accelerator.is_main_process and cfg.training.profile:
                            torch.cuda.empty_cache()
                            print(torch.cuda.memory_summary())
                            self.tracker.report()
                            self.tracker.stop()

                        should_record = (
                            accelerator.sync_gradients
                            and (self.update_step % log_interval == 0)
                        )
                        total_norm = None
                        part_grad_norms = None
                        if accelerator.sync_gradients and should_record:
                            part_grad_norms = {}
                            unwrapped_model = accelerator.unwrap_model(self.model)
                            part_params = {
                                "action_expert": unwrapped_model.action_expert_parameters,
                                "vlm": unwrapped_model.lora_trainable_vlm_parameters if cfg.lora else unwrapped_model.trainable_vlm_parameters,
                                "diffloss": unwrapped_model.diffloss_parameters,
                            }
                            def grad_stats_l2_norm(params):
                                total_sq = None
                                for param in params:
                                    grad = self.grad_stats.get(id(param))
                                    if grad is None:
                                        continue
                                    grad = grad.float()
                                    sq = torch.sum(grad * grad)
                                    total_sq = sq if total_sq is None else total_sq + sq
                                if total_sq is None:
                                    return None
                                return torch.sqrt(total_sq)
                            for name, params in part_params.items():
                                part_grad_norms[name] = grad_stats_l2_norm(params)
                        if accelerator.sync_gradients and cfg.training.clipping.enabled:
                            total_norm = accelerator.clip_grad_norm_(
                                self.model.parameters(), float('inf'))

                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1
                    if accelerator.sync_gradients:
                        self.update_step += 1
                        total_samples_processed += inputs["input_ids"].shape[0]
                        self.model_averaging.maybe_initialize(self.update_step)
                        self.model_averaging.maybe_update(self.update_step)

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
                        if self.is_deepspeed:
                            current_lr = self.model.get_lr()[0]
                        else:
                            current_lr = self.lr_scheduler.get_last_lr()[0]
                        step_log = {
                            'global_step': self.global_step,
                            'update_step': self.update_step,
                            'epoch': self.epoch,
                            'lr': current_lr,
                        }

                    if should_record:
                        raw_loss_cpu = {}
                        for key, value in raw_loss.items():
                            raw_loss_cpu[key] = value.item()
                        step_wall_time = time.time()
                        step_time_sec = time.perf_counter() - step_perf_start
                        bs = inputs["input_ids"].shape[0]
                        elapsed_time_sec = step_wall_time - training_start_time
                        step_log.update({
                            'elapsed_time_sec': elapsed_time_sec,
                            'step_time_sec': step_time_sec,
                            'avg_samples_per_sec': total_samples_processed / elapsed_time_sec if elapsed_time_sec > 0 else 0,
                            'samples_per_sec': bs / step_time_sec if step_time_sec > 0 else 0,
                        })
                        if total_norm is not None:
                            step_log['grad_norm'] = total_norm
                        if part_grad_norms is not None:
                            step_log.update({
                                'grad_norm_action_expert': part_grad_norms["action_expert"],
                                'grad_norm_vlm': part_grad_norms["vlm"],
                                'grad_norm_diffloss': part_grad_norms["diffloss"],
                            })
                        with torch.no_grad():
                            unwrapped_model = accelerator.unwrap_model(self.model)
                            if cfg.training.train_vlm:
                                vlm_params = (
                                    unwrapped_model.lora_trainable_vlm_parameters
                                    if cfg.lora
                                    else unwrapped_model.trainable_vlm_parameters
                                )
                                step_log["weight_norm/vlm"] = params_l2_norm(vlm_params)
                            step_log["weight_norm/action"] = params_l2_norm(
                                unwrapped_model.action_expert_parameters)
                            step_log["weight_norm/diffloss"] = params_l2_norm(
                                unwrapped_model.diffloss_parameters)
                        step_log.update(raw_loss_cpu)

                    if should_eval:
                        self.evaluation(accelerator, val_dataloader, step_log)

                    if should_ckpt:
                        self.save_topk_ckpt(accelerator, topk_manager, step_log)

                    if should_interval_ckpt:
                        self.save_interval_ckpt(accelerator)

                    if step_log is not None:
                        accelerator.log(step_log, step=self.update_step)

                    if cfg.training.max_train_steps and batch_idx >= (cfg.training.max_train_steps - 1):
                        break

                    if self.global_step % 100 == 0 and accelerator.is_main_process:
                        print(f"Global step {self.global_step} completed")

                    if cfg.training.profile and accelerator.is_main_process:
                        prof.step()

                self.epoch += 1

        accelerator.end_training()
