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

from .base_workspace import BaseWorkspace
from src.policy.legendvla import LegendVLA
from src.utils.checkpoint_util import TopKCheckpointManager
from src.model.common.model_average import ModelAveraging
from src.utils.training_utils import (
    TrainingState,
    capture_output_to_training_log,
    DeviceTransferWrapper,
    FullMemoryTracker,
    params_l2_norm,
)

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
        self.compile_cfg = cfg.training.get("compile", {})
        print(f"Training with objective function: {self.objective_func}")

    def maybe_compile_model(self, accelerator):
        enabled = self.compile_cfg.get("enabled", False)
        if not enabled:
            return

        compile_kwargs = OmegaConf.to_container(self.compile_cfg, resolve=True)
        compile_kwargs.pop("enabled", None)

        if accelerator.is_main_process:
            print(f"Compiling model with kwargs: {compile_kwargs}")

        self.model = torch.compile(self.model, **compile_kwargs)
        
    def reset_run_seed(self, accelerator):
        """Reset runtime seed before building dataset/dataloader."""
        base_seed = int(self.cfg.training.seed)
        dynamic_data_seed = bool(self.cfg.training.get("dynamic_data_seed", False))

        timestamp_seed = None
        run_seed = base_seed
        if dynamic_data_seed:
            if accelerator.is_main_process:
                timestamp_seed = int(time.time())
                objects = [timestamp_seed]
            else:
                objects = [None]
            objects = accelerate.utils.broadcast_object_list(objects, from_process=0)
            timestamp_seed = int(objects[0])
            run_seed = base_seed + timestamp_seed

        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        np.random.seed(run_seed % (2**32 - 1))
        random.seed(run_seed)
        self.run_seed = run_seed

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if dynamic_data_seed:
                print(
                    f"Using runtime seed: {run_seed} "
                    f"(base={base_seed}, timestamp={timestamp_seed})"
                )
            else:
                print(f"Using fixed seed: {run_seed}")

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

        # Print accelerator initialization info
        if accelerator.is_main_process:
            print("=" * 80)
            print("Accelerator Initialization Info:")
            print(f"  distributed_type: {accelerator.distributed_type}")
            print(f"  mixed_precision: {accelerator.mixed_precision}")
            print(f"  num_processes: {accelerator.num_processes}")
            print(f"  process_index: {accelerator.process_index}")
            print(f"  device: {accelerator.device}")
            print("=" * 80)

        # Initialize wandb tracking
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        project_name = wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # Broadcast output directory to all processes
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

        # Configure optimizers
        model = self.model

        # Load pretrained weights before optimizer setup
        if cfg.training.load_pretrained_pi05_weights:
            model.load_pretrained_pi05_weights()
        elif cfg.training.load_pretrained_vlm_weights:
            model.load_pretrained_vlm_weights()
        elif cfg.training.finetune_checkpoint_path:
            state_dict = torch.load(cfg.training.finetune_checkpoint_path, map_location='cpu')
            # handle wrapping
            for key in ['module', 'model', 'model_state_dict']:
                if key in state_dict:
                    state_dict = state_dict[key]
                    break
            model.load_state_dict(state_dict)
            print("Successfully loaded finetuning weights.")
        if cfg.lora:
            model.freeze_non_lora_weights_in_vlm()

        self.maybe_compile_model(accelerator)

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

        if cfg.training.train_depth is False:
            model.freeze_weights_in_depth()

        self.grad_stats = {}
        def get_grad_hook(param):
            def hook(grad):
                if grad is not None:
                    self.grad_stats[id(param)] = grad.detach()
                return grad
            return hook
        for _, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(get_grad_hook(param))
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

        self.optimizer = torch.optim.AdamW(all_trainable_parameters, fused=True)

        # ============================================================
        # WebDataset: dataset and dataloader creation
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
        train_dataloader = DataLoader(
            dataset=dataset,
            collate_fn=dataset.get_collator(),
            **cfg.dataloader.loader,
        )
        # Validation dataloader
        val_dataset = dataset.get_validation_dataset()
        val_dataset.distribute(
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            collate_fn=val_dataset.get_collator(),
            **cfg.val_dataloader.loader,
        )

        # Steps per epoch: configured value (streaming has no fixed length)
        steps_per_epoch = cfg.training.get("steps_per_epoch", 100000)

        # Wrap dataloaders with DeviceTransferWrapper (not managed by accelerate)
        train_dataloader = DeviceTransferWrapper(train_dataloader, accelerator.device)
        val_dataloader = DeviceTransferWrapper(val_dataloader, accelerator.device)
        # ============================================================

        # Configure learning rate schedulers
        num_update_steps_per_epoch = math.ceil(steps_per_epoch / accelerator.gradient_accumulation_steps)
        max_train_steps = num_update_steps_per_epoch * cfg.training.num_epochs
        if cfg.training.max_train_steps is not None:
            max_train_steps = cfg.training.max_train_steps
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

        # Configure checkpoint manager
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
            if self.is_deepspeed:
                print(f"  Model type: {type(self.model).__name__}")
                print(f"  Has model.step(): {hasattr(self.model, 'step')}")
                print(f"  Has model.backward(): {hasattr(self.model, 'backward')}")

        # Resume training from checkpoint after accelerator prepare
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

                    # Preprocess batch
                    inputs = self.preprocess_batch(batch, split_mask=False, sample_fm_time=self.objective_func != "train_ar")

                    if batch_idx == 10 and accelerator.is_main_process and cfg.training.profile:
                        self.tracker.track()
                    # Clear hook-captured grad stats for this step
                    self.grad_stats.clear()
                    with accelerator.accumulate(self.model):
                        # Forward pass
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
                        # Gradient clipping
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
                                self.model.parameters(),
                                float('inf')
                            )

                        # Standard training without DeepSpeed optimizer
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
                        # Logging
                        raw_loss_cpu = {}
                        for key, value in raw_loss.items():
                            raw_loss_cpu[key] = value.item()
                        step_wall_time = time.time()
                        step_time_sec = time.perf_counter() - step_perf_start
                        batch_size_local = inputs["input_ids"].shape[0]
                        elapsed_time_sec = step_wall_time - training_start_time
                        step_log.update({
                            'elapsed_time_sec': elapsed_time_sec,
                            'step_time_sec': step_time_sec,
                            'avg_samples_per_sec': total_samples_processed / elapsed_time_sec if elapsed_time_sec > 0 else 0,
                            'samples_per_sec': batch_size_local / step_time_sec if step_time_sec > 0 else 0,
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
                                unwrapped_model.action_expert_parameters
                            )
                            step_log["weight_norm/diffloss"] = params_l2_norm(
                                unwrapped_model.diffloss_parameters
                            )
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

                    if cfg.training.max_train_steps and self.update_step >= cfg.training.max_train_steps:
                        if accelerator.is_main_process:
                            print(f"Max train steps {cfg.training.max_train_steps} reached, stopping training.")
                        break

                    if self.global_step % 100 == 0 and accelerator.is_main_process:
                        print(f"Global step {self.global_step} completed")

                    if cfg.training.profile and accelerator.is_main_process:
                        prof.step()

                if cfg.training.max_train_steps and self.update_step >= cfg.training.max_train_steps:
                    break
                self.epoch += 1

        accelerator.end_training()

    # Combine validation and sampling, so we can process data only once.
    def evaluation(self, accelerator, dataloader, step_log):
        from src.workspace.eval_utils import evaluation
        evaluation(self, accelerator, dataloader, step_log)

    def save_checkpoint_accelerator(self, accelerator, path=None, tag='latest'):
        from src.workspace.eval_utils import save_checkpoint_accelerator
        save_checkpoint_accelerator(self, accelerator, path, tag)

    def save_topk_ckpt(self, accelerator, topk_manager, step_log):
        from src.workspace.eval_utils import save_topk_ckpt
        save_topk_ckpt(self, accelerator, topk_manager, step_log)

    def save_interval_ckpt(self, accelerator):
        from src.workspace.eval_utils import save_interval_ckpt
        save_interval_ckpt(self, accelerator)

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
        bsz = input_ids.shape[0]
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
            "depth_values": batch["depth_values"].to(self.dtype) if "depth_values" in batch else None,
            "has_depth_values": batch["has_depth_values"] if "has_depth_values" in batch else None,
        }
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


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainLegendVLAWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
