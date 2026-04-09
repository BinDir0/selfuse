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
from accelerate import Accelerator
from accelerate.utils import ProfileKwargs, InitProcessGroupKwargs
from src.workspace.eval_utils import _unwrap_model
from .base_workspace import BaseWorkspace
from src.policy.legendvla import LegendVLA
from src.utils.checkpoint_util import TopKCheckpointManager
from src.model.common.model_average import ModelAveraging
from src.utils.training_utils import (
    TrainingState,
    capture_output_to_training_log,
    DeviceTransferWrapper,
    FullMemoryTracker,
    GarbageCollection,
    params_l2_norm,
    scalar_metric_value,
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
        gc_cfg = cfg.training.get("gradient_checkpointing", False)
        if isinstance(gc_cfg, bool):
            # Backward compat: True -> all components, every_n=1
            if gc_cfg:
                self.model.enable_gradient_checkpointing()
        else:
            # Structured per-component config
            gc_dict = OmegaConf.to_container(gc_cfg, resolve=True)
            has_any_enabled = any(
                v.get("enabled", False) if isinstance(v, dict) else bool(v)
                for v in gc_dict.values()
            )
            if has_any_enabled:
                self.model.enable_gradient_checkpointing(config=gc_dict)
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
        if not self.compile_cfg.get("enabled", False):
            return

        # Patch Qwen3 VL vision attention before compile to avoid
        # FakeTensor / graph-break issues with flash_attn_varlen_func.
        from src.model.vlm.qwen3_vl_compile_patch import apply_patch
        apply_patch()

        compile_cfg = OmegaConf.to_container(self.compile_cfg, resolve=True)
        compile_kwargs = {
            key: value
            for key, value in compile_cfg.items()
            if key != "enabled" and value is not None
        }

        if accelerator.is_main_process:
            print(f"Compiling blocks with kwargs: {compile_kwargs}")

        self.model.compile_blocks(compile_kwargs)
    @staticmethod
    def _build_lr_scheduler(
        optimizer,
        schedule_name: str,
        num_warmup_steps: int,
        num_training_steps: int,
        vlm_group_indices: set[int],
        vlm_freeze_steps: int = 0,
        vlm_rewarmup_steps: int = 0,
    ):
        """Build a per-group LambdaLR scheduler.

        Non-VLM groups follow the standard warmup → cosine/linear decay.
        VLM groups can stay at lr=0 for ``vlm_freeze_steps``, then optionally
        run their own warmup over ``vlm_rewarmup_steps``, and finally follow
        the same cosine/linear decay for the remaining budget.
        """
        from functools import partial
        from torch.optim.lr_scheduler import LambdaLR
        from transformers.optimization import (
            _get_cosine_schedule_with_warmup_lr_lambda,
            _get_linear_schedule_with_warmup_lr_lambda,
        )

        schedule_fn = {
            "cosine": _get_cosine_schedule_with_warmup_lr_lambda,
            "linear": _get_linear_schedule_with_warmup_lr_lambda,
        }
        if schedule_name not in schedule_fn:
            raise ValueError(f"Unsupported lr_scheduler: {schedule_name}")
        lr_lambda_fn = schedule_fn[schedule_name]
        # cosine variant requires num_cycles; linear ignores it via **kwargs
        extra_kwargs = {"num_cycles": 0.5} if schedule_name == "cosine" else {}

        base_lambda = partial(
            lr_lambda_fn,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **extra_kwargs,
        )

        def make_lambda(group_idx: int):
            if group_idx not in vlm_group_indices:
                return base_lambda

            if vlm_freeze_steps <= 0 and vlm_rewarmup_steps <= 0:
                return base_lambda

            if vlm_freeze_steps >= num_training_steps:
                return lambda step: 0.0

            vlm_total = max(1, num_training_steps - vlm_freeze_steps)
            vlm_warmup = min(max(0, vlm_rewarmup_steps), vlm_total)
            vlm_base_lambda = partial(
                lr_lambda_fn,
                num_warmup_steps=vlm_warmup,
                num_training_steps=vlm_total,
                **extra_kwargs,
            )

            def vlm_lambda(step: int) -> float:
                if step < vlm_freeze_steps:
                    return 0.0
                shifted_step = step - vlm_freeze_steps
                return vlm_base_lambda(shifted_step)

            return vlm_lambda

        num_groups = len(optimizer.param_groups)
        lr_lambdas = [make_lambda(i) for i in range(num_groups)]
        return LambdaLR(optimizer, lr_lambdas)

    @staticmethod
    def _unwrap_optimizer(optimizer):
        return getattr(optimizer, "optimizer", optimizer)

    @staticmethod
    def _get_vlm_stage_steps(training_cfg) -> tuple[int, int]:
        freeze_steps = int(training_cfg.get("vlm_freeze_steps", 0))
        rewarmup_steps = int(training_cfg.get("vlm_rewarmup_steps", 0))
        return freeze_steps, rewarmup_steps

    def _is_vlm_freeze_active(self) -> bool:
        return bool(self._vlm_group_indices) and self.update_step < self._vlm_freeze_updates

    def _maybe_reset_vlm_optimizer_state(self, accelerator) -> None:
        if (
            self._vlm_optimizer_state_reset_done
            or not self._vlm_group_indices
            or self._vlm_freeze_updates <= 0
            or self.update_step < self._vlm_freeze_updates
        ):
            return

        optimizer = self._unwrap_optimizer(self.optimizer)
        num_cleared = 0
        for group_idx in self._vlm_group_indices:
            for param in optimizer.param_groups[group_idx]["params"]:
                if optimizer.state.pop(param, None) is not None:
                    num_cleared += 1

        self._vlm_optimizer_state_reset_done = True
        if accelerator.is_main_process:
            print(
                f"Reset optimizer state for {num_cleared} VLM parameters "
                f"at update_step={self.update_step} before VLM re-warmup."
            )

    def _get_param_group_lrs(self) -> tuple[list[float], list[int]]:
        optimizer = self._unwrap_optimizer(self.optimizer)
        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        non_vlm_group_indices = [
            idx for idx in range(len(current_lrs)) if idx not in self._vlm_group_indices
        ]
        return current_lrs, non_vlm_group_indices
        
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

        # Per-rank offset for independent random augmentation across devices
        per_device_seed = run_seed + accelerator.process_index
        torch.manual_seed(per_device_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(per_device_seed)
        np.random.seed(per_device_seed % (2**32 - 1))
        random.seed(per_device_seed)
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

        accelerator = Accelerator(
            log_with='wandb',
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
        if cfg.training.finetune_checkpoint_path:
            state_dict = torch.load(cfg.training.finetune_checkpoint_path, map_location='cpu')
            # handle wrapping
            for key in ['module', 'model', 'model_state_dict']:
                if key in state_dict:
                    state_dict = state_dict[key]
                    break
            model.load_state_dict(state_dict)
            print("Successfully loaded finetuning weights.")
        # Ensure non-backbone modules match the VLM dtype (bf16) for single-GPU
        # training where accelerate autocast may not cover custom modules.
        if cfg.training.use_bf16 and accelerator.distributed_type.value == "NO":
            model.to(dtype=torch.bfloat16)

        runtime_cfg = getattr(cfg, "runtime", None)
        use_lora = bool(getattr(runtime_cfg, "use_lora", getattr(cfg, "lora", False)))
        if use_lora:
            model.freeze_non_lora_weights_in_vlm()

        # Compile after accelerator.prepare() to avoid _orig_mod issues with FSDP2.
        # Moved from here; see post-prepare block below.

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
        self._vlm_group_indices = set()
        if cfg.training.train_vlm:
            vlm_trained_parameters = model.trainable_vlm_parameters
            vlm_trainable_parameters = self.get_grouped_parameters(
                vlm_trained_parameters,
                cfg.optimizer.vlm,
            )
            start_idx = len(all_trainable_parameters)
            all_trainable_parameters.extend(vlm_trainable_parameters)
            self._vlm_group_indices = set(
                range(start_idx, start_idx + len(vlm_trainable_parameters))
            )
        else:
            model.freeze_non_lora_weights_in_vlm()

        if cfg.training.train_depth is False:
            model.freeze_weights_in_depth()

        diffloss_trainable_paramters = self.get_grouped_parameters(
            model.diffloss_parameters,
            cfg.optimizer.diffloss,
        )
        all_trainable_parameters.extend(diffloss_trainable_paramters)

        if model.use_world_model:
            wm_trainable_parameters = self.get_grouped_parameters(
                model.world_model_parameters,
                cfg.optimizer.world_model,
            )
            all_trainable_parameters.extend(wm_trainable_parameters)

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
        self._vlm_freeze_updates, self._vlm_rewarmup_updates = self._get_vlm_stage_steps(cfg.training)
        self._vlm_optimizer_state_reset_done = (
            self._vlm_freeze_updates <= 0 or not self._vlm_group_indices
        )
        # ============================================================
        # WebDataset: dataset and dataloader creation
        # ============================================================
        print("--> Configure WebDataset dataset and dataloader...")
        dataset = hydra.utils.instantiate(cfg.dataset)
        self.use_relative_action = dataset.vla_dataset.use_relative_action
        print("--> dataset instantiated")
        accelerator.wait_for_everyone()

        data_collator = hydra.utils.instantiate(cfg.data_collator)
        dataset.vla_dataset.set_collator(data_collator)
        if dataset.vlm_dataset is not None:
            dataset.vlm_dataset.set_collator(data_collator)

        # Normalizer must be pre-computed for WebDataset
        print("Loading normalizer...")
        assert cfg.training.normalizer_path is not None, (
            "WebDataset training requires a pre-computed normalizer_path.")
        with open(cfg.training.normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
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
        # Validation dataloader — NOT managed by accelerate.
        # Each rank reads its own shards via wds.split_by_node.
        # Unequal batch counts across ranks are safe because
        # eval_with_averaged_model pre-unshards all FSDP params,
        # making every forward pass purely local (no collective).
        val_dataset = dataset.get_validation_dataset()
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
        # Accelerate's prepared scheduler only steps once every num_processes
        # optimizer steps, so multiply total and warmup steps by num_processes
        # to keep the effective schedule correct.
        max_train_steps = max_train_steps * accelerator.num_processes
        num_warmup_steps = cfg.training.lr_warmup_steps * accelerator.num_processes
        vlm_freeze_steps = self._vlm_freeze_updates * accelerator.num_processes
        vlm_rewarmup_steps = self._vlm_rewarmup_updates * accelerator.num_processes
        if accelerator.is_main_process:
            print(f"num_warmup_steps: {num_warmup_steps}, max_train_steps: {max_train_steps}")
            if self._vlm_freeze_updates > 0:
                print(
                    f"VLM staged training: freeze for {self._vlm_freeze_updates} update steps, "
                    f"then re-warmup for {self._vlm_rewarmup_updates} steps"
                )
        self.lr_scheduler = self._build_lr_scheduler(
            optimizer=self.optimizer,
            schedule_name=cfg.training.lr_scheduler,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
            vlm_group_indices=self._vlm_group_indices,
            vlm_freeze_steps=vlm_freeze_steps,
            vlm_rewarmup_steps=vlm_rewarmup_steps,
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
        
        # Resume training from checkpoint after accelerator prepare but BEFORE
        # compile, so that the state_dict keys don't have the _orig_mod prefix
        # that torch.compile introduces.
        if cfg.training.resume_checkpoint_path:
            accelerator.load_state(cfg.training.resume_checkpoint_path)
            self.update_step = self.training_state.update_step
            self.global_step = self.training_state.global_step
            self.epoch = self.training_state.epoch
            if self._vlm_group_indices and self._vlm_freeze_updates > 0:
                self._vlm_optimizer_state_reset_done = self.update_step > self._vlm_freeze_updates
        # Compile after FSDP2 wrapping and checkpoint loading.
        self.maybe_compile_model(accelerator)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_eval_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.eval_every = 1

        profile_context = nullcontext()
        if cfg.training.profile and accelerator.is_main_process:
            profile_context = accelerator.profile()

        # Training loop
        gc_handler = GarbageCollection(gc_freq=1000)
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
                step_perf_end = time.perf_counter()
                for batch_idx, batch in enumerate(dataloader):
                    data_wait_sec = time.perf_counter() - step_perf_end
                    # Enforce steps_per_epoch limit
                    if batch_idx >= steps_per_epoch:
                        break

                    step_perf_start = time.perf_counter()
                    if training_start_time is None:
                        training_start_time = time.time()
                    if cfg.training.profile and torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # Preprocess batch
                    inputs = self.preprocess_batch(batch)

                    if batch_idx == 10 and accelerator.is_main_process and cfg.training.profile:
                        self.tracker.track()
                    step_skipped = False
                    with accelerator.accumulate(self.model):
                        # Forward pass
                        with accelerator.autocast():
                            raw_loss = self.model(self.objective_func, inputs)

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
                        vlm_freeze_active = self._is_vlm_freeze_active()
                        # Per-component gradient clipping to prevent cross-component interference.
                        # clip_grad_norm_ returns the total norm before clipping.
                        part_grad_norms = None
                        if accelerator.sync_gradients and cfg.training.clipping.enabled:
                            max_norm = cfg.training.clipping.max_grad_norm
                            part_params = {
                                "action_expert": self.model.action_expert_parameters,
                                "diffloss": self.model.diffloss_parameters,
                            }
                            if self.model.use_world_model:
                                part_params["world_model"] = self.model.world_model_parameters
                            if cfg.training.train_vlm and not vlm_freeze_active:
                                part_params["vision"] = self.model.trainable_vision_parameters
                                part_params["text"] = self.model.trainable_text_parameters
                            norms = {
                                name: accelerator.clip_grad_norm_(params, max_norm)
                                for name, params in part_params.items()
                            }
                            if should_record:
                                part_grad_norms = norms

                            # NaN/Inf guard: clip_grad_norm_ returns a scalar
                            # norm that DTensor dispatch already reduces across
                            # all FSDP2 shards, so every rank sees the same
                            # value. No extra all_reduce needed — all ranks
                            # will make the same skip/no-skip decision.
                            if any(
                                not math.isfinite(scalar_metric_value(n))
                                for n in norms.values()
                            ):
                                step_skipped = True
                                if accelerator.is_main_process:
                                    print(
                                        f"[WARN] Non-finite grad norm at "
                                        f"update_step={self.update_step} "
                                        f"global_step={self.global_step}: "
                                        f"{({k: scalar_metric_value(v) for k, v in norms.items()})}. "
                                        f"Skipping step."
                                    )

                        if not step_skipped:
                            self._maybe_reset_vlm_optimizer_state(accelerator)
                            self.optimizer.step()
                            self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    if step_skipped:
                        continue
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
                        current_group_lrs, non_vlm_group_indices = self._get_param_group_lrs()
                        current_lr = current_group_lrs[non_vlm_group_indices[0]] if non_vlm_group_indices else current_group_lrs[0]
                        step_log = {
                            'global_step': self.global_step,
                            'update_step': self.update_step,
                            'epoch': self.epoch,
                            'lr': current_lr,
                            'lr_non_vlm': current_lr,
                            'vlm_freeze_active': float(self._is_vlm_freeze_active()),
                        }
                        if self._vlm_group_indices:
                            step_log['lr_vlm'] = current_group_lrs[min(self._vlm_group_indices)]       
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
                            'data_wait_sec': data_wait_sec,
                            'avg_samples_per_sec': total_samples_processed / elapsed_time_sec if elapsed_time_sec > 0 else 0,
                            'samples_per_sec': batch_size_local / step_time_sec if step_time_sec > 0 else 0,
                        })
                        if part_grad_norms is not None:
                            step_log.update({
                                'grad_norm_action_expert': part_grad_norms["action_expert"],
                                'grad_norm_diffloss': part_grad_norms["diffloss"],
                            })
                            if "vision" in part_grad_norms:
                                step_log['grad_norm_vision'] = part_grad_norms["vision"]
                            if "text" in part_grad_norms:
                                step_log['grad_norm_text'] = part_grad_norms["text"]
                            if "world_model" in part_grad_norms:
                                step_log['grad_norm_world_model'] = part_grad_norms["world_model"]
                        with torch.no_grad():
                            if cfg.training.train_vlm:
                                step_log["weight_norm/vision"] = params_l2_norm(
                                    self.model.trainable_vision_parameters
                                )
                                step_log["weight_norm/text"] = params_l2_norm(
                                    self.model.trainable_text_parameters
                                )
                            step_log["weight_norm/action"] = params_l2_norm(
                                self.model.action_expert_parameters
                            )
                            step_log["weight_norm/diffloss"] = params_l2_norm(
                                self.model.diffloss_parameters
                            )
                            if self.model.use_world_model:
                                step_log["weight_norm/world_model"] = params_l2_norm(
                                    self.model.world_model_parameters
                                )
                        step_log.update(raw_loss_cpu)
                        if accelerator.is_main_process:
                            loss_str = " ".join(f"{k}={v:.4f}" for k, v in raw_loss_cpu.items())
                            extra = ""
                            if part_grad_norms is not None and "world_model" in part_grad_norms:
                                extra += f" gn_wm={scalar_metric_value(part_grad_norms['world_model']):.4f}"
                            if "weight_norm/world_model" in step_log:
                                extra += f" wn_wm={scalar_metric_value(step_log['weight_norm/world_model']):.4f}"
                            print(f"[step {self.update_step}] {loss_str}{extra}")

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

                    if self.global_step % 500 == 0 and accelerator.is_main_process:
                        import psutil
                        proc = psutil.Process()
                        children = proc.children(recursive=True)
                        worker_rss = [(c.pid, c.memory_info().rss / 1e9) for c in children]
                        worker_rss.sort(key=lambda x: -x[1])
                        print(f"[Step {self.global_step}] Main RSS: {proc.memory_info().rss/1e9:.2f}GB")
                        for pid, rss in worker_rss[:6]:
                            print(f"  Worker PID {pid}: {rss:.2f}GB")

                    if cfg.training.profile and accelerator.is_main_process:
                        prof.step()

                    gc_handler.run(self.global_step)
                    step_perf_end = time.perf_counter()

                if cfg.training.max_train_steps and self.update_step >= cfg.training.max_train_steps:
                    break
                self.epoch += 1

        gc_handler.finalize()
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

    def preprocess_batch(self, batch):
        input_ids = batch["input_ids"]
        inputs = {
            "input_ids": input_ids,
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"].to(self.dtype)
            if batch["pixel_values"] is not None else None,
            "image_grid_thw": batch["image_grid_thw"],
            "pixel_values_videos": batch["pixel_values_videos"].to(self.dtype)
            if batch["pixel_values_videos"] is not None else None,
            "video_grid_thw": batch["video_grid_thw"],
            "mm_token_type_ids": batch["mm_token_type_ids"],
            "states": batch["states"].to(self.dtype),
            "answer_start_idx": batch["answer_start_idx"],
            "is_vla_data": batch["is_vla_data"],
            "n_states": batch["n_states"],
            "n_actions": batch["n_actions"],
        }
        if self.objective_func != "train_ar":
            inputs["actions"] = batch["actions"].to(self.dtype)
            inputs["actions_valid_mask"] = batch["actions_valid_mask"]
        if self.objective_func != "train_flow":
            inputs["labels"] = batch["labels"]
        # Camera intrinsic as token embedding.
        if "camera_intrinsic" in batch:
            inputs["camera_intrinsic"] = batch["camera_intrinsic"].to(self.dtype)
        # World model future frames (uint8, no dtype cast).
        if "future_frames" in batch:
            inputs["future_frames"] = batch["future_frames"]
            inputs["n_future_frames"] = batch["n_future_frames"]
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
        # Convert OmegaConf ListConfig to plain list so PyTorch's
        # _iterate_state_dict can serialize it during FSDP2 checkpoint save.
        betas = list(cfg.betas)
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': cfg.weight_decay, 'lr': cfg.lr, 'betas': betas},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': cfg.lr, 'betas': betas}
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
