import os
import hydra
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import numpy as np
import pickle
from tqdm import tqdm
import json
from hydra.core.hydra_config import HydraConfig
import zarr

from src.policy.legendvla import LegendVLA
from src.utils.pytorch_util import dict_apply
from src.utils.metric import compute_and_save_metrics
from src.utils.plotting import plot_multilayer_attention_maps

OmegaConf.register_new_resolver("eval", eval, replace=True)


class LegendVLAInference:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        
        # # 使用 torchdynamo 编译模型（在 prepare 之前）
        # if hasattr(cfg.inference.compile, 'enabled') and cfg.inference.compile.enabled:
        #     dynamo_plugin = TorchDynamoPlugin(
        #         backend=cfg.inference.compile.backend,  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
        #         mode=cfg.inference.compile.mode,      # Options: "default", "reduce-overhead", "max-autotune"
        #         fullgraph=cfg.inference.compile.fullgraph,
        #         dynamic=cfg.inference.compile.dynamic
        #     )
        # else: 
        #     dynamo_plugin = None

        # 初始化 Accelerator
        self.accelerator = Accelerator()
        
        # 设置设备
        self.device = self.accelerator.device
        print(f"使用设备: {self.device}")
        
        # 设置数据类型
        self.dtype = torch.bfloat16 if cfg.inference.use_mixed_precision else torch.float32
        
        # 加载模型配置
        if self.is_main_process:
            print("正在加载模型配置...")
        if hasattr(cfg.inference, 'model_config_path') and cfg.inference.model_config_path:
            model_config_path = pathlib.Path(cfg.inference.model_config_path)
            if not model_config_path.exists():
                raise FileNotFoundError(f"模型配置文件不存在: {model_config_path}")
            if self.is_main_process:
                print(f"从配置文件加载模型配置: {model_config_path}")
            model_cfg = OmegaConf.load(model_config_path)
            self.model_cfg = model_cfg
            # 使用模型配置中的 policy 配置来初始化模型
            if 'policy' not in model_cfg:
                raise ValueError(f"模型配置文件中未找到 'policy' 配置: {model_config_path}")
            policy_cfg = model_cfg.policy
        else:
            # 如果没有指定 model_config_path，使用当前配置中的 policy
            if self.is_main_process:
                print("使用当前配置中的 policy 配置")
            self.model_cfg = None
            policy_cfg = cfg.policy
        
        # 初始化模型
        if self.is_main_process:
            print("正在初始化模型...")
        self.model: LegendVLA = hydra.utils.instantiate(policy_cfg)
        
        # 加载checkpoint
        if cfg.inference.checkpoint_path:
            if self.is_main_process:
                print(f"正在加载checkpoint: {cfg.inference.checkpoint_path}")
            self.load_checkpoint(cfg.inference.checkpoint_path)
        elif self.model_cfg is not None and self.model_cfg.training.pretrained_pi05_model_path is not None:
            self.model.load_pretrained_pi05_weights()
        elif self.model_cfg is not None and self.model_cfg.training.pretrained_vlm_model_path is not None:
            self.model.load_pretrained_vlm_weights()
        
        if hasattr(cfg.inference.compile, 'enabled') and cfg.inference.compile.enabled:
            self.model = torch.compile(
                self.model,
                mode=cfg.inference.compile.mode,
                dynamic=cfg.inference.compile.dynamic,
                fullgraph=cfg.inference.compile.fullgraph,
                backend=cfg.inference.compile.backend
            )

        self.model.eval()
        
        # Get inference mode: "ar" (autoregressive) or "flow" (flow matching)
        self.mode = cfg.inference.get("mode", "flow")
        if self.mode not in ["ar", "flow"]:
            raise ValueError(f"Invalid inference mode: {self.mode}. Must be 'ar' or 'flow'")
        
        if self.is_main_process:
            print(f"推理模式: {self.mode}")
            print("模型初始化完成")
    
    @property
    def output_dir(self):
        return HydraConfig.get().runtime.output_dir
    
    @property
    def is_main_process(self):
        return self.accelerator.is_main_process
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        checkpoint_path = pathlib.Path(checkpoint_path)
        
        # 加载模型状态
        model_path = checkpoint_path
        if model_path.exists():
            state_dict = torch.load(model_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'module' in state_dict:
                state_dict = state_dict['module']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            self.model.load_state_dict(state_dict)
            if self.is_main_process:
                print(f"成功加载模型权重")
        else:
            if self.is_main_process:
                print(f"警告: 未找到模型文件 {model_path}")
    
    def preprocess_batch(self, batch):
        """预处理batch用于推理"""
        input_ids = batch["input_ids"].to(self.device)
        
        # Get unwrapped model for mask building
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
        # 构建causal mask和position ids
        causal_mask, vlm_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(
                batch["attention_mask"].to(self.device), 
                batch["answer_start_idx"].to(self.device), 
                self.dtype
            )
        )
        max_vlm_tokens = input_ids.shape[-1]
        vlm_mask, action_mask = (
            model.split_full_mask_into_submasks(causal_mask, max_vlm_tokens)
        )
        
        inputs = {
            "input_ids": input_ids,
            "pixel_values": batch["pixel_values"].to(self.device).to(self.dtype),
            "vlm_position_ids": vlm_position_ids,
            "action_position_ids": action_position_ids,
            "vlm_mask": vlm_mask,
            "action_mask": action_mask,
            "causal_mask": causal_mask,
        }
        
        if "depth_values" in batch:
            inputs["depth_values"] = batch["depth_values"].to(self.device).to(self.dtype)
            inputs["depth_ids"] = batch["depth_ids"].to(self.device)

        # 添加ground truth actions用于对比（如果有）
        if "actions" in batch:
            inputs["actions"] = batch["actions"].to(self.device).to(self.dtype)
            inputs["actions_valid_mask"] = batch["actions_valid_mask"].to(self.device)
        
        return inputs
    
    def preprocess_batch_ar(self, batch):
        """Preprocess batch for autoregressive generation"""
        input_ids = batch["input_ids"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device).to(self.dtype)
        attention_mask = batch.get("attention_mask", None)
        
        inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask.to(self.device)
        
        if "depth_values" in batch:
            inputs["depth_values"] = batch["depth_values"].to(self.device).to(self.dtype)
            inputs["depth_ids"] = batch["depth_ids"].to(self.device)
        
        # Add ground truth actions for comparison (if available)
        if "actions" in batch:
            inputs["actions"] = batch["actions"].to(self.device).to(self.dtype)
            inputs["actions_valid_mask"] = batch["actions_valid_mask"].to(self.device)
        
        return inputs
    
    def _preprocess_for_autoregressive(self, inputs):
        """
        Preprocess inputs for autoregressive generation.
        Extract generation parameters from config.
        
        Args:
            inputs (dict): Preprocessed inputs from preprocess_batch_ar
            
        Returns:
            dict: Generation parameters including max_new_tokens, temperature, etc.
        """
        generation_params = self.cfg.inference.autoregressive_config
        generation_params = OmegaConf.to_container(generation_params, resolve=True)
        if self.vla_processor is not None: 
            generation_params['eos_token_id'] = self.vla_processor.eos_token_id
            generation_params['allowed_token_ids'] = self.vla_processor.total_motion_token_list
        return generation_params
    
    def _postprocess_autoregressive_results(self, generated_ids):
        """
        Postprocess autoregressive generation results.
        Decode generated token IDs to actions using processor.
        
        Args:
            generated_ids (torch.LongTensor): [B, prefill_len + num_generated] Generated token IDs
            
        Returns:
            torch.FloatTensor: [B, T_action_original, D] Decoded actions (normalized)
        """
        generated_ids_np = generated_ids.cpu().numpy()  # [B, seq_len]
        batch_size = generated_ids_np.shape[0]
        pred_actions_list = []
        
        for i in range(batch_size):
            output_ids = generated_ids_np[i]  # [seq_len]
            print(f"output_ids: {output_ids}")
            
            # Decode using processor
            decoded_result = self.vla_processor.decode(
                output_ids=output_ids,
                T_state_original=self.T_state_original,
                T_action_original=self.T_action_original,
            )
            
            if decoded_result and 'actions' in decoded_result:
                pred_actions_list.append(decoded_result['actions'])  # [T_action_original, D]
            else:
                # If decoding failed, use zeros as fallback
                if self.T_action_original is not None:
                    action_dim = 48  # Default action dimension
                    pred_actions_list.append(
                        np.zeros((self.T_action_original, action_dim), dtype=np.float32)
                    )
        
        # Stack decoded actions
        if len(pred_actions_list) > 0:
            pred_actions_np = np.stack(pred_actions_list, axis=0)  # [B, T_action_original, D]
            pred_actions = torch.from_numpy(pred_actions_np).to(self.device)
        else:
            pred_actions = None
        
        return pred_actions
    
    def _get_origin_indices(self, dataset, sample_indices):
        """Get origin indices and dataset indices for each sample"""
        origin_indices = []
        dataset_indices = []
        history = dataset.history if hasattr(dataset, 'history') else 0
        
        # Extract origin indices directly from sampler without needing raw samples
        for sample_idx in sample_indices:
            # Find which dataset and local_idx this sample belongs to
            curr_idx = int(sample_idx)
            dataset_idx = None
            local_idx = None
            
            for i, length in enumerate(dataset.sampler_lens):
                if curr_idx < length:
                    local_idx = curr_idx
                    dataset_idx = i
                    break
                curr_idx -= length
            
            if dataset_idx is None or local_idx is None:
                raise ValueError(f"Index {sample_idx} is out of range")
            
            # Get buffer_start_idx and sample_start_idx from sampler indices
            sampler = dataset.samplers[dataset_idx]
            buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = sampler.indices[local_idx]

            if history < sample_start_idx:
                # sample['image'][history] is padding, actual frame is at buffer_start_idx
                observation_frame_idx = buffer_start_idx
            else:
                # sample['image'][history] is real data at buffer_start_idx + history - sample_start_idx
                observation_frame_idx = buffer_start_idx + history - sample_start_idx
            
            # Save the actual observation frame index, not buffer_start_idx
            origin_indices.append(observation_frame_idx)
            dataset_indices.append(dataset_idx)
        
        origin_indices = np.array(origin_indices)
        dataset_indices = np.array(dataset_indices)
        return origin_indices, dataset_indices
    
    def update_results(self, results, batch_result): 
        for key in batch_result:
            if key not in results:
                results[key] = []
            results[key].append(batch_result[key])

    def run(self):
        cfg = self.cfg
        
        # 初始化数据集
        if self.is_main_process:
            print("正在初始化数据集...")
        if self.model_cfg is not None and self.model_cfg.dataset is not None: 
            vla_dataset_cfg = self.model_cfg.dataset.vla_dataset
            vla_dataset_cfg.zarr_paths = cfg.vla_dataset_paths
            vla_dataset_cfg.mode = f'infer'
            self.dataset = hydra.utils.instantiate(vla_dataset_cfg)
        else:
            self.dataset = hydra.utils.instantiate(cfg.dataset)
        
        # Get dataset names for separate tracking
        self.dataset_names = []
        if hasattr(cfg, 'vla_dataset_paths'):
            for path in cfg.vla_dataset_paths:
                self.dataset_names.append(path['name'])
        else: 
            raise ValueError("未指定vla_dataset_paths")
        if self.is_main_process:
            print(f"数据集: {self.dataset_names}")
        
        # Initialize processor
        if self.is_main_process:
            print("正在初始化processor...")
        if self.model_cfg is not None and self.model_cfg.vla_processor is not None:
            self.vla_processor = hydra.utils.instantiate(self.model_cfg.vla_processor)
        else:
            assert hasattr(cfg, 'vla_processor'), "未指定vla_processor"
            self.vla_processor = hydra.utils.instantiate(cfg.vla_processor)
        self.dataset.set_preprocessor(self.vla_processor)
        
        # Get T_state_original and T_action_original from config
        T_state_original = self.model_cfg['n_obs_state_steps']
        T_action_original = self.model_cfg['n_action_steps']
        
        # Store for later use
        self.T_state_original = T_state_original
        self.T_action_original = T_action_original
        
        if self.is_main_process:
            print(f"T_state_original: {T_state_original}, T_action_original: {T_action_original}")
        
        # 加载normalizer
        if self.is_main_process:
            print("正在加载normalizer...")

        if self.model_cfg is not None and self.model_cfg.training.normalizer_path is not None:
            self.normalizer = pickle.load(open(self.model_cfg.training.normalizer_path, 'rb'))
            self.dataset.set_normalizer(self.normalizer)
            if self.is_main_process:
                print(f"成功加载normalizer: {self.model_cfg.training.normalizer_path}")
        elif hasattr(cfg, 'normalizer_path'):
            self.normalizer = pickle.load(open(cfg.normalizer_path, 'rb'))
            self.dataset.set_normalizer(self.normalizer)
            if self.is_main_process:
                print(f"成功加载normalizer: {cfg.normalizer_path}")
        else:
            if self.is_main_process:
                print("警告: 未指定normalizer路径，使用默认normalizer")
            self.normalizer = self.dataset.get_normalizer()
            self.dataset.set_normalizer(self.normalizer)
        
        # Select dataset based on configuration
        if hasattr(cfg.inference, 'use_val_dataset') and cfg.inference.use_val_dataset:
            inference_dataset = self.dataset.get_validation_dataset()
            if self.is_main_process:
                print("使用验证数据集")
        else:
            inference_dataset = self.dataset
            if self.is_main_process:
                print("使用完整数据集（包含训练集和验证集）")
        
        # 创建dataloader
        dataloader = DataLoader(
            inference_dataset, 
            collate_fn=inference_dataset.get_collator(),
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=False,
            pin_memory=True
        )
        
        # 使用 Accelerator 准备模型和 dataloader
        self.model, dataloader = self.accelerator.prepare(self.model, dataloader)
        
        # 创建输出目录（只在主进程创建）
        output_dir = pathlib.Path(self.output_dir)
        if self.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"输出目录: {output_dir}")
        # 同步所有进程，确保目录已创建
        self.accelerator.wait_for_everyone()
        
        # 为每个数据集创建单独的 zarr 文件（只在主进程创建）
        dataset_zarr_roots = {}
        zarr_datasets_per_dataset = {}
        
        if self.is_main_process:
            for dataset_name in self.dataset_names:
                # Create separate zarr file for each dataset
                zarr_path = output_dir / f"inference_results_{dataset_name}.zarr"
                store = zarr.DirectoryStore(str(zarr_path))
                dataset_zarr_roots[dataset_name] = zarr.group(store=store, overwrite=True)
                zarr_datasets_per_dataset[dataset_name] = {}
                print(f"创建zarr文件: {zarr_path}")
        
        # 开始推理
        if self.is_main_process:
            print("开始推理...")
        # Track statistics per dataset
        actual_batch_count = 0  # Track actual batch count (after skipping)
        
        # Track min/max loss samples for attention visualization
        min_loss_sample = {'loss': float('inf'), 'attn_weights': None, 'metadata': None}
        max_loss_sample = {'loss': float('-inf'), 'attn_weights': None, 'metadata': None}
        
        with torch.no_grad():
            dataloader_iter = tqdm(dataloader, desc="推理进度") if self.is_main_process else dataloader
            for batch_idx, batch in enumerate(dataloader_iter):
                # 检查是否达到最大推理步数
                if cfg.inference.max_steps and actual_batch_count >= cfg.inference.max_steps:
                    if self.is_main_process:
                        print(f"达到最大推理步数 {cfg.inference.max_steps}，停止推理")
                    break
                
                # 检查是否跳过前N个batch
                if batch_idx < cfg.inference.skip_first:
                    continue
                
                # 获取实际的batch大小
                actual_batch_size = batch["input_ids"].shape[0]
                
                # 计算样本索引（每个进程的索引是相对于该进程的数据）
                # 由于使用 DistributedSampler，每个进程处理不同的数据子集
                # 我们会在收集结果后重新计算全局索引
                batch_start_idx = batch_idx * cfg.dataloader.batch_size
                batch_end_idx = batch_start_idx + actual_batch_size
                sample_indices = np.arange(start=batch_start_idx, stop=batch_end_idx)
                
                # Preprocess based on mode
                if self.mode == "flow":
                    inputs = self.preprocess_batch(batch)
                else:  # self.mode == "ar"
                    inputs = self.preprocess_batch_ar(batch)
                
                # Inference (different for each mode)
                with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    if self.mode == "flow":
                        # Flow Matching Inference
                        pred_actions = self.model("infer_action", inputs)
                    else:  # self.mode == "ar"
                        # Autoregressive generation
                        # 1. Preprocess: get generation parameters
                        generation_params = self._preprocess_for_autoregressive(inputs)
                        
                        # 3. Model inference
                        generation_output = self.model("infer_autoregressive", inputs, **generation_params)

                        pred_actions = self._postprocess_autoregressive_results(generation_output['generated_ids'])
                
                pred_actions = self.normalizer['actions'].unnormalize(pred_actions)
                # 保存结果
                batch_result = {
                    "pred_actions": pred_actions,
                }
                
                # 获取每个样本对应的原始数据索引
                origin_frame_indices, dataset_indices = self._get_origin_indices(inference_dataset, sample_indices)
                
                batch_result["sample_indices"] = torch.from_numpy(sample_indices).to(self.device)
                batch_result["origin_frame_indices"] = torch.from_numpy(origin_frame_indices).to(self.device)
                batch_result["dataset_indices"] = torch.from_numpy(dataset_indices).to(self.device)
                
                # 如果有ground truth，也保存并计算误差
                if "actions" in inputs:
                    gt_actions = inputs["actions"]
                    gt_actions = self.normalizer['actions'].unnormalize(gt_actions)
                    actions_valid_mask = inputs["actions_valid_mask"]
                    
                    batch_result["gt_actions"] = gt_actions
                    batch_result["actions_valid_mask"] = actions_valid_mask
                    
                    # Track min/max loss samples for attention visualization
                    if hasattr(self.model, 'module'):
                        model = self.model.module
                    else:
                        model = self.model
                    
                    if hasattr(model, 'attn_weights') and len(model.attn_weights) > 0:
                        # Compute per-sample L1 loss
                        abs_diff = torch.abs(pred_actions - gt_actions)
                        per_sample_l1_loss = torch.sum(
                            abs_diff.flatten(start_dim=1), dim=1
                        ) / torch.sum(actions_valid_mask.flatten(start_dim=1), dim=1)  # [B]
                        
                        # attn_weights: list of [B, num_heads, seq_len, seq_len] for each layer
                        attn_weights = torch.stack(model.attn_weights, dim=0)  # [num_layers, B, num_heads, seq_len, seq_len]
                        
                        # Get token segments if available
                        token_segments = None
                        if hasattr(inputs, 'token_segments'):
                            token_segments = inputs['token_segments']
                        
                        # Find min/max loss samples in this batch
                        for sample_idx in range(per_sample_l1_loss.shape[0]):
                            current_loss = per_sample_l1_loss[sample_idx].item()
                            metadata = {
                                'batch_idx': batch_idx,
                                'sample_idx': sample_idx,
                                'l1_loss': current_loss,
                                'token_segments': token_segments,
                            }
                            
                            # Update min loss sample
                            if current_loss < min_loss_sample['loss']:
                                min_loss_sample['loss'] = current_loss
                                min_loss_sample['attn_weights'] = attn_weights[:, sample_idx, :, :, :].float().cpu()
                                min_loss_sample['metadata'] = metadata
                            
                            # Update max loss sample
                            if current_loss > max_loss_sample['loss']:
                                max_loss_sample['loss'] = current_loss
                                max_loss_sample['attn_weights'] = attn_weights[:, sample_idx, :, :, :].float().cpu()
                                max_loss_sample['metadata'] = metadata
                
                actual_batch_count += 1
                
                # Gather results from all processes to main process
                gathered_batch_results = self.accelerator.gather(batch_result)
                gathered_batch_results_np = dict_apply(
                    gathered_batch_results, lambda x: x.cpu().float().numpy()
                )
                total_samples = gathered_batch_results_np["pred_actions"].shape[0]
                
                # Recalculate global sample_indices (after merging)
                if "sample_indices" in gathered_batch_results_np:
                    # Recalculate global indices using merged data
                    gathered_batch_results_np["sample_indices"] = np.arange(total_samples)
                
                batch_result = gathered_batch_results_np
                
                # Split batch by dataset and write to corresponding zarr files (只在主进程)
                if self.is_main_process:
                    for dataset_idx in range(len(self.dataset_names)):
                        dataset_name = self.dataset_names[dataset_idx]
                        mask = batch_result["dataset_indices"] == dataset_idx
                        
                        if np.any(mask):
                            # Extract samples for this dataset
                            dataset_batch_result = {}
                            for key, value in batch_result.items():
                                if isinstance(value, np.ndarray):
                                    dataset_batch_result[key] = value[mask]
                                # Skip non-array fields like sample_indices
                            
                            # Write to corresponding zarr file
                            self.append_batch_to_zarr(
                                dataset_zarr_roots[dataset_name],
                                zarr_datasets_per_dataset[dataset_name],
                                dataset_batch_result
                            )
                
                # 同步所有进程
                self.accelerator.wait_for_everyone()
        
        # 同步所有进程
        self.accelerator.wait_for_everyone()
        
        # Visualize attention weights for selected samples (只在主进程)
        if self.is_main_process:
            self.visualize_attention_samples(min_loss_sample, max_loss_sample, output_dir)
        
        # 保存 inference config 文件（只在主进程）
        if self.is_main_process:
            inference_config_path = output_dir / "inference_config.yaml"
            with open(inference_config_path, 'w') as f:
                OmegaConf.save(cfg, f)
            print(f"推理配置已保存至: {inference_config_path}")

            # 打印并保存每个数据集的统计信息
            print("\n开始计算统计信息和生成可视化...")
            for dataset_name in self.dataset_names:
                zarr_path = output_dir / f"inference_results_{dataset_name}.zarr"
                if not zarr_path.exists():
                    print(f"警告: 未找到 zarr 文件 {zarr_path}，跳过")
                    continue
                
                print(f"\n处理数据集: {dataset_name}")
                # 创建数据集特定的输出目录
                dataset_output_dir = output_dir / dataset_name
                dataset_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 计算统计信息和生成可视化
                compute_and_save_metrics(zarr_path, dataset_output_dir, dataset_name)
            
            print("\n统计信息和可视化计算完成！")
        
        self.accelerator.end_training()
        
    
    def visualize_attention_samples(self, min_loss_sample, max_loss_sample, output_dir):
        """
        Visualize attention maps for min/max loss samples.
        
        Args:
            min_loss_sample: Dict containing lowest loss sample's attention weights and metadata
            max_loss_sample: Dict containing highest loss sample's attention weights and metadata
            output_dir: Base output directory for saving visualizations
        """
        if min_loss_sample['attn_weights'] is None:
            print("No attention weights to visualize")
            return
        
        print(f"\nVisualizing attention weights for selected samples...")
        
        selected_samples = {
            'lowest_loss': {
                'attn_weights': min_loss_sample['attn_weights'],
                'metadata': min_loss_sample['metadata']
            },
            'highest_loss': {
                'attn_weights': max_loss_sample['attn_weights'],
                'metadata': max_loss_sample['metadata']
            }
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
            
            attn_weights_sample = sample_data['attn_weights'].numpy()  # [num_layers, num_heads, seq_len, seq_len]
            
            # Create output directory
            attention_output_dir = pathlib.Path(output_dir) / 'attention_visualization' / name
            
            token_segments = sample_data['metadata'].get('token_segments', None)
            
            # Visualize attention maps
            try:
                plot_multilayer_attention_maps(
                    attention_maps=attn_weights_sample,
                    output_dir=str(attention_output_dir),
                    token_segments=token_segments
                )
                print(f"  Saved to: {attention_output_dir}")
            except Exception as e:
                print(f"  Error visualizing attention: {e}")
    
    def append_batch_to_zarr(self, root, zarr_datasets, batch_result):
        """Append a batch of results to zarr file incrementally"""
        # Keys to save
        keys_to_save = ['pred_actions', 'gt_actions', 'actions_valid_mask', 'origin_frame_indices']
        
        for key in keys_to_save:
            if key not in batch_result:
                continue
            
            data = batch_result[key]
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            if key not in zarr_datasets:
                # First time: create dataset with maxshape to allow appending
                zarr_datasets[key] = root.create_dataset(
                    key,
                    shape=(0,) + data.shape[1:],
                    maxshape=(None,) + data.shape[1:],
                    dtype=data.dtype,
                    chunks=True,
                    compression='gzip',
                    compression_opts=1
                )
            
            # Directly append using zarr's append method
            dataset = zarr_datasets[key]
            dataset.append(data, axis=0)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("src/config/experiment")),
    config_name="inference_pretrain_legendvla"
)
def main(cfg):
    inference = LegendVLAInference(cfg)
    inference.run()


if __name__ == "__main__":
    main()