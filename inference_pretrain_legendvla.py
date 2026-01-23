import os
import hydra
import torch
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
from src.utils.metric import (
    get_action_accuracy,
    compute_all_metrics,
    plot_all_visualizations,
    compute_smoothness_metrics,
    compute_error_heatmap,
    compute_covariance_matrix,
    compute_loss_over_time,
    compute_trajectory_metrics,
    compute_per_dimension_metrics,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class LegendVLAInference:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        
        # 设置设备
        self.device = torch.device(cfg.inference.device if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 设置数据类型
        self.dtype = torch.bfloat16 if cfg.training.use_bf16 else torch.float32
        
        # 加载模型配置
        print("正在加载模型配置...")
        if hasattr(cfg.inference, 'model_config_path') and cfg.inference.model_config_path:
            model_config_path = pathlib.Path(cfg.inference.model_config_path)
            if not model_config_path.exists():
                raise FileNotFoundError(f"模型配置文件不存在: {model_config_path}")
            print(f"从配置文件加载模型配置: {model_config_path}")
            model_cfg = OmegaConf.load(model_config_path)
            self.model_cfg = model_cfg
            # 使用模型配置中的 policy 配置来初始化模型
            if 'policy' not in model_cfg:
                raise ValueError(f"模型配置文件中未找到 'policy' 配置: {model_config_path}")
            policy_cfg = model_cfg.policy
        else:
            # 如果没有指定 model_config_path，使用当前配置中的 policy
            print("使用当前配置中的 policy 配置")
            self.model_cfg = None
            policy_cfg = cfg.policy
        
        # 初始化模型
        print("正在初始化模型...")
        self.model: LegendVLA = hydra.utils.instantiate(policy_cfg)
        
        # 加载checkpoint
        if cfg.inference.checkpoint_path:
            print(f"正在加载checkpoint: {cfg.inference.checkpoint_path}")
            self.load_checkpoint(cfg.inference.checkpoint_path)
        
        self.model.to(self.device)
        self.model.eval()
        print("模型初始化完成")
        
    @property
    def output_dir(self):
        return HydraConfig.get().runtime.output_dir
    
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
            print(f"成功加载模型权重")
        else:
            print(f"警告: 未找到模型文件 {model_path}")
    
    def preprocess_batch(self, batch):
        """预处理batch用于推理"""
        input_ids = batch["input_ids"].to(self.device)
        
        # 构建causal mask和position ids
        causal_mask, vlm_position_ids, action_position_ids = (
            self.model.build_causal_mask_and_position_ids(
                batch["attention_mask"].to(self.device), 
                batch["answer_start_idx"].to(self.device), 
                self.dtype
            )
        )
        max_vlm_tokens = input_ids.shape[-1]
        vlm_mask, action_mask = (
            self.model.split_full_mask_into_submasks(causal_mask, max_vlm_tokens)
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
        print("正在初始化数据集...")
        self.dataset = hydra.utils.instantiate(cfg.dataset)
        
        # Get dataset names for separate tracking
        self.dataset_names = []
        if hasattr(cfg, 'vla_dataset_paths'):
            for path in cfg.vla_dataset_paths:
                self.dataset_names.append(path['name'])
        else: 
            raise ValueError("未指定vla_dataset_paths")
        print(f"数据集: {self.dataset_names}")
        
        # 初始化processor
        print("正在初始化processor...")
        if hasattr(self.model_cfg, 'vla_processor'):
            self.vla_processor = hydra.utils.instantiate(self.model_cfg.vla_processor)
        else:
            assert hasattr(cfg, 'vla_processor'), "未指定vla_processor"
            self.vla_processor = hydra.utils.instantiate(cfg.vla_processor)
        self.dataset.set_preprocessor(self.vla_processor)
        
        # 加载normalizer
        print("正在加载normalizer...")

        if hasattr(self.model_cfg, 'normalizer_path'):
            self.normalizer = pickle.load(open(self.model_cfg.normalizer_path, 'rb'))
            self.dataset.set_normalizer(self.normalizer)
            print(f"成功加载normalizer: {self.model_cfg.normalizer_path}")
        elif hasattr(cfg, 'normalizer_path'):
            self.normalizer = pickle.load(open(cfg.normalizer_path, 'rb'))
            self.dataset.set_normalizer(self.normalizer)
            print(f"成功加载normalizer: {cfg.normalizer_path}")
        else:
            print("警告: 未指定normalizer路径，使用默认normalizer")
            self.normalizer = self.dataset.get_normalizer()
            self.dataset.set_normalizer(self.normalizer)
        
        # Select dataset based on configuration
        if hasattr(cfg.inference, 'use_val_dataset') and cfg.inference.use_val_dataset:
            inference_dataset = self.dataset.get_validation_dataset()
            print("使用验证数据集")
        else:
            inference_dataset = self.dataset
            print("使用训练数据集" if cfg.dataset.vla_dataset.val_ratio > 0 else "使用完整数据集（包含训练集和验证集）")
        
        # 创建dataloader
        dataloader = DataLoader(
            inference_dataset, 
            collate_fn=inference_dataset.get_collator(),
            batch_size=cfg.inference.batch_size,
            num_workers=cfg.inference.num_workers,
            shuffle=False,
            pin_memory=True
        )
        
        # 创建输出目录
        output_dir = pathlib.Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {output_dir}")
        
        # 为每个数据集创建单独的 zarr 文件
        dataset_zarr_roots = {}
        zarr_datasets_per_dataset = {}
        
        for dataset_name in self.dataset_names:
            # Create separate zarr file for each dataset
            zarr_path = output_dir / f"inference_results_{dataset_name}.zarr"
            store = zarr.DirectoryStore(str(zarr_path))
            dataset_zarr_roots[dataset_name] = zarr.group(store=store, overwrite=True)
            zarr_datasets_per_dataset[dataset_name] = {}
            print(f"创建zarr文件: {zarr_path}")
        
        # 开始推理
        print("开始推理...")
        # Track statistics per dataset
        actual_batch_count = 0  # Track actual batch count (after skipping)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="推理进度")):
                # 检查是否达到最大推理步数
                if cfg.inference.max_steps and actual_batch_count >= cfg.inference.max_steps:
                    print(f"达到最大推理步数 {cfg.inference.max_steps}，停止推理")
                    break
                
                # 检查是否跳过前N个batch
                if batch_idx < cfg.inference.skip_first:
                    continue
                
                # 获取实际的batch大小
                actual_batch_size = batch["input_ids"].shape[0]
                
                # 计算实际的样本索引
                batch_start_idx = batch_idx * cfg.inference.batch_size
                batch_end_idx = min(batch_start_idx + actual_batch_size, len(inference_dataset))
                sample_indices = np.arange(start=batch_start_idx, stop=batch_end_idx)
                
                # 预处理
                inputs = self.preprocess_batch(batch)
                
                # 推理
                with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    # 1. Flow Matching Inference
                    pred_actions_fm = self.model("infer_action", inputs)
                    
                # --- 处理 Flow Matching 结果 ---
                pred_actions_fm = pred_actions_fm.cpu().float().numpy()
                pred_actions_fm = self.normalizer['actions'].unnormalize(pred_actions_fm)
                
                # 保存结果
                batch_result = {
                    "pred_actions": pred_actions_fm,      # Flow Matching 结果
                }
                
                # 获取每个样本对应的原始数据索引
                origin_frame_indices, dataset_indices = self._get_origin_indices(inference_dataset, sample_indices)
                
                batch_result["sample_indices"] = sample_indices
                batch_result["origin_frame_indices"] = origin_frame_indices
                batch_result["dataset_indices"] = dataset_indices
                
                # 如果有ground truth，也保存并计算误差
                if "actions" in inputs:
                    gt_actions = inputs["actions"].cpu().float().numpy()
                    gt_actions = self.normalizer['actions'].unnormalize(gt_actions)
                    actions_valid_mask = inputs["actions_valid_mask"].cpu().numpy()
                    
                    batch_result["gt_actions"] = gt_actions
                    batch_result["actions_valid_mask"] = actions_valid_mask
                
                actual_batch_count += 1
                
                # Split batch by dataset and write to corresponding zarr files
                for dataset_idx in range(len(self.dataset_names)):
                    dataset_name = self.dataset_names[dataset_idx]
                    mask = dataset_indices == dataset_idx
                    
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
        # 保存 inference config 文件：
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
            self.compute_and_save_metrics(zarr_path, dataset_output_dir, dataset_name)
        
        print("\n统计信息和可视化计算完成！")
    
    def compute_and_save_metrics(self, zarr_path, output_dir, dataset_name):
        """
        从 zarr 文件读取数据，计算所有统计指标并生成可视化
        
        Args:
            zarr_path: zarr 文件路径
            output_dir: 输出目录
            dataset_name: 数据集名称
        """
        # 读取 zarr 文件
        store = zarr.DirectoryStore(str(zarr_path))
        zarr_root = zarr.group(store=store)
        
        # 检查是否有必要的数据
        if 'pred_actions' not in zarr_root:
            print(f"  警告: {dataset_name} 中没有 pred_actions，跳过")
            return
        
        pred_actions = zarr_root['pred_actions'][:]  # [N, H, D]
        
        # 保存基本信息
        info_dict = {
            'dataset_name': str(dataset_name),
            'num_samples': int(pred_actions.shape[0]),
            'horizon': int(pred_actions.shape[1]),
            'action_dim': int(pred_actions.shape[2]),
            'has_gt': 'gt_actions' in zarr_root,
        }
        info_path = output_dir / "info.json"
        with open(info_path, 'w') as f:
            json.dump(info_dict, f, indent=2)
        
        # 如果有 ground truth，计算统计信息和可视化
        if 'gt_actions' not in zarr_root:
            print(f"  注意: {dataset_name} 中没有 ground truth 数据，跳过统计计算")
            return
        
        gt_actions = zarr_root['gt_actions'][:]  # [N, H, D]
        
        # 转换为 torch tensor
        pred_tensor = torch.from_numpy(pred_actions).float()
        gt_tensor = torch.from_numpy(gt_actions).float()
        
        # 计算所有指标
        print(f"  计算统计指标...")
        
        # 1. 基本指标 (compute_all_metrics 包含的部分)
        basic_metrics = compute_all_metrics(gt_tensor, pred_tensor)
        
        # 2. 动作准确度
        accuracy_thresholds = [0.1, 0.2, 0.3, 0.5]
        action_accuracy = get_action_accuracy(gt_tensor, pred_tensor, thresholds=accuracy_thresholds)
        
        # 3. 平滑度指标 (详细版本)
        smoothness = compute_smoothness_metrics(gt_tensor, pred_tensor)
        
        # 4. 误差热图
        error_heatmap = compute_error_heatmap(gt_tensor, pred_tensor)
        
        # 5. 协方差矩阵
        covariance = compute_covariance_matrix(gt_tensor, pred_tensor)
        
        # 6. 每个时间步的损失
        loss_l1_over_time = compute_loss_over_time(gt_tensor, pred_tensor, loss_type='l1')
        loss_l2_over_time = compute_loss_over_time(gt_tensor, pred_tensor, loss_type='l2')
        
        # 7. 轨迹级别指标 (已在 basic_metrics 中，但保留详细版本)
        trajectory_metrics = compute_trajectory_metrics(gt_tensor, pred_tensor)
        
        # 8. 每个维度指标 (已在 basic_metrics 中，但保留详细版本)
        per_dim_metrics = compute_per_dimension_metrics(gt_tensor, pred_tensor)
        
        # 汇总所有指标到字典
        metrics_dict = {
            # 基本指标
            'overall_mae': float(basic_metrics['overall_mae'].item()),
            'first_diff_error': float(basic_metrics['first_diff_error'].item()),
            'second_diff_error': float(basic_metrics['second_diff_error'].item()),
            
            # 动作准确度
            'action_accuracy': {
                f'threshold_{t}': float(action_accuracy[i].item())
                for i, t in enumerate(accuracy_thresholds)
            },
            
            # 平滑度详细指标
            'smoothness': {
                'first_diff_error': float(smoothness['first_diff_error'].item()),
                'second_diff_error': float(smoothness['second_diff_error'].item()),
                'first_diff_error_heatmap_mean': float(torch.mean(smoothness['first_diff_error_heatmap']).item()),
                'second_diff_error_heatmap_mean': float(torch.mean(smoothness['second_diff_error_heatmap']).item()),
            },
            
            # 误差热图统计
            'error_heatmap': {
                'mean': float(torch.mean(error_heatmap).item()),
                'std': float(torch.std(error_heatmap).item()),
                'max': float(torch.max(error_heatmap).item()),
                'min': float(torch.min(error_heatmap).item()),
            },
            
            # 协方差矩阵统计
            'covariance': {
                'gt_cov_trace': float(torch.trace(covariance['gt_cov']).item()),
                'pred_cov_trace': float(torch.trace(covariance['pred_cov']).item()),
                'error_cov_trace': float(torch.trace(covariance['error_cov']).item()),
                'gt_cov_det': float(torch.det(covariance['gt_cov']).item()),
                'pred_cov_det': float(torch.det(covariance['pred_cov']).item()),
                'error_cov_det': float(torch.det(covariance['error_cov']).item()),
            },
            
            # 每个时间步的损失
            'loss_over_time': {
                'l1_mean': float(torch.mean(loss_l1_over_time).item()),
                'l1_std': float(torch.std(loss_l1_over_time).item()),
                'l1_max': float(torch.max(loss_l1_over_time).item()),
                'l1_min': float(torch.min(loss_l1_over_time).item()),
                'l2_mean': float(torch.mean(loss_l2_over_time).item()),
                'l2_std': float(torch.std(loss_l2_over_time).item()),
                'l2_max': float(torch.max(loss_l2_over_time).item()),
                'l2_min': float(torch.min(loss_l2_over_time).item()),
            },
            
            # 轨迹级别指标
            'trajectory': {
                'endpoint_error_mean': float(torch.mean(trajectory_metrics['endpoint_error']).item()),
                'endpoint_error_std': float(torch.std(trajectory_metrics['endpoint_error']).item()),
                'trajectory_length_error_mean': float(torch.mean(trajectory_metrics['trajectory_length_error']).item()),
                'trajectory_length_error_std': float(torch.std(trajectory_metrics['trajectory_length_error']).item()),
                'mean_error_mean': float(torch.mean(trajectory_metrics['mean_error']).item()),
                'mean_error_std': float(torch.std(trajectory_metrics['mean_error']).item()),
                'max_error_mean': float(torch.mean(trajectory_metrics['max_error']).item()),
                'max_error_std': float(torch.std(trajectory_metrics['max_error']).item()),
            },
            
            # 每个维度指标
            'per_dimension': {
                'mae_per_dim': per_dim_metrics['mae_per_dim'].cpu().numpy().tolist(),
                'mae_per_dim_mean': float(torch.mean(per_dim_metrics['mae_per_dim']).item()),
                'mae_per_dim_std': float(torch.std(per_dim_metrics['mae_per_dim']).item()),
                'mae_per_dim_max': float(torch.max(per_dim_metrics['mae_per_dim']).item()),
                'mae_per_dim_min': float(torch.min(per_dim_metrics['mae_per_dim']).item()),
            },
        }
        
        # 保存指标到 JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"  指标已保存至: {metrics_path}")
        
        # 打印关键指标
        print(f"    总体 MAE: {metrics_dict['overall_mae']:.4f}")
        print(f"    一阶差分误差: {metrics_dict['first_diff_error']:.4f}")
        print(f"    二阶差分误差: {metrics_dict['second_diff_error']:.4f}")
        print(f"    平均端点误差: {metrics_dict['trajectory']['endpoint_error_mean']:.4f}")
        print(f"    动作准确度 (threshold=0.1): {metrics_dict['action_accuracy']['threshold_0.1']:.4f}")
        print(f"    动作准确度 (threshold=0.2): {metrics_dict['action_accuracy']['threshold_0.2']:.4f}")
        
        # 生成所有可视化
        print(f"  生成可视化...")
        plot_all_visualizations(
            gt_tensor,
            pred_tensor,
            str(output_dir),
            prefix='',
        )
        print(f"  可视化已保存至: {output_dir}")
        
    
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
    config_path=str(pathlib.Path(__file__).parent.joinpath("src/config")),
    config_name="experiment/inference_single"
)
def main(cfg):
    inference = LegendVLAInference(cfg)
    inference.run()


if __name__ == "__main__":
    main()