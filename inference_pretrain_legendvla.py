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

OmegaConf.register_new_resolver("eval", eval, replace=True)


def convert_to_serializable(obj):
    """Convert numpy types and other non-serializable types to native Python types"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


class LegendVLAInference:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        
        # 设置设备
        self.device = torch.device(cfg.inference.device if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 设置数据类型
        self.dtype = torch.bfloat16 if cfg.training.use_bf16 else torch.float32
        
        # 初始化模型
        print("正在初始化模型...")
        self.model: LegendVLA = hydra.utils.instantiate(cfg.policy)
        
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
        self.dataset = hydra.utils.instantiate(cfg.dataset.vla_dataset)
        
        # Get dataset names for separate tracking
        self.dataset_names = []
        if hasattr(cfg, 'vla_dataset_paths'):
            for path in cfg.vla_dataset_paths:
                # Extract dataset name from path (e.g., "oakink2_test_seen" from ".../.../oakink2_test_seen.zarr")
                # Ensure it's a string for JSON serialization
                dataset_name = str(pathlib.Path(str(path)).stem)
                self.dataset_names.append(dataset_name)
        print(f"数据集: {self.dataset_names}")
        
        # 初始化processor
        print("正在初始化processor...")
        self.vla_processor = hydra.utils.instantiate(cfg.vla_processor)
        self.dataset.set_preprocessor(self.vla_processor)
        
        # 加载normalizer
        print("正在加载normalizer...")
        if cfg.training.normalizer_path is not None:
            self.normalizer = pickle.load(open(cfg.training.normalizer_path, 'rb'))
            self.dataset.set_normalizer(self.normalizer)
            print(f"成功加载normalizer: {cfg.training.normalizer_path}")
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
        current_write_pos_per_dataset = {}
        
        for dataset_name in self.dataset_names:
            # Create separate zarr file for each dataset
            zarr_path = output_dir / f"inference_results_{dataset_name}.zarr"
            store = zarr.DirectoryStore(str(zarr_path))
            dataset_zarr_roots[dataset_name] = zarr.group(store=store, overwrite=True)
            zarr_datasets_per_dataset[dataset_name] = {}
            current_write_pos_per_dataset[dataset_name] = 0
            print(f"创建zarr文件: {zarr_path}")
        
        # 开始推理
        print("开始推理...")
        # Track statistics per dataset
        dataset_stats = {
            name: {
                'total_l1_loss': 0.0,
                'num_valid_samples': 0,
                'total_samples': 0
            }
            for name in self.dataset_names
        }
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
                    
                    actions_valid_num = np.sum(actions_valid_mask, axis=(1,2))
                    
                    is_valid = actions_valid_num > 0
                    if np.any(is_valid):
                        # 计算 Flow Matching 的 L1 loss
                        valid_pred_fm = pred_actions_fm * actions_valid_mask
                        valid_gt = gt_actions * actions_valid_mask
                        batch_l1_loss_fm = np.sum(np.abs(valid_pred_fm - valid_gt), axis=(1,2)) / actions_valid_num.clip(min=1)
                        batch_result["l1_loss"] = batch_l1_loss_fm
                        batch_result["l1_error"] = np.abs(valid_pred_fm - valid_gt)
                        
                        # Accumulate loss and count per dataset
                        for dataset_idx in range(len(self.dataset_names)):
                            dataset_name = self.dataset_names[dataset_idx]
                            mask = (dataset_indices == dataset_idx) & is_valid
                            if np.any(mask):
                                dataset_stats[dataset_name]['total_l1_loss'] += np.sum(batch_l1_loss_fm[mask])
                                dataset_stats[dataset_name]['num_valid_samples'] += np.sum(mask)
                
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
                            dataset_batch_result,
                            current_write_pos_per_dataset[dataset_name]
                        )
                        
                        num_samples = np.sum(mask)
                        current_write_pos_per_dataset[dataset_name] += num_samples
                        dataset_stats[dataset_name]['total_samples'] += num_samples
                
                # 定期保存检查点
                save_interval = getattr(cfg.inference, 'save_interval', None)
                if save_interval and actual_batch_count % save_interval == 0:
                    total_samples = sum(current_write_pos_per_dataset.values())
                    print(f"\n定期保存: 已处理 {actual_batch_count} 个batch，当前已保存 {total_samples} 个样本...")
                    for dataset_name in self.dataset_names:
                        print(f"  - {dataset_name}: {current_write_pos_per_dataset[dataset_name]} 个样本")
                        # Save metadata to each dataset's zarr file
                        dataset_zarr_roots[dataset_name].attrs['last_saved_batch'] = int(actual_batch_count)
                        dataset_zarr_roots[dataset_name].attrs['last_saved_samples'] = int(current_write_pos_per_dataset[dataset_name])
        
        # 更新最终元数据
        print("\n保存最终结果...")
        for dataset_name in self.dataset_names:
            zarr_root = dataset_zarr_roots[dataset_name]
            zarr_datasets = zarr_datasets_per_dataset[dataset_name]
            write_pos = current_write_pos_per_dataset[dataset_name]
            
            # Resize datasets to actual size
            for key, dataset in zarr_datasets.items():
                if write_pos < dataset.shape[0]:
                    dataset.resize((write_pos,) + dataset.shape[1:])
            
            # Save metadata for each dataset zarr file
            zarr_root.attrs['total_samples'] = int(write_pos)
            zarr_root.attrs['dataset_name'] = str(dataset_name)
            zarr_root.attrs['total_batches'] = int(actual_batch_count)
            if hasattr(self.cfg.inference, 'checkpoint_path'):
                zarr_root.attrs['checkpoint_path'] = str(self.cfg.inference.checkpoint_path)
            
            zarr_path = output_dir / f"inference_results_{dataset_name}.zarr"
            print(f"✓ {dataset_name}: {zarr_path} (样本数: {write_pos})")
        
        # 打印并保存每个数据集的统计信息
        print(f"\n推理完成!")
        all_stats = {
            'datasets': {},
            'overall': {
                'total_samples': 0,
                'total_valid_samples': 0,
                'avg_l1_loss': 0.0
            }
        }
        
        for dataset_name in self.dataset_names:
            stats = dataset_stats[dataset_name]
            num_samples = stats['total_samples']
            num_valid = stats['num_valid_samples']
            
            print(f"\n数据集: {dataset_name}")
            print(f"  样本数量: {num_samples}")
            
            # Ensure dataset_name is a string for JSON serialization
            dataset_name_str = str(dataset_name)
            
            if num_valid > 0:
                avg_l1_loss = stats['total_l1_loss'] / num_valid
                print(f"  有效样本数: {num_valid}")
                print(f"  平均L1损失: {avg_l1_loss:.4f}")
                
                all_stats['datasets'][dataset_name_str] = {
                    'total_samples': int(num_samples),
                    'valid_samples': int(num_valid),
                    'avg_l1_loss': float(avg_l1_loss)
                }
                
                all_stats['overall']['total_samples'] += int(num_samples)
                all_stats['overall']['total_valid_samples'] += int(num_valid)
                all_stats['overall']['avg_l1_loss'] += float(stats['total_l1_loss'])
            else:
                all_stats['datasets'][dataset_name_str] = {
                    'total_samples': int(num_samples),
                    'valid_samples': 0,
                    'avg_l1_loss': None
                }
                all_stats['overall']['total_samples'] += int(num_samples)
        
        # Calculate overall average
        if all_stats['overall']['total_valid_samples'] > 0:
            all_stats['overall']['avg_l1_loss'] /= all_stats['overall']['total_valid_samples']
            print(f"\n总体统计:")
            print(f"  总样本数: {all_stats['overall']['total_samples']}")
            print(f"  总有效样本数: {all_stats['overall']['total_valid_samples']}")
            print(f"  总体平均L1损失: {all_stats['overall']['avg_l1_loss']:.4f}")
        
        # Convert all values to native Python types for JSON serialization
        all_stats = convert_to_serializable(all_stats)
        
        # Save statistics to JSON
        stats_path = output_dir / "inference_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n统计信息已保存至: {stats_path}")
    
    def append_batch_to_zarr(self, root, zarr_datasets, batch_result, write_pos):
        """Append a batch of results to zarr file incrementally"""
        batch_size = len(batch_result['pred_actions'])
        
        # Keys to save
        keys_to_save = ['pred_actions', 'gt_actions', 'actions_valid_mask', 'origin_frame_indices']
        
        for key in keys_to_save:
            if key not in batch_result:
                continue
            
            data = batch_result[key]
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            if key not in zarr_datasets:
                # First time: create dataset with unknown size (resizable)
                estimated_size = write_pos + batch_size * 100  # Rough estimate
                shape = (estimated_size,) + data.shape[1:]
                
                zarr_datasets[key] = root.create_dataset(
                    key,
                    shape=shape,
                    dtype=data.dtype,
                    chunks=True,
                    compression='gzip',
                    compression_opts=1
                )
            
            # Resize if needed
            dataset = zarr_datasets[key]
            if write_pos + batch_size > dataset.shape[0]:
                new_size = max(write_pos + batch_size, dataset.shape[0] * 2)
                dataset.resize((new_size,) + dataset.shape[1:])
            
            # Write data
            dataset[write_pos:write_pos + batch_size] = data


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