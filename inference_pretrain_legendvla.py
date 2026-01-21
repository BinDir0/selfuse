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
        
        # 初始化 zarr 文件（用于增量写入）
        output_path = output_dir / "inference_results.zarr"
        store = zarr.DirectoryStore(str(output_path))
        root = zarr.group(store=store, overwrite=True)
        zarr_datasets = {}  # Track zarr datasets for incremental writing
        current_write_pos = 0  # Track current write position
        
        # 开始推理
        print("开始推理...")
        total_l1_loss = 0.0
        total_l1_loss_ar = 0.0
        num_valid_samples = 0
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
                    
                    # 2. Autoregressive (Discrete) Inference
                    output_ar = self.model("infer_discrete_action", inputs, action_end_token_id=self.vla_processor.action_end_token_id)
                    pred_token_ids = output_ar['generated_ids']
                    
                   
                # --- 处理 Flow Matching 结果 ---
                pred_actions_fm = pred_actions_fm.cpu().float().numpy()
                pred_actions_fm = self.normalizer['actions'].unnormalize(pred_actions_fm)
                
                # --- 处理 AR Discrete 结果 ---
                pred_token_ids_np = pred_token_ids.cpu().numpy()
                answer_start_idx_np = batch["answer_start_idx"].cpu().numpy()
                pred_actions_ar_list = []
                
                
                # 逐个样本进行解码
          
                for i in range(len(pred_token_ids_np)):

                    single_sample_tokens = pred_token_ids_np[i]
                    
                    decoded = self.vla_processor.decode(single_sample_tokens)
                    
                    if 'actions' in decoded:
                        # decoded['actions'] 是 [Horizon, Action_Dim]
                        # 确保形状和 Flow Matching 的结果一致，防止后面 stack 报错
                        decoded_action = decoded['actions']
                        
                        # 简单的形状检查与修正 (如果 AR 生成的长度不对)
                        target_shape = pred_actions_fm[i].shape # [Horizon, Dim]
                        if decoded_action.shape != target_shape:
                            # 如果长度不够，补0；如果太长，截断 (视情况而定，这里做简单的 resize/padding)
                            new_action = np.zeros(target_shape, dtype=decoded_action.dtype)
                            min_len = min(len(decoded_action), len(new_action))
                            new_action[:min_len] = decoded_action[:min_len]
                            pred_actions_ar_list.append(new_action)
                        else:
                            pred_actions_ar_list.append(decoded_action)
                    else:
                        # 解码失败 (例如生成了无效 token)，Fallback 到全 0
                        pred_actions_ar_list.append(np.zeros_like(pred_actions_fm[i]))
            
                # 堆叠成 batch numpy 数组
                pred_actions_ar = np.stack(pred_actions_ar_list)

                # 保存结果
                batch_result = {
                    "pred_actions": pred_actions_fm,      # Flow Matching 结果
                    "pred_actions_ar": pred_actions_ar,   # AR Discrete 结果
                }
                
                # 如果有ground truth，也保存并计算误差 (基于 Flow Matching 和 AR Discrete 结果计算 Loss)
                if "actions" in inputs:
                    gt_actions = inputs["actions"].cpu().float().numpy()
                    gt_actions = self.normalizer['actions'].unnormalize(gt_actions)
                    actions_valid_mask = inputs["actions_valid_mask"].cpu().numpy()
                    
                    batch_result["gt_actions"] = gt_actions
                    batch_result["actions_valid_mask"] = actions_valid_mask
                    
                    actions_valid_num = np.sum(actions_valid_mask, axis=(1,2))
                    
                    if np.any(actions_valid_num > 0):
                        # 计算 Flow Matching 的 L1 loss
                        valid_pred_fm = pred_actions_fm * actions_valid_mask
                        valid_gt = gt_actions * actions_valid_mask
                        batch_l1_loss_fm = np.sum(np.abs(valid_pred_fm - valid_gt), axis=(1,2)) / actions_valid_num.clip(min=1)
                        batch_result["l1_loss"] = batch_l1_loss_fm
                        batch_result["l1_error"] = np.abs(valid_pred_fm - valid_gt)
                        total_l1_loss += np.mean(batch_l1_loss_fm)
                        
                        # 计算 AR Discrete 的 L1 loss
                        valid_pred_ar = pred_actions_ar * actions_valid_mask
                        batch_l1_loss_ar = np.sum(np.abs(valid_pred_ar - valid_gt), axis=(1,2)) / actions_valid_num.clip(min=1)
                        batch_result["l1_loss_ar"] = batch_l1_loss_ar
                        batch_result["l1_error_ar"] = np.abs(valid_pred_ar - valid_gt)
                        total_l1_loss_ar += np.mean(batch_l1_loss_ar)
                        
                        num_valid_samples += 1
                
                # 保存样本索引用于加载原始数据
                batch_result["sample_indices"] = sample_indices
                
                # 获取每个样本对应的原始数据索引
                origin_frame_indices, dataset_indices = self._get_origin_indices(inference_dataset, sample_indices)
                
                batch_result["origin_frame_indices"] = origin_frame_indices
                batch_result["dataset_indices"] = dataset_indices
                
                actual_batch_count += 1
                
                # 直接增量写入到 zarr
                self.append_batch_to_zarr(root, zarr_datasets, batch_result, current_write_pos)
                current_write_pos += len(batch_result['pred_actions'])
                
                # 定期保存检查点
                save_interval = getattr(cfg.inference, 'save_interval', None)
                if save_interval and actual_batch_count % save_interval == 0:
                    print(f"\n定期保存: 已处理 {actual_batch_count} 个batch，当前已保存 {current_write_pos} 个样本...")
                    root.attrs['last_saved_batch'] = actual_batch_count
                    root.attrs['last_saved_samples'] = current_write_pos
        
        # 更新最终元数据
        print("\n保存最终结果...")
        for key, dataset in zarr_datasets.items():
            if current_write_pos < dataset.shape[0]:
                dataset.resize((current_write_pos,) + dataset.shape[1:])
        
        root.attrs['total_samples'] = current_write_pos
        root.attrs['total_batches'] = actual_batch_count
        if hasattr(self.cfg.inference, 'checkpoint_path'):
            root.attrs['checkpoint_path'] = self.cfg.inference.checkpoint_path
        
        print(f"\n✓ 结果已保存至: {output_path}")
        print(f"  样本数量: {current_write_pos}")
        
        # 打印统计信息
        if current_write_pos > 0:
            if num_valid_samples > 0:
                avg_l1_loss = total_l1_loss / num_valid_samples
                avg_l1_loss_ar = total_l1_loss_ar / num_valid_samples
                print(f"\n推理完成!")
                print(f"总样本数: {current_write_pos}")
                print(f"有效样本数: {num_valid_samples}")
                print(f"平均L1损失 (FM): {avg_l1_loss:.4f}")
                print(f"平均L1损失 (AR): {avg_l1_loss_ar:.4f}")
                
                stats = {
                    "total_samples": current_write_pos,
                    "valid_samples": num_valid_samples,
                    "avg_l1_loss": avg_l1_loss,
                    "avg_l1_loss_ar": avg_l1_loss_ar,
                }
                stats_path = output_dir / "inference_stats.json"
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"统计信息已保存至: {stats_path}")
            else:
                print(f"\n推理完成! 总样本数: {current_write_pos}")
    
    def append_batch_to_zarr(self, root, zarr_datasets, batch_result, write_pos):
        """Append a batch of results to zarr file incrementally"""
        batch_size = len(batch_result['pred_actions'])
        
        # Keys to save - 增加了 'pred_actions_ar' 及其 loss
        keys_to_save = ['pred_actions', 'pred_actions_ar', 'gt_actions', 'actions_valid_mask', 'origin_frame_indices']
        
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