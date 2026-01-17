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
    
    def _get_origin_index(self, dataset, idx: int):
        """Get the origin index (buffer_start_idx) for a given dataset index
        
        Returns:
            buffer_start_idx: The starting index of the sequence in the original zarr dataset.
            
        Note:
            - origin_indices: sequence start index (buffer_start_idx)
            - origin_indices + history: observation frame index (the frame that the model actually sees)
            To access the observation frame data (instruction, intrinsic, etc.), use origin_indices + history.
        """
        curr_idx = idx
        dataset_idx = 0
        local_idx = None
        
        # Find which sampler this index belongs to
        for i, length in enumerate(dataset.sampler_lens):
            if curr_idx < length:
                local_idx = curr_idx
                dataset_idx = i
                break
            curr_idx -= length
        
        if local_idx is None:
            raise ValueError(f"Index {idx} is out of range")
        
        # Get buffer_start_idx from sampler indices
        sampler = dataset.samplers[dataset_idx]
        buffer_start_idx, buffer_end_idx, _, _ = sampler.indices[local_idx]
        
        return buffer_start_idx
    
    def _get_origin_indices(self, dataset, sample_indices):
        """Get origin indices for each sample
        
        Uses set_return_raw_sample to get raw samples, then extracts buffer_start_idx
        from sampler indices and calculates the observation frame index.
        
        Args:
            dataset: The dataset instance
            sample_indices: Array of sample indices
            
        Returns:
            origin_indices: Array of observation frame indices for each sample
        """
        # Temporarily enable raw sample mode to get dataset information
        original_return_raw_sample = dataset.return_raw_sample
        dataset.set_return_raw_sample(True)
        
        try:
            origin_indices = []
            history = dataset.history if hasattr(dataset, 'history') else 0
            
            # Get raw samples and extract origin indices
            # We need to reconstruct the actual observation frame index for each sample
            # by finding the original idx used in sampler creation
            for sample_idx in sample_indices:
                # Get raw sample to access dataset_idx
                raw_sample = dataset[int(sample_idx)]
                dataset_idx = raw_sample['dataset_idx'].item() if hasattr(raw_sample['dataset_idx'], 'item') else raw_sample['dataset_idx']
                
                # Find local_idx for this sample
                curr_idx = int(sample_idx)
                local_idx = None
                for i, length in enumerate(dataset.sampler_lens):
                    if curr_idx < length:
                        local_idx = curr_idx
                        break
                    curr_idx -= length
                
                if local_idx is None:
                    raise ValueError(f"Index {sample_idx} is out of range")
                
                # Get buffer_start_idx and sample_start_idx from sampler indices
                sampler = dataset.samplers[dataset_idx]
                buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = sampler.indices[local_idx]
                
                # Calculate actual observation frame index
                # The observation frame is at position 'history' in the loaded sequence
                # For image data, sampler loads history+1 frames starting from buffer_start_idx
                # The observation frame in the original dataset is at buffer_start_idx + history
                # However, when pad_before > 0, multiple samples may have the same buffer_start_idx
                # but different actual observation frames. We need to reconstruct the original idx.
                
                # Reconstruct original idx from sample_start_idx and buffer_start_idx
                replay_buffer = sampler.replay_buffer
                episode_ends = replay_buffer.episode_ends
                episode_idx = np.searchsorted(episode_ends, buffer_start_idx)
                episode_start_idx = episode_ends[episode_idx - 1] if episode_idx > 0 else 0
                
                # From sampler logic: start_offset = buffer_start_idx - (idx + start_idx)
                # sample_start_idx = start_offset
                # So: idx = buffer_start_idx - episode_start_idx - sample_start_idx
                original_idx = buffer_start_idx - episode_start_idx - sample_start_idx
                
                # Calculate actual observation frame index
                # raw_sample['image'][0] corresponds to sample['image'][history]
                # sample['image'] is loaded from buffer_start_idx, with padding at the beginning
                # When pad_before > 0, sample['image'][:sample_start_idx] are padded copies
                # So sample['image'][history] corresponds to:
                # - If history < sample_start_idx: sample['image'][history] is padding, actual frame is at buffer_start_idx
                # - If history >= sample_start_idx: sample['image'][history] = buffer_start_idx + history - sample_start_idx
                # But when original_idx < 0, buffer_start_idx = episode_start_idx (clamped)
                # So the actual observation frame index is: episode_start_idx + original_idx + history
                # where original_idx = buffer_start_idx - episode_start_idx - sample_start_idx
                # Simplifying: observation_frame_idx = buffer_start_idx - sample_start_idx + history
                # But this only works when original_idx >= 0
                # When original_idx < 0: observation_frame_idx = episode_start_idx + original_idx + history
                # = episode_start_idx + (buffer_start_idx - episode_start_idx - sample_start_idx) + history
                # = buffer_start_idx - sample_start_idx + history
                # So in both cases: observation_frame_idx = buffer_start_idx - sample_start_idx + history
                
                # However, we need to account for padding: if history < sample_start_idx,
                # sample['image'][history] is padding, so it corresponds to buffer_start_idx
                if history < sample_start_idx:
                    # sample['image'][history] is padding, actual frame is at buffer_start_idx
                    observation_frame_idx = buffer_start_idx
                else:
                    # sample['image'][history] is real data at buffer_start_idx + history - sample_start_idx
                    observation_frame_idx = buffer_start_idx + history - sample_start_idx
                
                # Save the actual observation frame index, not buffer_start_idx
                origin_indices.append(observation_frame_idx)
            
            origin_indices = np.array(origin_indices)
            return origin_indices
        finally:
            # Restore original return_raw_sample setting
            dataset.set_return_raw_sample(original_return_raw_sample)
    
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
        
        # 根据配置选择数据集
        if cfg.inference.use_val_dataset:
            inference_dataset = self.dataset.get_validation_dataset()
            print("使用验证数据集")
        else:
            inference_dataset = self.dataset
            print("使用训练数据集")
        
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
        
        # 开始推理
        print("开始推理...")
        results = {}
        total_l1_loss = 0.0
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
                
                # 获取实际的batch大小（最后一个batch可能小于batch_size）
                actual_batch_size = batch["input_ids"].shape[0]
                
                # 计算实际的样本索引（考虑skip_first）
                batch_start_idx = batch_idx * cfg.inference.batch_size
                batch_end_idx = min(batch_start_idx + actual_batch_size, len(inference_dataset))
                sample_indices = np.arange(start=batch_start_idx, stop=batch_end_idx)
                
                # 预处理
                inputs = self.preprocess_batch(batch)
                
                # 推理
                with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    pred_actions = self.model("infer_action", inputs)
                
                pred_actions = self.normalizer['actions'].unnormalize(pred_actions.cpu().float().numpy())
                pred_actions = pred_actions
                # 保存结果
                batch_result = {
                    "pred_actions": pred_actions,
                }
                
                # 如果有ground truth，也保存并计算误差
                if "actions" in inputs:
                    gt_actions = inputs["actions"]
                    gt_actions = self.normalizer['actions'].unnormalize(gt_actions.cpu().float().numpy())
                    actions_valid_mask = inputs["actions_valid_mask"].cpu().numpy()
                    
                    batch_result["gt_actions"] = gt_actions
                    batch_result["actions_valid_mask"] = actions_valid_mask
                    
                    # 计算L1 loss
                    valid_pred = pred_actions * actions_valid_mask
                    valid_gt = gt_actions * actions_valid_mask
                    actions_valid_num = np.sum(actions_valid_mask, axis=(1,2))
                    
                    if np.any(actions_valid_num > 0):
                        batch_l1_loss = np.sum(np.abs(valid_pred - valid_gt), axis=(1,2)) / actions_valid_num.clip(min=1)
                        batch_result["l1_loss"] = batch_l1_loss
                        batch_result["l1_error"] = np.abs(valid_pred - valid_gt)
                        total_l1_loss += np.mean(batch_l1_loss)
                        num_valid_samples += 1
                
                # 保存样本索引用于加载原始数据
                batch_result["sample_indices"] = sample_indices
                
                # 获取每个样本对应的原始数据索引 (origin_frame_indices)
                # origin_frame_indices 是观察帧在原始数据集中的索引
                origin_frame_indices = self._get_origin_indices(inference_dataset, sample_indices)
                
                batch_result["origin_frame_indices"] = origin_frame_indices
                
                actual_batch_count += 1
                
                self.update_results(results, batch_result)
        
        # 保存最终结果
        print("保存最终结果...")
        self.save_results(results, output_dir, suffix="_final")
        
        # 打印统计信息
        if num_valid_samples > 0:
            avg_l1_loss = total_l1_loss / num_valid_samples
            print(f"\n推理完成!")
            print(f"总样本数: {len(results['sample_indices'])}")
            print(f"有效样本数: {num_valid_samples}")
            print(f"平均L1损失: {avg_l1_loss:.4f}")
            
            # 保存统计信息
            stats = {
                "total_samples": len(results['sample_indices']),
                "valid_samples": num_valid_samples,
                "avg_l1_loss": avg_l1_loss,
            }
            stats_path = output_dir / "inference_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"统计信息已保存至: {stats_path}")
        else:
            print(f"\n推理完成! 总样本数: {len(results['sample_indices']) if 'sample_indices' in results else 0}")
    
    def save_results(self, results, output_dir, suffix=""):
        """保存推理结果为zarr格式，包含完整的原始数据"""
        output_path = output_dir / f"inference_results{suffix}.zarr"
        
        print("\n正在保存推理结果...")
        # 合并所有batch的results
        merged_results = {}
        for key in results:
            merged_results[key] = np.concatenate(results[key], axis=0)
        
        total_samples = len(merged_results['pred_actions'])
        inference_indices = merged_results['sample_indices']
        origin_frame_indices = merged_results['origin_frame_indices']
        print(f"总共 {total_samples} 个样本")
        
        # 创建输出zarr
        store = zarr.DirectoryStore(str(output_path))
        root = zarr.group(store=store, overwrite=True)
        
        # 保存所有数据
        for key, value in tqdm(merged_results.items(), desc="保存数据"):
            if value.nbytes > 1024 * 1024:  # 大于1MB使用压缩
                root.create_dataset(
                    key, 
                    data=value, 
                    chunks=True,
                    compression='gzip',
                    compression_opts=1
                )
            else:
                root.create_dataset(key, data=value)
        origin_dataset = zarr.open(self.cfg.vla_dataset_paths[0], mode='r')
        included_key = [
            'image',
            'depth', 
            'state/wrist',
            'state/shape', 
            'state/mano',
            'state/fingertips', 
            'instruction',
            'instruction_num', 
            'extrinsic',
            'intrinsic',
            'presence',
        ]
        # Use origin_frame_indices directly to index original dataset (already includes history offset)
        for key in included_key:
            if key in origin_dataset['data']: 
                root.create_dataset(
                    key, 
                    data=origin_dataset['data'][key][origin_frame_indices], 
                    chunks=True,
                    compression='gzip',
                    compression_opts=1
                )
        
        # 保存元信息
        root.attrs['total_samples'] = total_samples
        root.attrs['zarr_paths'] = self.cfg.vla_dataset_paths[0]
        if hasattr(self.cfg.inference, 'checkpoint_path'):
            root.attrs['checkpoint_path'] = self.cfg.inference.checkpoint_path
        
        print(f"\n✓ 结果已保存至: {output_path}")
        print(f"  源数据集: {self.cfg.vla_dataset_paths}")
        print(f"  样本数量: {total_samples}")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("src/config")),
    config_name="experiment/inference_pretrain_legendvla"
)
def main(cfg):
    inference = LegendVLAInference(cfg)
    inference.run()


if __name__ == "__main__":
    main()
