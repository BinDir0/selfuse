# TODO: The dataset loading section of this script (LegendVLAInference._init_dataset_and_processor)
# still references zarr-based datasets. It needs to be migrated to use WebDataset
# (VLAWdsDataset / UnifiedWdsDataset) for consistency with the training pipeline.

import os
import random
import hydra
import torch
import accelerate
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
import numcodecs
from PIL import Image

from src.policy.legendvla import LegendVLA
from src.utils.pytorch_util import dict_apply
from src.utils.metric import compute_and_save_metrics_from_data

OmegaConf.register_new_resolver("eval", eval, replace=True)


class LegendVLAInference:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
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
            policy_cfg.diffloss.num_sampling_steps = f"{cfg.diffusion_sampling_steps}"
            policy_cfg.diffloss.use_ddim_sampling = cfg.diffusion_use_ddim_sampling
            policy_cfg.cfg.num_inference_steps = cfg.flow_sampling_steps
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

        self.model.eval()
        
        # Get inference mode: "ar" (autoregressive action), "flow" (flow matching), or "vlm" (text)
        self.mode = cfg.inference.get("mode", "flow")
        if self.mode not in ["ar", "flow", "vlm"]:
            raise ValueError(
                f"Invalid inference mode: {self.mode}. Must be 'ar', 'flow', or 'vlm'"
            )
        
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
        input_ids = batch["input_ids"]
        
        # Get unwrapped model for mask building
        model = self.model
        if hasattr(self.model, 'module'):
            model = self.model.module
        
        
        inputs = {
            "input_ids": input_ids,
            'attention_mask': batch["attention_mask"],
            "pixel_values": batch["pixel_values"].to(self.dtype),
            "is_vla_data": batch["is_vla_data"],
        }

        if self.mode == 'flow': 
            # 构建causal mask和position ids
            causal_mask, vlm_position_ids, action_position_ids = (
                model.build_causal_mask_and_position_ids(
                    batch["attention_mask"], 
                    batch["answer_start_idx"], 
                    batch["n_actions"],
                    self.dtype
                )
            )
            max_vlm_tokens = input_ids.shape[-1]
            vlm_mask, action_mask = (
                model.split_full_mask_into_submasks(causal_mask, max_vlm_tokens)
            )
            inputs["causal_mask"] = causal_mask
            inputs["vlm_mask"] = vlm_mask
            inputs["action_mask"] = action_mask
            inputs["vlm_position_ids"] = vlm_position_ids
            inputs["action_position_ids"] = action_position_ids

        if "states" in batch:
            inputs["states"] = batch["states"].to(self.dtype)
            inputs["n_states"] = batch["n_states"]
        
        if "depth_values" in batch:
            inputs["depth_values"] = batch["depth_values"].to(self.dtype)
            inputs["has_depth_values"] = batch["has_depth_values"]

        # 添加ground truth actions用于对比（如果有）
        if "actions" in batch:
            inputs["actions"] = batch["actions"].to(self.dtype)
            inputs["n_actions"] = batch["n_actions"]
            inputs["actions_valid_mask"] = batch["actions_valid_mask"]
        
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
        if self.processor is not None: 
            generation_params['eos_token_id'] = self.processor.eos_token_id
        return generation_params
    
    def _save_sample_images(self, pixel_values, num_images, output_dir, dataset_name, dataset_local_idx):
        images_dir = os.path.join(str(output_dir), "vlm_outputs", "images", dataset_name)
        os.makedirs(images_dir, exist_ok=True)
        image_paths = []
        for img_idx in range(num_images):
            image = pixel_values[img_idx]
            image_path = os.path.join(
                images_dir,
                f"p{self.accelerator.process_index}_s{dataset_local_idx}_img{img_idx}.png"
            )
            Image.fromarray(image).save(image_path)
            image_paths.append(image_path)
        return image_paths

    def _unnormalize_actions(self, actions):
        if self.normalizer is None:
            return actions
        if getattr(self, "use_relative_action", False):
            return self.normalizer['actions'].unnormalize(actions)
        return self.normalizer['motions'].unnormalize(actions)
    
    def _init_dataset_and_processor(self, cfg):
        if self.is_main_process:
            print("正在初始化数据集...")
        if self.mode == "vlm":
            if self.model_cfg is not None and self.model_cfg.dataset is not None:
                vlm_dataset_cfg = self.model_cfg.dataset.vlm_dataset
            elif hasattr(cfg, "vlm_dataset"):
                vlm_dataset_cfg = cfg.vlm_dataset
            else:
                vlm_dataset_cfg = cfg.dataset
            if hasattr(cfg, "vlm_dataset_paths"):
                vlm_dataset_cfg.dataset_paths = cfg.vlm_dataset_paths
            vlm_dataset_cfg.mode = "infer-ar"
            vlm_dataset_cfg.return_dataset_info = True
            self.dataset = hydra.utils.instantiate(vlm_dataset_cfg)
        else:
            if self.model_cfg is not None and self.model_cfg.dataset is not None:
                vla_dataset_cfg = self.model_cfg.dataset.vla_dataset
            elif hasattr(cfg, "vla_dataset"):
                vla_dataset_cfg = cfg.vla_dataset
            else:
                vla_dataset_cfg = cfg.dataset
            if hasattr(cfg, "vla_dataset_paths"):
                vla_dataset_cfg.zarr_paths = cfg.vla_dataset_paths
            vla_dataset_cfg.mode = "infer" if self.mode == "flow" else "infer-ar"
            vla_dataset_cfg.return_dataset_info = True
            self.dataset = hydra.utils.instantiate(vla_dataset_cfg)

        if self.is_main_process:
            print("正在初始化processor...")
        if self.mode == "vlm":
            if self.model_cfg is not None and getattr(self.model_cfg, "vlm_processor", None) is not None:
                self.processor = hydra.utils.instantiate(self.model_cfg.vlm_processor)
            else:
                raise ValueError("model_cfg 中未指定 vlm_processor")
        else:
            if self.model_cfg is not None and self.model_cfg.vla_processor is not None:
                self.processor = hydra.utils.instantiate(self.model_cfg.vla_processor)
            else:
                raise ValueError("model_cfg 中未指定 vla_processor")
        if hasattr(self.processor, "tokenizer_padding"):
            self.processor.tokenizer_padding = cfg.tokenizer_padding
        self.dataset.set_preprocessor(self.processor)

        if hasattr(self.dataset, "use_relative_action"):
            self.use_relative_action = self.dataset.use_relative_action
        elif hasattr(self.dataset, "vla_dataset") and hasattr(self.dataset.vla_dataset, "use_relative_action"):
            self.use_relative_action = self.dataset.vla_dataset.use_relative_action
        else:
            self.use_relative_action = False

    def _save_vlm_json(self, output_dir, vlm_records):
        json_path = output_dir / "pred_texts.json"
        with open(json_path, "w") as f:
            json.dump(vlm_records, f, ensure_ascii=False, indent=2)
        print(f"VLM 结果已保存至: {json_path}")

    def _prepare_vla_output_dirs(self, output_dir):
        zarr_root = None
        zarr_datasets = {}
        if self.is_main_process:
            zarr_path = output_dir / "inference_results.zarr"
            store = zarr.DirectoryStore(str(zarr_path))
            zarr_root = zarr.group(store=store, overwrite=True)
            print(f"创建zarr文件: {zarr_path}")
        return zarr_root, zarr_datasets

    def _save_vla_metrics(self, output_dir, results):
        print("\n开始计算统计信息和生成可视化...")
        dataset_output_dir = output_dir / "metrics"
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        merged_results = {}
        for key, chunks in results.items():
            if len(chunks) == 0:
                continue
            if isinstance(chunks[0], np.ndarray):
                merged_results[key] = np.concatenate(chunks, axis=0)
            else:
                merged_results[key] = np.array(chunks)
        compute_and_save_metrics_from_data(
            merged_results,
            dataset_output_dir,
            "all",
        )
        print("\n统计信息和可视化计算完成！")

    def _build_vlm_local_records(
        self,
        vlm_outputs,
        actual_batch_size,
        output_dir,
        dataset_names,
        dataset_local_indices,
    ):
        if vlm_outputs is None:
            return None
        pixel_values_np = vlm_outputs["images"]
        vlm_local_records = []
        for i in range(actual_batch_size):
            dataset_name = dataset_names[i] if dataset_names is not None else "vlm"
            image_paths = self._save_sample_images(
                pixel_values_np[i],
                pixel_values_np.shape[1],
                output_dir,
                dataset_name,
                dataset_local_indices[i],
            )
            vlm_local_records.append({
                "dataset": dataset_name,
                "dataset_local_idx": int(dataset_local_indices[i]),
                "instruction": vlm_outputs["instructions"][i],
                "image_paths": image_paths,
                "pred_text": vlm_outputs["pred_texts"][i],
            })
        return vlm_local_records
    
    def update_results(self, results, batch_result): 
        for key in batch_result:
            if key not in results:
                results[key] = []
            results[key].append(batch_result[key])

    def run(self):
        cfg = self.cfg
        
        self._init_dataset_and_processor(cfg)
        
        # 加载normalizer
        if self.mode != 'vlm': 
            if self.is_main_process:
                print("正在加载normalizer...")
            if self.model_cfg is not None and self.model_cfg.training.normalizer_path is not None:
                self.normalizer = pickle.load(open(self.model_cfg.training.normalizer_path, 'rb'))
                self.dataset.set_normalizer(self.normalizer)
                if self.is_main_process:
                    print(f"成功加载normalizer: {self.model_cfg.training.normalizer_path}")
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
                print("使用完整数据集")
        
        # 创建dataloader
        dataloader = DataLoader(
            inference_dataset, 
            collate_fn=inference_dataset.get_collator(),
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=cfg.dataloader.shuffle,
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
        
        save_zarr = cfg.inference.get("save_zarr", True)
        if self.mode == "vlm":
            vlm_records = []
        else:
            if save_zarr:
                zarr_root, zarr_datasets = self._prepare_vla_output_dirs(output_dir)
            else:
                zarr_root, zarr_datasets = None, {}
            vlm_records = None
            results = {} if self.is_main_process else None
        
        # 开始推理
        if self.is_main_process:
            print("开始推理...")
        # Track statistics per dataset
        actual_batch_count = 0  # Track actual batch count (after skipping)
        
        # Track random samples for attention visualization
        attention_sample_count = int(cfg.inference.get("attention_sample_count", 0))
        random_attention_samples = []
        total_attention_seen = 0
        
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
                inputs = self.preprocess_batch(batch)
                
                pred_actions = None
                vlm_attn_maps = None
                action_expert_attn_maps = None
                prefill_vlm_hidden_states = None
                prefill_image_hidden_states = None
                prefill_state_hidden_states = None
                prefill_text_hidden_states = None
                generated_hidden_states = None
                # Inference (different for each mode)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype), torch.inference_mode():
                    if self.mode == "flow":
                        # Flow Matching Inference (normalized actions)
                        pred_actions, vlm_attn_maps, action_expert_attn_maps = self.model(
                            "infer_action", inputs, return_attn_weights=True
                        )
                    elif self.mode == "ar":
                        # Autoregressive action inference (normalized actions)
                        generation_params = self._preprocess_for_autoregressive(inputs)
                        generation_output = self.model(
                            "infer_vla", inputs, **generation_params, return_attn_weights=True
                        )
                        pred_actions = generation_output["generated_actions"]
                        vlm_attn_maps = generation_output.get("attn_weights")
                        prefill_vlm_hidden_states = generation_output["prefill_vlm_hidden_states"]
                        prefill_image_hidden_states = generation_output["prefill_image_hidden_states"]
                        prefill_state_hidden_states = generation_output["prefill_state_hidden_states"]
                        prefill_text_hidden_states = generation_output["prefill_text_hidden_states"]
                        generated_hidden_states = generation_output["generated_hidden_states"]
                    else:  # self.mode == "vlm"
                        generation_params = self._preprocess_for_autoregressive(inputs)
                        generation_output = self.model(
                            "infer_vlm", inputs, **generation_params, return_attn_weights=True
                        )
                        vlm_attn_maps = generation_output["attn_weights"]
                        vlm_outputs = self.processor.postprocess(
                            generated_ids=generation_output['generated_ids'],
                            pixel_values=inputs["pixel_values"],
                            input_ids=inputs["input_ids"],
                        )
                
                if self.mode in ["ar", "flow"] and pred_actions is not None:
                    pred_actions = self._unnormalize_actions(pred_actions)
                
                # 保存结果
                batch_result = {}
                if self.mode in ["ar", "flow"]:
                    batch_result["pred_actions"] = pred_actions
                if self.mode == "ar":
                    hidden_dim = None
                    for candidate in (
                        prefill_vlm_hidden_states,
                        prefill_image_hidden_states,
                        prefill_state_hidden_states,
                        prefill_text_hidden_states,
                        generated_hidden_states,
                    ):
                        if candidate is not None:
                            hidden_dim = candidate.shape[-1]
                            break
                
                dataset_name_list = batch["dataset_name"]
                dataset_local_idx = batch["dataset_local_idx"]
                if torch.is_tensor(dataset_local_idx):
                    dataset_local_idx_list = dataset_local_idx.cpu().numpy().astype(int).tolist()
                batch_result["dataset_local_idx"] = dataset_local_idx
                
                # 如果有ground truth，也保存并计算误差
                if self.mode in ["ar", "flow"] and "actions" in inputs:
                    gt_actions = self._unnormalize_actions(inputs["actions"])
                    actions_valid_mask = inputs["actions_valid_mask"]
                    
                    batch_result["gt_actions"] = gt_actions
                    batch_result["actions_valid_mask"] = actions_valid_mask
                    
                if vlm_attn_maps is not None and attention_sample_count > 0:
                    self._update_random_attention_samples(
                        random_attention_samples,
                        attention_sample_count,
                        total_attention_seen,
                        vlm_attn_maps,
                        action_expert_attn_maps,
                        inputs,
                        batch_idx,
                        dataset_name_list,
                        dataset_local_idx_list,
                    )
                    total_attention_seen += vlm_attn_maps.shape[1]

                vlm_local_records = None
                if self.mode == "vlm":
                    vlm_local_records = self._build_vlm_local_records(
                        vlm_outputs=vlm_outputs,
                        actual_batch_size=actual_batch_size,
                        output_dir=output_dir,
                        dataset_names=dataset_name_list,
                        dataset_local_indices=dataset_local_idx_list,
                    )
                
                actual_batch_count += 1
                
                # Gather results from all processes to main process
                gathered_batch_results = self.accelerator.gather(batch_result)
                gathered_batch_results_np = dict_apply(
                    gathered_batch_results, lambda x: x.cpu().float().numpy()
                )
                if self.mode == "ar":
                    hidden_states_payload = {
                        "prefill_vlm_hidden_states": (
                            prefill_vlm_hidden_states.detach().cpu()
                            if prefill_vlm_hidden_states is not None
                            else None
                        ),
                        "prefill_image_hidden_states": (
                            prefill_image_hidden_states.detach().cpu()
                            if prefill_image_hidden_states is not None
                            else None
                        ),
                        "prefill_state_hidden_states": (
                            prefill_state_hidden_states.detach().cpu()
                            if prefill_state_hidden_states is not None
                            else None
                        ),
                        "prefill_text_hidden_states": (
                            prefill_text_hidden_states.detach().cpu()
                            if prefill_text_hidden_states is not None
                            else None
                        ),
                        "generated_hidden_states": (
                            generated_hidden_states.detach().cpu()
                            if generated_hidden_states is not None
                            else None
                        ),
                    }
                    gathered_hidden_states = accelerate.utils.gather_object([hidden_states_payload])
                    if self.is_main_process:
                        prefill_list = []
                        prefill_image_list = []
                        prefill_state_list = []
                        prefill_text_list = []
                        generated_list = []
                        for item in gathered_hidden_states:
                            if item is None:
                                continue
                            if item.get("prefill_vlm_hidden_states") is not None:
                                prefill_list.append(
                                    item["prefill_vlm_hidden_states"].float().numpy()
                                )
                            if item.get("prefill_image_hidden_states") is not None:
                                prefill_image_list.append(
                                    item["prefill_image_hidden_states"].float().numpy()
                                )
                            if item.get("prefill_state_hidden_states") is not None:
                                prefill_state_list.append(
                                    item["prefill_state_hidden_states"].float().numpy()
                                )
                            if item.get("prefill_text_hidden_states") is not None:
                                prefill_text_list.append(
                                    item["prefill_text_hidden_states"].float().numpy()
                                )
                            if item.get("generated_hidden_states") is not None:
                                generated_list.append(
                                    item["generated_hidden_states"].float().numpy()
                                )
                        if prefill_list:
                            gathered_batch_results_np["prefill_vlm_hidden_states"] = np.concatenate(
                                prefill_list, axis=0
                            )
                        elif hidden_dim is not None:
                            gathered_batch_results_np["prefill_vlm_hidden_states"] = np.empty(
                                (0, hidden_dim), dtype=np.float32
                            )
                        if prefill_image_list:
                            gathered_batch_results_np["prefill_image_hidden_states"] = np.concatenate(
                                prefill_image_list, axis=0
                            )
                        elif hidden_dim is not None:
                            gathered_batch_results_np["prefill_image_hidden_states"] = np.empty(
                                (0, hidden_dim), dtype=np.float32
                            )
                        if prefill_state_list:
                            gathered_batch_results_np["prefill_state_hidden_states"] = np.concatenate(
                                prefill_state_list, axis=0
                            )
                        elif hidden_dim is not None:
                            gathered_batch_results_np["prefill_state_hidden_states"] = np.empty(
                                (0, hidden_dim), dtype=np.float32
                            )
                        if prefill_text_list:
                            gathered_batch_results_np["prefill_text_hidden_states"] = np.concatenate(
                                prefill_text_list, axis=0
                            )
                        elif hidden_dim is not None:
                            gathered_batch_results_np["prefill_text_hidden_states"] = np.empty(
                                (0, hidden_dim), dtype=np.float32
                            )
                        if generated_list:
                            gathered_batch_results_np["generated_hidden_states"] = np.concatenate(
                                generated_list, axis=0
                            )
                        elif hidden_dim is not None:
                            gathered_batch_results_np["generated_hidden_states"] = np.empty(
                                (0, hidden_dim), dtype=np.float32
                            )
                if self.mode == "vlm":
                    gathered_records = accelerate.utils.gather_object(vlm_local_records)
                    if self.is_main_process and vlm_records is not None:
                        for records in gathered_records:
                            if records is None:
                                continue
                            if isinstance(records, list):
                                vlm_records.extend(records)
                            else:
                                vlm_records.append(records)
                else:
                    gathered_dataset_names = accelerate.utils.gather_object(dataset_name_list)
                    dataset_names_all = []
                    for names in gathered_dataset_names:
                        if names is None:
                            continue
                        if isinstance(names, str):
                            dataset_names_all.append(names)
                        else:
                            dataset_names_all.extend(names)
                    gathered_batch_results_np["dataset_name"] = np.array(dataset_names_all, dtype=object)
                batch_result = gathered_batch_results_np
                
                # Split batch by dataset and write to corresponding zarr files (只在主进程)
                if self.is_main_process and self.mode != "vlm":
                    if save_zarr:
                        self.append_batch_to_zarr(zarr_root, zarr_datasets, batch_result)
                    self.update_results(results, batch_result)
                
                # 同步所有进程
                self.accelerator.wait_for_everyone()
        
        # 同步所有进程
        self.accelerator.wait_for_everyone()
        
        # Visualize attention weights for selected samples (只在主进程)
        if self.is_main_process:
            self.save_attention_samples(random_attention_samples, output_dir)
        
        # 保存 inference config 文件（只在主进程）
        if self.is_main_process:
            inference_config_path = output_dir / "inference_config.yaml"
            with open(inference_config_path, 'w') as f:
                OmegaConf.save(cfg, f)
            print(f"推理配置已保存至: {inference_config_path}")

            if self.mode == "vlm" and vlm_records is not None:
                self._save_vlm_json(output_dir, vlm_records)

            # 打印并保存每个数据集的统计信息
            print("\n开始计算统计信息和生成可视化...")
            if self.mode == "vlm":
                print("VLM 模式下跳过统计信息计算")
                self.accelerator.end_training()
                return
            self._save_vla_metrics(output_dir, results=results)
        
        self.accelerator.end_training()
    
    def save_attention_samples(self, samples, output_dir):
        """
        Visualize attention maps for random samples.
        
        Args:
            samples: List of dicts containing attention weights and metadata
            output_dir: Base output directory for saving visualizations
        """
        if not samples:
            print("No attention weights to visualize")
            return
        
        print(f"\nSaving attention weights for {len(samples)} random samples...")
        
        # Save each selected sample
        for sample_idx, sample_data in enumerate(samples):
            if sample_data.get('attn_weights') is None:
                continue
                
            metadata = sample_data.get("metadata", {})
            print(
                f"\nProcessing sample {sample_idx}: batch {metadata.get('batch_idx')}, "
                f"sample {metadata.get('sample_idx')}"
            )
            
            # Create output directory
            output_dir_path = os.path.join(
                str(output_dir),
                'attention_visualization',
                f"random_{sample_idx}"
            )
            os.makedirs(output_dir_path, exist_ok=True)
            output_path = os.path.join(output_dir_path, 'attention_and_inputs.npz')

            save_payload = {}
            for key, value in sample_data.items():
                if torch.is_tensor(value):
                    save_payload[key] = value.detach().cpu().float().numpy()
                else:
                    save_payload[key] = np.array(value, dtype=object if key == "metadata" else None)
            try:
                np.savez_compressed(output_path, **save_payload)
                print(f"  Saved to: {output_path}")
            except Exception as e:
                print(f"  Error saving attention data: {e}")

    def _update_random_attention_samples(
        self,
        samples,
        sample_count,
        total_seen,
        attn_weights,
        action_expert_attn_weights,
        inputs,
        batch_idx,
        dataset_name_list,
        dataset_local_idx_list,
    ):
        """
        Reservoir sampling to keep a random subset of attention maps.
        """
        if sample_count <= 0:
            return
        if attn_weights is not None: 
            print(f"attn_weights shape: {attn_weights.shape}")
        if action_expert_attn_weights is not None:
            print(f"action_expert_attn_weights shape: {action_expert_attn_weights.shape}")
        batch_size = attn_weights.shape[1]
        for sample_idx in range(batch_size):
            global_idx = total_seen + sample_idx
            metadata = {
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
            }
            if dataset_name_list is not None:
                metadata["dataset_name"] = dataset_name_list[sample_idx]
            if dataset_local_idx_list is not None:
                metadata["dataset_local_idx"] = dataset_local_idx_list[sample_idx]

            sample_data = {
                "attn_weights": attn_weights[:, sample_idx, :, :, :].float().cpu(),
                "metadata": metadata,
            }
            if action_expert_attn_weights is not None:
                sample_data["action_expert_attn_weights"] = (
                    action_expert_attn_weights[:, sample_idx, :, :, :].float().cpu()
                )
            sample_data.update({
                key: (value[sample_idx].detach().cpu() if torch.is_tensor(value) else value[sample_idx])
                for key, value in inputs.items()
            })

            if len(samples) < sample_count:
                samples.append(sample_data)
            else:
                j = random.randint(0, global_idx)
                if j < sample_count:
                    samples[j] = sample_data
    
    def append_batch_to_zarr(self, root, zarr_datasets, batch_result):
        """Append a batch of results to zarr file incrementally"""
        # Keys to save
        keys_to_save = [
            'pred_actions',
            'gt_actions',
            'actions_valid_mask',
            'dataset_local_idx',
            'dataset_name',
            'prefill_vlm_hidden_states',
            'prefill_image_hidden_states',
            'prefill_state_hidden_states',
            'prefill_text_hidden_states',
            'generated_hidden_states',
        ]
        
        for key in keys_to_save:
            if key not in batch_result:
                continue
            
            data = batch_result[key]
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            object_codec = None
            if data.dtype == object:
                # Use a variable-length UTF-8 codec for object/string arrays
                object_codec = numcodecs.VLenUTF8()

            if key not in zarr_datasets:
                # First time: create dataset with maxshape to allow appending
                zarr_datasets[key] = root.create_dataset(
                    key,
                    shape=(0,) + data.shape[1:],
                    maxshape=(None,) + data.shape[1:],
                    dtype=data.dtype,
                    chunks=True,
                    compression='gzip',
                    compression_opts=1,
                    object_codec=object_codec
                )
            
            # Directly append using zarr's append method
            dataset = zarr_datasets[key]
            dataset.append(data, axis=0)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("src/config/experiment")),
    config_name="inference_pretrain_legendvla"
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    inference = LegendVLAInference(cfg)
    inference.run()


if __name__ == "__main__":
    main()