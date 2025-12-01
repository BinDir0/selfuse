import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pathlib
import hydra
import os
import pickle
import wandb
import json

from src.workspace.base_workspace import BaseWorkspace
from src.model.action.vq_model import MotionVQModel
from src.utils.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainVQTokenizerWorkspace(BaseWorkspace): 
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        self.tokenizer_cfg = cfg.tokenizer
        self.shape_meta = cfg.shape_meta
        self.dataset = hydra.utils.instantiate(cfg.dataset)
        if cfg.training.normalizer_path is not None:
            normalizer = pickle.load(open(cfg.training.normalizer_path, 'rb'))
            print(f"Loaded normalizer from {cfg.training.normalizer_path}")
            self.dataset.set_normalizer(normalizer)
        else:
            print("Computing normalizer...")
            normalizer = self.dataset.get_normalizer()
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        pickle.dump(normalizer, open(normalizer_path, 'wb'))
        
        self.train_dataloader = DataLoader(self.dataset, collate_fn=self.dataset.get_collator(), **cfg.dataloader)

        self.val_dataset = self.dataset.get_validation_dataset()
        self.val_dataloader = DataLoader(self.val_dataset, collate_fn=self.val_dataset.get_collator(), **cfg.val_dataloader)

        if self.cfg.training.save_path is not None:
            self.save_path = pathlib.Path(self.cfg.training.save_path)
        else:
            self.save_path = pathlib.Path(self.output_dir, "tokenizer")
            
    def run(self):
        assert len(self.train_dataloader) > 0, "No data to train tokenizer"
        
        # Initialize wandb
        wandb.init(
            config=OmegaConf.to_container(self.cfg, resolve=True),
            dir=self.output_dir,
            **OmegaConf.to_container(self.cfg.logging, resolve=True)
        )
        
        self.tokenizer = {}
        if self.cfg.training.valid_tokenizer_path is None: 
            for key, tokenizer_cfg in self.tokenizer_cfg.items():
                if tokenizer_cfg.use_part:
                    self.tokenizer[key] = {
                        part: hydra.utils.instantiate(config) \
                            for part, config in tokenizer_cfg.items() if part != "use_part"
                    }
                else:
                    self.tokenizer[key] = hydra.utils.instantiate(tokenizer_cfg)
            self.train()
            self.save_vq_tokenizer(path=self.save_path)
        else:
            for key, tokenizer_cfg in self.tokenizer_cfg.items():
                if tokenizer_cfg.use_part:
                    self.tokenizer[key] = {
                        part: MotionVQModel.from_pretrained(
                            os.path.join(self.cfg.training.valid_tokenizer_path, key, part)
                        ) for part in tokenizer_cfg.keys() if part != "use_part"
                    }
                else:
                    self.tokenizer[key] = MotionVQModel.from_pretrained(
                        os.path.join(self.cfg.training.valid_tokenizer_path, key)
                    )
            self.validate()
        
        # Finish wandb
        if self.cfg.get('use_wandb', True):
            wandb.finish()

    def train(self): 
        device = torch.device(self.cfg.training.device)
        dict_apply(self.tokenizer, lambda x: x.to(device))
        dict_apply(self.tokenizer, lambda x: x.train())

        all_trainable_parameters = []
        for key, tokenizer in self.tokenizer.items():
            if isinstance(tokenizer, dict): # partial tokenizer
                for part, model in tokenizer.items():
                    all_trainable_parameters.extend(
                        self.get_grouped_parameters(model.parameters(), self.cfg.optimizer[key][part])
                    )
            else: # full tokenizer
                all_trainable_parameters.extend(
                    self.get_grouped_parameters(tokenizer.parameters(), self.cfg.optimizer[key])
                )
        self.optimizer = torch.optim.AdamW(all_trainable_parameters)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.training.num_epochs * len(self.train_dataloader)
        )

        global_step = 0
        for epoch in range(self.cfg.training.num_epochs):
            loss_dict = {}
            with tqdm(self.train_dataloader, desc=f"Training tokenizer epoch {epoch}", 
                mininterval=self.cfg.training.tqdm_interval_sec) as tepoch:
                for idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device) if torch.is_tensor(x) else x)
                    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
                    for key, tokenizer in self.tokenizer.items():
                        if isinstance(tokenizer, dict): # partial tokenizer
                            loss_dict[key] = {}
                            for part, model in tokenizer.items():
                                data = self.get_partial_motion(batch[key], part)
                                output = model(data)
                                total_loss = total_loss + output['loss']
                                loss_dict[key][part] = output
                        else: # full tokenizer
                            data = self.get_partial_motion(batch[key])
                            output = tokenizer(data)
                            total_loss = total_loss + output['loss']
                            loss_dict[key] = output
                    total_loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # Log to wandb
                    log_dict = self._flatten_loss_dict(loss_dict, prefix='train')
                    log_dict['train/total_loss'] = total_loss.item()
                    log_dict['train/epoch'] = epoch
                    log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                    wandb.log(log_dict, step=global_step)
                    
                    global_step += 1
                    tepoch.set_postfix(loss=total_loss.item())
                    if self.cfg.training.max_train_steps and idx >= self.cfg.training.max_train_steps-1:
                        break
            if (epoch + 1) % self.cfg.training.val_per_epoch == 0:
                avg_loss_dict = self.validate()
                if (epoch + 1) % self.cfg.training.ckpt_save_interval == 0:
                    self.save_vq_tokenizer(
                        path=os.path.join(self.output_dir, 'epoch_checkpoints', f'epoch={epoch}'), 
                        metric_dict=avg_loss_dict
                    )

    def validate(self):
        """
        Validate all tokenizers and log averaged metrics to wandb.
        """
        device = torch.device(self.cfg.training.device)
        dict_apply(self.tokenizer, lambda x: x.to(device))
        dict_apply(self.tokenizer, lambda x: x.eval())
        
        # Accumulate losses for all batches
        loss_accumulator = {}
        num_batches = 0
        
        with torch.no_grad():
            with tqdm(self.val_dataloader, desc="Validating tokenizer", 
                mininterval=self.cfg.training.tqdm_interval_sec) as tepoch:
                for idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device) if torch.is_tensor(x) else x)
                    total_loss = 0.0
                    
                    for key, tokenizer in self.tokenizer.items():
                        if key not in loss_accumulator:
                            loss_accumulator[key] = {}
                        if isinstance(tokenizer, dict):  # partial tokenizer
                            for part, model in tokenizer.items():
                                data = self.get_partial_motion(batch[key], part)
                                output = model(data)
                                total_loss += output['loss'].item()
                                
                                # Initialize accumulator for this part
                                if part not in loss_accumulator[key]:
                                    loss_accumulator[key][part] = {}
                                
                                # Accumulate each metric
                                for metric_name, metric_value in output.items():
                                    if metric_value.dim() > 0:
                                        continue # ignore tensor with dimension > 0
                                    if isinstance(metric_value, torch.Tensor):
                                        metric_value = metric_value.item()
                                    if metric_name not in loss_accumulator[key][part]:
                                        loss_accumulator[key][part][metric_name] = []
                                    loss_accumulator[key][part][metric_name].append(metric_value)
                        else:  # full tokenizer
                            data = self.get_partial_motion(batch[key])
                            output = tokenizer(data)
                            total_loss += output['loss'].item()
                            
                            # Accumulate each metric
                            for metric_name, metric_value in output.items():
                                if metric_value.dim() > 0:
                                    continue # ignore tensor with dimension > 0
                                if isinstance(metric_value, torch.Tensor):
                                    metric_value = metric_value.item()
                                if metric_name not in loss_accumulator[key]:
                                    loss_accumulator[key][metric_name] = []
                                loss_accumulator[key][metric_name].append(metric_value)
                    
                    num_batches += 1
                    tepoch.set_postfix(loss=total_loss)
                    
                    if self.cfg.training.max_val_steps and idx >= self.cfg.training.max_val_steps - 1:
                        break
        
        # Compute average losses
        avg_loss_dict = {}
        total_val_loss = 0.0
        
        for key, value in loss_accumulator.items():
            if isinstance(value, dict):
                # Check if this is a nested structure (partial tokenizer)
                if any(isinstance(v, dict) for v in value.values()):
                    # Nested: partial tokenizer
                    avg_loss_dict[key] = {}
                    for part, metrics in value.items():
                        avg_loss_dict[key][part] = {}
                        for metric_name, metric_values in metrics.items():
                            avg_value = np.mean(metric_values)
                            avg_loss_dict[key][part][metric_name] = avg_value
                            if metric_name == 'loss':
                                total_val_loss += avg_value
                else:
                    # Single level: full tokenizer
                    avg_loss_dict[key] = {}
                    for metric_name, metric_values in value.items():
                        avg_value = np.mean(metric_values)
                        avg_loss_dict[key][metric_name] = avg_value
                        if metric_name == 'loss':
                            total_val_loss += avg_value
        
        # Log to wandb with 'valid' prefix
        log_dict = self._flatten_loss_dict(avg_loss_dict, prefix='valid')
        log_dict['valid/total_loss'] = total_val_loss
        wandb.log(log_dict)
        
        # Print summary
        print("\n" + "="*80)
        print("Validation Summary")
        print("="*80)
        for key, formatted_str in self._format_loss_dict(avg_loss_dict).items():
            print(f"{key}: {formatted_str}")
        print(f"Total validation loss: {total_val_loss:.6f}")
        print("="*80 + "\n")
        
        dict_apply(self.tokenizer, lambda x: x.train())

        return avg_loss_dict

    def save_vq_tokenizer(self, path = None, metric_dict = None):
        if path is None:
            path = self.save_path
        
        for key, tokenizer in self.tokenizer.items():
            if isinstance(tokenizer, dict): # partial tokenizer
                for part in tokenizer.keys():
                    save_path = os.path.join(path, key, part)
                    os.makedirs(save_path, exist_ok=True)
                    if metric_dict is not None:
                        with open(os.path.join(save_path, 'metrics.json'), 'w', encoding='utf-8') as f:
                            json.dump(metric_dict[key][part], f, ensure_ascii=False, indent=2)
                    # safe_serialization=False is used, because we share the codebook between different Layers    
                    tokenizer[part].save_pretrained(save_path, safe_serialization=False) 
            else: # full tokenizer
                save_path = os.path.join(path, key)
                os.makedirs(save_path, exist_ok=True)
                if metric_dict is not None:
                    with open(os.path.join(save_path, 'metrics.json'), 'w', encoding='utf-8') as f:
                        json.dump(metric_dict[key], f, ensure_ascii=False, indent=2)
                # safe_serialization=False is used, because we share the codebook between different Layers
                tokenizer.save_pretrained(save_path, safe_serialization=False)

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
    
    def get_partial_motion(self, data, part = None): 
        '''
        Args: 
            data: torch.Tensor with layout [left_trans(3), right_trans(3), left_rot(6), right_rot(6), hand_left(15), hand_right(15)]
            part: str, 'wrist', 'hand', or None(both)
        Returns:
            partial_data: torch.Tensor with partial motion, stacked along batch dimension (dim=0)
        '''
        if part is None : 
            return data.clone() if torch.is_tensor(data) else data.copy() # copy to avoid in-place modification
        
        # Data layout: [left_trans(3), right_trans(3), left_rot(6), right_rot(6), hand_left(15), hand_right(15)]
        if part == 'wrist': 
            # Reconstruct: left_wrist = [left_trans + left_rot], right_wrist = [right_trans + right_rot]
            left_wrist = torch.cat([data[..., 0:3], data[..., 6:12]], dim=-1)  # [..., 9]
            right_wrist = torch.cat([data[..., 3:6], data[..., 12:18]], dim=-1)  # [..., 9]
            return torch.cat([left_wrist, right_wrist], dim=0)  # Stack left and right along batch dim
        elif part == 'hand':
            # Extract hand data directly (already in correct layout)
            left_hand = data[..., 18:33]  # [..., 15]
            right_hand = data[..., 33:48]  # [..., 15]
            return torch.cat([left_hand, right_hand], dim=0)  # Stack left and right along batch dim
        else:   
            raise ValueError(f"Invalid part: {part}")

    def _format_loss_dict(self, loss_dict):
        """
        Format loss_dict for printing.
        
        Args:
            loss_dict: Nested dictionary containing loss values
            
        Returns:
            Dictionary with formatted strings for each key/part
        """
        formatted = {}
        
        for key, value in loss_dict.items():
            if isinstance(value, dict):
                # Check if this is a nested structure (partial tokenizer)
                if any(isinstance(v, dict) for v in value.values()):
                    # Nested: e.g., {'wrist': {...}, 'hand': {...}}
                    for part, metrics in value.items():
                        metric_strs = [f"{k}={v:.6f}" for k, v in metrics.items()]
                        formatted[f"{key}/{part}"] = ", ".join(metric_strs)
                else:
                    # Single level: e.g., {'loss': 0.5, 'recon_loss': 0.3}
                    metric_strs = [f"{k}={v:.6f}" for k, v in value.items()]
                    formatted[key] = ", ".join(metric_strs)
        
        return formatted
    
    def _flatten_loss_dict(self, loss_dict, prefix='train'):
        '''
        Flatten nested loss_dict for wandb logging.
        
        Args:
            loss_dict: Nested dictionary containing loss values
                Example: {
                    'states': {
                        'wrist': {'loss': 0.5, 'recon_loss': 0.3, 'commit_loss': 0.2},
                        'hand': {'loss': 0.4, ...}
                    },
                    'actions': {...}
                }
            prefix: Prefix for all keys (e.g., 'train' or 'val')
            
        Returns:
            Flattened dictionary for wandb.log()
                Example: {
                    'train/states/wrist/loss': 0.5,
                    'train/states/wrist/recon_loss': 0.3,
                    'train/states/hand/loss': 0.4,
                    ...
                }
        '''
        flat_dict = {}
        
        for key, value in loss_dict.items():
            if isinstance(value, dict):
                # Check if this is a nested structure (partial tokenizer)
                if all(isinstance(v, dict) for v in value.values()):
                    # Nested: e.g., {'wrist': {...}, 'hand': {...}}
                    for part, metrics in value.items():
                        for metric_name, metric_value in metrics.items():
                            # Extract scalar value (handle torch.Tensor)
                            if isinstance(metric_value, torch.Tensor):
                                if metric_value.dim() > 0:
                                    continue # ignore tensor with dimension > 0
                                metric_value = metric_value.item()
                            elif isinstance(metric_value, np.ndarray):
                                if metric_value.ndim > 0:
                                    continue # ignore array with dimension > 0
                                metric_value = metric_value.item()
                            flat_dict[f'{prefix}/{key}/{part}/{metric_name}'] = metric_value
                else:
                    # Single level: e.g., {'loss': 0.5, 'recon_loss': 0.3}
                    for metric_name, metric_value in value.items():
                        if isinstance(metric_value, torch.Tensor):
                            if metric_value.dim() > 0:
                                continue # ignore tensor with dimension > 0
                            metric_value = metric_value.item()
                        elif isinstance(metric_value, np.ndarray):
                            if metric_value.ndim > 0:
                                continue # ignore array with dimension > 0
                            metric_value = metric_value.item()
                        flat_dict[f'{prefix}/{key}/{metric_name}'] = metric_value
            else:
                # Direct value
                if isinstance(value, torch.Tensor):
                    if value.dim() > 0:
                        continue # ignore tensor with dimension > 0
                    value = value.item()
                elif isinstance(value, np.ndarray):
                    if value.ndim > 0:
                        continue # ignore array with dimension > 0
                    value = value.item()
                flat_dict[f'{prefix}/{key}'] = value
        
        return flat_dict
        