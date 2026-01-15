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
import random
import warnings

from src.workspace.base_workspace import BaseWorkspace
from src.model.action.vq_model import MotionVQModel
from src.utils.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainVQTokenizerWorkspace(BaseWorkspace): 
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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
            # Direct instantiation: tokenizer_cfg is now a dict with 'wrist' and 'hand' keys
            for part, model_cfg in self.tokenizer_cfg.items():
                self.tokenizer[part] = hydra.utils.instantiate(model_cfg)
            self.train()
            self.save_vq_tokenizer(path=self.save_path)
        else:
            # Load from pretrained
            for part in self.tokenizer_cfg.keys():
                self.tokenizer[part] = MotionVQModel.from_pretrained(
                    os.path.join(self.cfg.training.valid_tokenizer_path, part)
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
        for part, model in self.tokenizer.items():
            all_trainable_parameters.extend(
                self.get_grouped_parameters(model.parameters(), self.cfg.optimizer[part])
            )
        self.optimizer = torch.optim.AdamW(all_trainable_parameters)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.training.num_epochs * len(self.train_dataloader)
        )

        if self.cfg.training.clipping.enabled:
            max_grad_norm = self.cfg.training.clipping.max_grad_norm
        else:
            max_grad_norm = float('inf')

        global_step = 0
        for epoch in range(self.cfg.training.num_epochs):
            loss_dict = {}
            with tqdm(self.train_dataloader, desc=f"Training tokenizer epoch {epoch}", 
                mininterval=self.cfg.training.tqdm_interval_sec) as tepoch:
                for idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device) if torch.is_tensor(x) else x)
                    for key, value in batch.items(): 
                        if not isinstance(value, torch.Tensor):
                            continue
                        min_value, max_value = value.min().item(), value.max().item()
                        if min_value < -1.0 - 1e-5 or max_value > 1.0 + 1e-5:
                            print(f"{key}: {min_value}, {max_value}")
                            warnings.warn(f"Value out of range for {key}")
                    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
                    loss_dict = {}
                    for part, model in self.tokenizer.items():
                        # Get state and action data for this part (wrist or hand)
                        state_data, action_data = self.get_partial_motion(batch, part)
                        output = model(state_data, action_data)
                        total_loss = total_loss + output['loss']
                        loss_dict[part] = output
                    total_loss.backward()
                    
                    for part, model in self.tokenizer.items():
                        # Collect parameters for this specific model
                        params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                        if len(params) > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                            loss_dict[part]['grad_norm'] = grad_norm.item()
                        else:
                            loss_dict[part]['grad_norm'] = 0.0
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # Log to wandb
                    log_dict = self._flatten_loss_dict(loss_dict, prefix='train', log_perplexity=(idx % self.cfg.training.log_perplexity_interval == 0))
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
                    
                    for part, model in self.tokenizer.items():
                        if part not in loss_accumulator:
                            loss_accumulator[part] = {}
                        
                        # Get state and action data for this part (wrist or hand)
                        state_data, action_data = self.get_partial_motion(batch, part)
                        output = model(state_data, action_data)
                        total_loss += output['loss'].item()
                        
                        # Accumulate each metric
                        for metric_name, metric_value in output.items():
                            if metric_name not in loss_accumulator[part]:
                                loss_accumulator[part][metric_name] = []
                            loss_accumulator[part][metric_name].append(metric_value)
                    
                    num_batches += 1
                    tepoch.set_postfix(loss=total_loss)
                    
                    if self.cfg.training.max_val_steps and idx >= self.cfg.training.max_val_steps - 1:
                        break
        
        # Compute average losses
        avg_loss_dict = {}
        total_val_loss = 0.0
        
        for part, metrics in loss_accumulator.items():
            avg_loss_dict[part] = {}
            for metric_name, metric_values in metrics.items():
                avg_value = torch.mean(torch.stack(metric_values), dim=0)
                avg_loss_dict[part][metric_name] = avg_value
                if metric_name == 'loss':
                    total_val_loss += avg_value.item()
        
        # Log to wandb with 'valid' prefix
        log_dict = self._flatten_loss_dict(avg_loss_dict, prefix='valid', log_perplexity=True)
        log_dict['valid/total_loss'] = total_val_loss
        wandb.log(log_dict)
        
        # Print summary
        print("\n" + "="*80)
        print("Validation Summary")
        print("="*80)
        for key, value in log_dict.items():
            print(f"{key}: {value}")
        print(f"Total validation loss: {total_val_loss:.6f}")
        print("="*80 + "\n")
        
        dict_apply(self.tokenizer, lambda x: x.train())

        return log_dict

    def save_vq_tokenizer(self, path = None, metric_dict = None):
        if path is None:
            path = self.save_path
        
        for part, model in self.tokenizer.items():
            save_path = os.path.join(path, part)
            os.makedirs(save_path, exist_ok=True)
            # safe_serialization=False is used, because we share the codebook between different Layers    
            model.save_pretrained(save_path, safe_serialization=False)

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
    
    def _extract_wrist_bimanual(self, data):
        """
        Extract wrist data with left and right concatenated along feature dimension.
        
        Args:
            data: [B, T, D_total] tensor with layout:
                [left_trans(3), right_trans(3), left_rot(6), right_rot(6), hand_left(15), hand_right(15)]
        
        Returns:
            wrist_bimanual: [B, T, 2*wrist_dim] = [left_wrist, right_wrist] concatenated
        """
        left_wrist = torch.cat([data[..., 0:3], data[..., 6:12]], dim=-1)  # [B, T, 9]
        right_wrist = torch.cat([data[..., 3:6], data[..., 12:18]], dim=-1)  # [B, T, 9]
        return torch.cat([left_wrist, right_wrist], dim=-1)  # [B, T, 18]
    
    def get_partial_motion(self, batch, part):
        """
        Extract partial motion data for training VQ tokenizer.
        
        Args:
            batch: Batch dictionary containing 'states' and 'actions' keys
            part: str, 'wrist' or 'hand'
        
        Returns:
            Tuple of (state_data, action_data)
            - state_data: [B, T_state, D_part] where D_part = 2*wrist_dim or 2*hand_dim
            - action_data: [B, T_action, D_part]
            Both state and action data are extracted from batch and concatenated along feature dimension
        """
        # Get state and action data from batch
        state_data = batch['states']
        action_data = batch['actions']
        
        # Extract partial data based on part type
        if part == 'wrist':
            state_partial = self._extract_wrist_bimanual(state_data)
            action_partial = self._extract_wrist_bimanual(action_data)
        elif part == 'hand':
            state_partial = state_data[..., 18:]
            action_partial = action_data[..., 18:]
        else:
            raise ValueError(f"Invalid part: {part}. Must be 'wrist' or 'hand'")
        
        return state_partial, action_partial

    def _flatten_loss_dict(self, loss_dict, prefix='train', log_perplexity=False):
        '''
        Flatten nested loss_dict for wandb logging.
        
        Args:
            loss_dict: Nested dictionary containing loss values
                Example: {
                    'wrist': {'loss': 0.5, 'recon_loss': 0.3, 'commit_loss': 0.2},
                    'hand': {'loss': 0.4, ...}
                }
            prefix: Prefix for all keys (e.g., 'train' or 'val')
            
        Returns:
            Flattened dictionary for wandb.log()
                Example: {
                    'train/wrist/loss': 0.5,
                    'train/wrist/recon_loss': 0.3,
                    'train/hand/loss': 0.4,
                    ...
                }
        '''
        flat_dict = {}
        
        def log_metric(flat_dict, key, value, prefix): 
            if isinstance(value, torch.Tensor) and value.dim() > 0: 
                if "perplexity" in key and log_perplexity:
                    if value.dim() == 1: 
                        for l in range(value.shape[0]):
                            flat_dict[f'{prefix}/{key}/Layer: {l}'] = value[l].item()
                    elif value.dim() == 2: 
                        for g in range(value.shape[0]):
                            for l in range(value.shape[1]):
                                flat_dict[f'{prefix}/{key}/Group: {g}/Layer: {l}'] = value[g, l].item()
                value = value.mean().item()
            elif isinstance(value, torch.Tensor): 
                value = value.item()
            flat_dict[f'{prefix}/{key}'] = value

        for part, metrics in loss_dict.items():
            for metric_name, metric_value in metrics.items():
                log_metric(flat_dict, metric_name, metric_value, f'{prefix}/{part}')
        
        return flat_dict
        