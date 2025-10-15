import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pathlib
import hydra
import os
import pickle
from typing import Dict
import wandb

from src.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainMultipleGRVQTokenizerWorkspace(BaseWorkspace):
    """
    Workspace for training multiple GRVQ tokenizers simultaneously.
    
    This approach maximizes GPU utilization by training multiple tokenizers
    on the same batch, amortizing the I/O cost across multiple models.
    """
    
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        
        # Initialize dataset and dataloader
        self.dataset = hydra.utils.instantiate(cfg.dataset)
        
        # Setup normalizer
        if cfg.training.normalizer_path is not None:
            normalizer = pickle.load(open(cfg.training.normalizer_path, 'rb'))
            self.dataset.set_normalizer(normalizer)
        else:
            normalizer = self.dataset.get_normalizer()
        
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        pickle.dump(normalizer, open(normalizer_path, 'wb'))
        
        self.dataloader = DataLoader(
            self.dataset, 
            collate_fn=self.dataset.get_collator(), 
            **cfg.dataloader
        )
        
        # Initialize multiple tokenizers
        self.tokenizers: Dict[str, nn.Module] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        
        print("Initializing tokenizers...")
        for name, tokenizer_cfg in cfg.tokenizers.items():
            print(f"  - {name}: {tokenizer_cfg.args.use_part}")
            tokenizer = hydra.utils.instantiate(tokenizer_cfg)
            self.tokenizers[name] = tokenizer.cuda()
            
            # Setup optimizer for this tokenizer
            optimizer_cfg = cfg.optimizers[name]
            self.optimizers[name] = torch.optim.AdamW(
                tokenizer.parameters(),
                lr=optimizer_cfg.lr,
                betas=optimizer_cfg.betas,
                weight_decay=optimizer_cfg.weight_decay
            )
        
        # Save paths
        if cfg.training.save_path is not None:
            self.save_path = pathlib.Path(cfg.training.save_path)
        else:
            self.save_path = pathlib.Path(self.output_dir, "tokenizers")
        
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def run(self):
        cfg = self.cfg
        
        print(f"Starting training for {cfg.training.num_epochs} epochs...")
        print(f"Training {len(self.tokenizers)} tokenizers simultaneously:")
        for name in self.tokenizers.keys():
            print(f"  - {name}")
        
        # Training loop
        for epoch in range(cfg.training.num_epochs):
            self.epoch = epoch
            epoch_losses = {name: [] for name in self.tokenizers.keys()}
            
            with tqdm(
                self.dataloader, 
                desc=f"Epoch {epoch}/{cfg.training.num_epochs}",
                mininterval=cfg.training.tqdm_interval_sec
            ) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    step_losses = {}
                    
                    # Train all tokenizers on the same batch
                    for name, tokenizer in self.tokenizers.items():
                        # Forward pass
                        loss_dict = tokenizer(
                            motion=batch["motion"].cuda(),
                            wrist_motion=batch["wrist"].cuda(),
                            hand_motion=batch["hand"].cuda()
                        )
                        
                        # Backward pass
                        self.optimizers[name].zero_grad()
                        loss_dict["loss"].backward()
                        self.optimizers[name].step()
                        
                        # Log losses
                        step_losses[f"{name}/loss"] = loss_dict["loss"].item()
                        step_losses[f"{name}/loss_recons"] = loss_dict["loss_recons"].item()
                        step_losses[f"{name}/loss_commit"] = loss_dict["loss_commit"].item()
                        
                        epoch_losses[name].append(loss_dict["loss"].item())
                    
                    # Update progress bar
                    pbar.set_postfix({
                        **{f"{k[:5]}": f"{v:.4f}" for k, v in step_losses.items()}
                    })
                    
                    # Logging
                    if self.global_step % cfg.training.log_interval == 0:
                        # Log to wandb or other tracking systems
                        step_losses["epoch"] = epoch
                        step_losses["global_step"] = self.global_step
                        # wandb.log(step_losses)  # Uncomment if using wandb
                    
                    # Save checkpoint
                    if self.global_step % cfg.training.save_interval == 0 and self.global_step > 0:
                        self.save_tokenizers()
                    
                    self.global_step += 1
                    
                    # Early stopping for debugging
                    if cfg.training.max_corpus_size and batch_idx >= cfg.training.max_corpus_size:
                        break
            
            # Epoch summary
            print(f"\nEpoch {epoch} Summary:")
            for name, losses in epoch_losses.items():
                avg_loss = np.mean(losses)
                print(f"  {name}: avg_loss={avg_loss:.6f}")
            
            # Save after each epoch
            self.save_tokenizers()
        
        print(f"\nTraining complete! Models saved to {self.save_path}")
        
        # Validation
        if cfg.training.valid_tokenizer_path is None:
            print("\nRunning validation...")
            self.validate()
    
    def save_tokenizers(self):
        """Save all tokenizers."""
        for name, tokenizer in self.tokenizers.items():
            save_dir = self.save_path / name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save using HuggingFace's save_pretrained if available
            if hasattr(tokenizer, 'save_pretrained'):
                tokenizer.save_pretrained(str(save_dir))
            else:
                torch.save(tokenizer.state_dict(), save_dir / "model.pth")
        
        print(f"Tokenizers saved to {self.save_path}")
    
    def validate(self):
        """Validate all tokenizers."""
        print("\nValidating tokenizers...")
        
        for tokenizer in self.tokenizers.values():
            tokenizer.eval()
        
        val_losses = {name: [] for name in self.tokenizers.keys()}
        
        with torch.no_grad():
            with tqdm(
                self.dataloader,
                desc="Validation",
                total=min(len(self.dataloader), self.cfg.training.max_val_steps or float('inf'))
            ) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    step_losses = {}
                    
                    for name, tokenizer in self.tokenizers.items():
                        loss_dict = tokenizer(
                            motion=batch["motion"].cuda(),
                            wrist_motion=batch["wrist"].cuda(),
                            hand_motion=batch["hand"].cuda()
                        )
                        
                        val_losses[name].append(loss_dict["loss"].item())
                        step_losses[f"{name[:5]}"] = loss_dict["loss"].item()
                    
                    pbar.set_postfix(step_losses)
                    
                    if self.cfg.training.max_val_steps and batch_idx >= self.cfg.training.max_val_steps:
                        break
        
        print("\nValidation Results:")
        for name, losses in val_losses.items():
            avg_loss = np.mean(losses)
            std_loss = np.std(losses)
            print(f"  {name}:")
            print(f"    avg_loss: {avg_loss:.6f}")
            print(f"    std_loss: {std_loss:.6f}")
        
        # Set back to training mode
        for tokenizer in self.tokenizers.values():
            tokenizer.train()


