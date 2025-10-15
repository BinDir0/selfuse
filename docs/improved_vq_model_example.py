"""
Improved VQ Model Implementation with Proper PreTrainedModel Integration

This shows how to properly use PreTrainedModel with custom configuration.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import Optional, Dict

from src.model.action.vq_config import VQModelConfig, PartialMotionVQModelConfig
from src.model.action.encdec import Encoder, Decoder
from vector_quantize_pytorch import GroupedResidualVQ, ResidualVQ


class ImprovedPartialMotionVQModel(PreTrainedModel):
    """
    Improved version of PartialMotionVQModel with proper config integration.
    
    Key improvements:
    1. Uses custom config class (PartialMotionVQModelConfig)
    2. All hyperparameters stored in config
    3. Fully compatible with save_pretrained()/from_pretrained()
    4. Can be pushed to HuggingFace Hub
    
    Example:
        >>> # Create from config
        >>> config = PartialMotionVQModelConfig(
        ...     use_part="wrist",
        ...     codebook_dim=256,
        ...     nb_code=1024
        ... )
        >>> model = ImprovedPartialMotionVQModel(config)
        >>> 
        >>> # Save (config is automatically saved too)
        >>> model.save_pretrained("/path/to/save")
        >>> 
        >>> # Load (config is automatically loaded)
        >>> model = ImprovedPartialMotionVQModel.from_pretrained("/path/to/save")
    """
    
    config_class = PartialMotionVQModelConfig
    
    def __init__(self, config: PartialMotionVQModelConfig):
        """
        Initialize the model from configuration.
        
        Args:
            config: Model configuration containing all hyperparameters
        """
        super().__init__(config)
        
        # Store config
        self.config = config
        self.use_part = config.use_part
        
        # Determine motion dimension based on use_part
        if config.use_part == "wrist":
            motion_dim = config.wrist_dim
        elif config.use_part == "hand":
            motion_dim = config.hand_dim
        elif config.use_part is None:
            motion_dim = config.motion_dim
        else:
            raise ValueError(f"Unknown use_part: {config.use_part}")
        
        # Build encoder
        self.encoder = Encoder(
            input_emb_width=motion_dim,
            output_emb_width=config.output_emb_width,
            down_t=config.down_t,
            stride_t=config.stride_t,
            width=config.width,
            depth=config.depth,
            dilation_growth_rate=config.dilation_growth_rate,
            activation=config.activate,
            norm=config.norm,
            num_conv_layers=config.num_conv_layers
        )
        
        # Build decoder
        self.decoder = Decoder(
            input_emb_width=motion_dim,
            output_emb_width=config.output_emb_width,
            down_t=config.down_t,
            stride_t=config.stride_t,
            width=config.width,
            depth=config.depth,
            dilation_growth_rate=config.dilation_growth_rate,
            activation=config.activate,
            norm=config.norm,
            num_conv_layers=config.num_conv_layers
        )
        
        # Build quantizer
        if config.quantizer_name == "group_residualvq":
            self.quantizer = GroupedResidualVQ(
                codebook_size=config.nb_code,
                dim=config.codebook_dim,
                num_quantizers=config.num_quantizers,
                groups=config.num_groups,
                shared_codebook=config.shared_codebook
            )
        elif config.quantizer_name == "residualvq":
            self.quantizer = ResidualVQ(
                codebook_size=config.nb_code,
                dim=config.codebook_dim,
                num_quantizers=config.num_quantizers,
                shared_codebook=config.shared_codebook
            )
        else:
            raise ValueError(f"Unknown quantizer: {config.quantizer_name}")
        
        # Loss function
        if config.recons_loss == "l2":
            self.loss_fn = nn.MSELoss()
        elif config.recons_loss == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss: {config.recons_loss}")
        
        self.commit_weight = config.commit_weight
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode motion to discrete codes."""
        x_enc = self.encoder(x.permute(0, 2, 1))  # (B, T, D) -> (B, D, T)
        x_enc = x_enc.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        _, indices, _ = self.quantizer(x_enc)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode discrete codes to motion."""
        x_quant = self.quantizer.get_output_from_indices(indices)
        x_dec = self.decoder(x_quant.permute(0, 2, 1))  # (B, T, D) -> (B, D, T)
        x_out = x_dec.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x_out
    
    def forward(
        self,
        motion: torch.Tensor,
        wrist_motion: Optional[torch.Tensor] = None,
        hand_motion: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with reconstruction and commitment loss.
        
        Args:
            motion: Full motion tensor (B, T, D)
            wrist_motion: Wrist motion tensor (B, T, wrist_dim)
            hand_motion: Hand motion tensor (B, T, hand_dim)
            
        Returns:
            Dictionary with loss, reconstructed motion, etc.
        """
        # Select which motion to use
        if self.use_part == "wrist":
            x_motion = wrist_motion
        elif self.use_part == "hand":
            x_motion = hand_motion
        elif self.use_part is None:
            x_motion = motion
        else:
            raise ValueError(f"Unknown use_part: {self.use_part}")
        
        # Encode
        x_enc = self.encoder(x_motion.permute(0, 2, 1).float())
        x_enc = x_enc.permute(0, 2, 1)
        
        # Quantize
        x_quant, indices, commit_loss = self.quantizer(x_enc)
        
        # Decode
        x_dec = self.decoder(x_quant.permute(0, 2, 1))
        pred_motion = x_dec.permute(0, 2, 1)
        
        # Compute losses
        recon_loss = self.loss_fn(pred_motion, x_motion.float())
        total_loss = recon_loss + self.commit_weight * commit_loss.mean()
        
        return {
            'loss': total_loss,
            'loss_recons': recon_loss,
            'loss_commit': commit_loss.mean(),
            'pred_motion': pred_motion,
            'indices': indices,
        }


# ============= Usage Examples =============

def example_training():
    """Example: Training a VQ model"""
    from torch.utils.data import DataLoader
    
    # 1. Create configuration
    config = PartialMotionVQModelConfig(
        use_part="wrist",
        wrist_dim=18,
        codebook_dim=256,
        nb_code=1024,
        num_quantizers=8,
        num_groups=1,
        width=256,
        depth=3
    )
    
    # 2. Create model
    model = ImprovedPartialMotionVQModel(config).cuda()
    
    # 3. Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-4
    )
    
    # 4. Training loop
    for epoch in range(10):
        for batch in dataloader:
            # Forward
            outputs = model(
                motion=batch['motion'].cuda(),
                wrist_motion=batch['wrist'].cuda(),
                hand_motion=batch['hand'].cuda()
            )
            
            # Backward
            loss = outputs['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Loss: {loss.item():.4f}")
        
        # Save checkpoint (config is automatically saved)
        model.save_pretrained(f"checkpoints/epoch_{epoch}")
    
    print("Training complete!")


def example_loading():
    """Example: Loading a pretrained model"""
    
    # Load model (config is automatically loaded)
    model = ImprovedPartialMotionVQModel.from_pretrained(
        "checkpoints/epoch_9"
    )
    model.eval()
    
    # Use for inference
    with torch.no_grad():
        outputs = model(
            motion=test_motion,
            wrist_motion=test_wrist,
            hand_motion=test_hand
        )
    
    reconstructed = outputs['pred_motion']
    print(f"Reconstruction error: {outputs['loss_recons'].item():.6f}")


def example_hub_integration():
    """Example: Push to/pull from HuggingFace Hub"""
    
    # Push to Hub
    model = ImprovedPartialMotionVQModel.from_pretrained("checkpoints/epoch_9")
    model.push_to_hub("your-username/vq-wrist-tokenizer")
    
    # Pull from Hub
    model = ImprovedPartialMotionVQModel.from_pretrained(
        "your-username/vq-wrist-tokenizer"
    )


def example_config_modification():
    """Example: Modify config and create new model"""
    
    # Load config
    config = PartialMotionVQModelConfig.from_pretrained("checkpoints/epoch_9")
    
    # Modify config
    config.width = 512  # Increase model width
    config.nb_code = 2048  # Increase codebook size
    
    # Create new model with modified config
    new_model = ImprovedPartialMotionVQModel(config)
    
    # Optionally load weights (with strict=False for shape mismatches)
    old_state = ImprovedPartialMotionVQModel.from_pretrained(
        "checkpoints/epoch_9"
    ).state_dict()
    new_model.load_state_dict(old_state, strict=False)


def example_migration_from_old_model():
    """Example: Migrate from old implementation to new one"""
    
    # Load old model checkpoint (state_dict only)
    old_checkpoint = torch.load("old_checkpoints/model.pth")
    
    # Create config manually
    config = PartialMotionVQModelConfig(
        use_part="wrist",
        wrist_dim=18,
        codebook_dim=256,
        nb_code=1024,
        # ... other parameters from your yaml config
    )
    
    # Create new model
    new_model = ImprovedPartialMotionVQModel(config)
    
    # Load old weights
    new_model.load_state_dict(old_checkpoint, strict=False)
    
    # Save in new format (with config)
    new_model.save_pretrained("new_checkpoints/migrated_model")
    
    print("Migration complete!")


if __name__ == "__main__":
    # Run examples
    print("Example 1: Training")
    example_training()
    
    print("\nExample 2: Loading")
    example_loading()
    
    print("\nExample 3: Hub Integration")
    example_hub_integration()

