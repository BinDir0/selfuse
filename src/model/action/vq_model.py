# VQ-VAE Model for Action Tokenization
# Adapted from Being-H0/beingvla/models/motion/m2m/tokenizer/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
import logging
from vector_quantize_pytorch import GroupedResidualVQ, ResidualVQ
logger = logging.getLogger(__name__)


# ============ Being-H0's Encoder/Decoder Implementation ============

class Swish(nn.Module):
    """Swish activation function (x * sigmoid(x))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    """Residual 1D convolutional block with optional normalization."""
    NORM_MAP = {
        "LN": nn.LayerNorm,
        "GN": lambda n_in: nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True),
        "BN": lambda n_in: nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True),
    }
    
    ACTIVATION_MAP = {
        "relu": nn.ReLU,
        "silu": Swish,
        "gelu": nn.GELU
    }
    
    def __init__(self, 
                 n_in: int, 
                 n_state: int, 
                 dilation: int = 1, 
                 activation: Literal['relu', 'silu', 'gelu'] = 'silu',
                 norm: Optional[Literal['LN', 'GN', 'BN']] = None):
        super().__init__()
  
        self.norm = norm
        self.norm1 = self._create_norm_layer(norm, n_in)
        self.norm2 = self._create_norm_layer(norm, n_in)

        padding = dilation
 
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)

        self.activation1 = self.ACTIVATION_MAP[activation]()
        self.activation2 = self.ACTIVATION_MAP[activation]()
      
    def _create_norm_layer(self, norm, n_in):
        return self.NORM_MAP.get(norm, nn.Identity)(n_in)
 
    def forward(self, x):
        x_orig = x

        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)  
        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)
        x = self.conv2(x)

        return x + x_orig


class Resnet1D(nn.Module):
    """1D ResNet with configurable dilation rates."""
    def __init__(self, 
                 n_in: int, 
                 n_depth: int, 
                 dilation_growth_rate: int = 1, 
                 reverse_dilation: bool = True,
                 activation: str = 'silu',
                 norm: Optional[str] = None):
        super().__init__()
        
        blocks = [
            ResConv1DBlock(
                n_in, n_in,
                dilation=dilation_growth_rate ** depth,
                activation=activation,
                norm=norm
            ) for depth in range(n_depth)
        ]
        
        self.model = nn.Sequential(*(blocks[::-1] if reverse_dilation else blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Encoder(nn.Module):
    """1D convolutional encoder with downsampling."""
    
    def __init__(self,
                 input_emb_width: int = 3,
                 output_emb_width: int = 512,
                 down_t: int = 3,
                 stride_t: int = 2,
                 width: int = 512,
                 depth: int = 3,
                 dilation_growth_rate: int = 3,
                 activation: str = 'relu',
                 norm: Optional[str] = None,
                 num_conv_layers = 1):
        super().__init__()
        
        
        filter_t, pad_t = stride_t * 2, stride_t // 2

        blocks = []
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(num_conv_layers-1):
            blocks.append(nn.Conv1d(width, width, 3, 1, 1))
            blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None,
                 num_conv_layers=1):
        super().__init__()

        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for _ in range(num_conv_layers-1):
            blocks.append(nn.Conv1d(width, width, 3, 1, 1))
            blocks.append(nn.ReLU())

        for _ in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    

class ActionVQModel(nn.Module):
    """
    VQ-VAE model for action tokenization.
    
    Similar to ManoVQModel but adapted for general action sequences.
    Supports both RVQ and GRVQ (Grouped Residual VQ).
    """
    
    def __init__(
        self,
        action_dim: int,
        time_horizon: int,
        codebook_size: int = 512,
        codebook_dim: int = 256,
        num_quantizers: int = 4,
        width: int = 512,
        depth: int = 3,
        down_t: int = 2,
        stride_t: int = 2,
        dilation_growth_rate: int = 3,  # Being-H0 uses 3
        activation: str = "relu",
        norm: Optional[str] = None,  # 'LN', 'GN', 'BN' or None
        shared_codebook: bool = False,
        quantizer_type: str = "residualvq",  # "residualvq" or "group_residualvq"
        num_groups: int = 2,  # for GRVQ
        num_conv_layers: int = 1,  # number of conv layers
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_quantizers = num_quantizers
        self.down_t = down_t
        self.quantizer_type = quantizer_type
        self.num_groups = num_groups
        
        # Calculate padding length (BeingH0 style)
        self.downsample_factor = 2 ** down_t
        self.res_len = self.downsample_factor - (time_horizon % self.downsample_factor)
        if self.res_len == self.downsample_factor:
            self.res_len = 0  # No padding needed if perfectly divisible
        self.padded_time_horizon = time_horizon + self.res_len
        
        # Encoder (Being-H0 style parameters)
        self.encoder = Encoder(
            input_emb_width=action_dim,         # action dimension
            output_emb_width=codebook_dim,          # latent dimension
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            num_conv_layers=num_conv_layers,
        )
        
        # Quantizer - support both RVQ and GRVQ
        if quantizer_type == "group_residualvq":
            self.quantizer = GroupedResidualVQ(
                dim = codebook_dim,  # encoder output dimension
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                groups=num_groups,
                shared_codebook=shared_codebook,
            )
            logger.info(f"Using GroupedResidualVQ with {num_groups} groups")
        else:  # default: residualvq
            self.quantizer = ResidualVQ(
                dim = codebook_dim,  # encoder output dimension
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                shared_codebook=shared_codebook,
            )
            logger.info(f"Using ResidualVQ")
        
        # Decoder (Being-H0 style parameters)
        self.decoder = Decoder(
            input_emb_width=action_dim,         # action dimension
            output_emb_width=codebook_dim,          # latent dimension
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            num_conv_layers=num_conv_layers,
        )
        
        logger.info(f"ActionVQModel initialized: action_dim={action_dim}, "
                   f"time_horizon={time_horizon}, codebook_size={codebook_size}, "
                   f"res_len={self.res_len} (padding {self.res_len} frames)")
    
    def forward(self, actions: torch.Tensor) -> tuple:
        """
        Full forward pass: encode -> quantize -> decode.
        
        Args:
            actions: [batch, time, action_dim]
        Returns:
            reconstructed: [batch, time, action_dim]
            commitment_loss: scalar
            indices: [batch, time_compressed, num_quantizers] or [B, T_compressed, num_groups, num_quantizers]
        """
        batch_size, original_time, action_dim = actions.shape
        
        # Zero padding (BeingH0 style) to make time divisible by downsample factor
        if self.res_len > 0:
            padding = torch.zeros(batch_size, self.res_len, action_dim, 
                                 dtype=actions.dtype, device=actions.device)
            actions = torch.cat([actions, padding], dim=1)  # [B, T+res_len, D]
        
        # [B, T, D] -> [B, D, T]
        actions_t = actions.permute(0, 2, 1)
        
        # Encode
        z = self.encoder(actions_t)  # [B, codebook_dim, T_compressed]
        
        # Transpose for quantizer: [B, codebook_dim, T] -> [B, T, codebook_dim]
        z = z.permute(0, 2, 1)
        
        # Quantize (expects [B, T, D])
        z_q, indices, commitment_loss = self.quantizer(z)  # [B, T, codebook_dim]
        
        # Transpose back for decoder: [B, T, codebook_dim] -> [B, codebook_dim, T]
        z_q = z_q.permute(0, 2, 1)
        
        # Decode
        reconstructed_t = self.decoder(z_q)  # [B, action_dim, T_padded]
        
        # [B, D, T] -> [B, T, D]
        reconstructed = reconstructed_t.permute(0, 2, 1)
        
        # Remove padding to get original time length
        if self.res_len > 0:
            reconstructed = reconstructed[:, :-self.res_len, :]  # [B, T, D]
        
        return reconstructed, commitment_loss, indices
    
    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode actions to discrete token indices.
        
        Args:
            actions: [batch, time, action_dim]
        Returns:
            indices: [batch, num_tokens] flattened token sequence
                     RVQ: [B, T_compressed * num_quantizers]
                     GRVQ: [B, T_compressed * num_groups * num_quantizers]
        """
        batch_size, original_time, action_dim = actions.shape
        
        # Zero padding (BeingH0 style)
        if self.res_len > 0:
            padding = torch.zeros(batch_size, self.res_len, action_dim, 
                                 dtype=actions.dtype, device=actions.device)
            actions = torch.cat([actions, padding], dim=1)  # [B, T+res_len, D]
        
        # [B, T, D] -> [B, D, T]
        actions_t = actions.permute(0, 2, 1)
        
        # Encode
        z = self.encoder(actions_t)  # [B, codebook_dim, T_compressed]
        
        # Transpose for quantizer: [B, codebook_dim, T] -> [B, T, codebook_dim]
        z = z.permute(0, 2, 1)
        
        # Quantize
        z_q, indices, _ = self.quantizer(z)
        
        # Flatten indices
        # RVQ: [B, T_compressed, num_quantizers] -> [B, T_compressed * num_quantizers]
        # GRVQ: [B, T_compressed, num_groups, num_quantizers] -> [B, T_compressed * num_groups * num_quantizers]
        batch_size = indices.shape[0]
        indices_flat = indices.reshape(batch_size, -1)
        
        return indices_flat
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete token indices to actions.
        
        Args:
            indices: [batch, num_tokens] flattened token sequence
                     RVQ: [B, T_compressed * num_quantizers]
                     GRVQ: [B, T_compressed * num_groups * num_quantizers]
        Returns:
            actions: [batch, time, action_dim]
        """
        # Unflatten indices (use padded time horizon for calculation)
        batch_size = indices.shape[0]
        time_compressed = self.padded_time_horizon // self.downsample_factor
        
        # Reshape indices based on quantizer type
        if self.quantizer_type == "group_residualvq":
            # GRVQ: [B, T*G*Q] -> [G, B, T, Q]
            indices = indices.reshape(batch_size, time_compressed, self.num_groups, self.num_quantizers)
            indices = indices.permute(2, 0, 1, 3)  # [B, T, G, Q] -> [G, B, T, Q]
        else:
            # RVQ: [B, T*Q] -> [B, T, Q]
            indices = indices.reshape(batch_size, time_compressed, self.num_quantizers)
        
        # Get codes from quantizer (handles both RVQ and GRVQ)
        z_q = self.quantizer.get_codes_from_indices(indices)
        
        # Handle different output formats
        if z_q.dim() == 4:  # [G, B, T, D]
            z_q = z_q.permute(1, 2, 0, 3)  # -> [B, T, G, D]
            z_q = z_q.sum(dim=2)  # Sum across groups: [B, T, G, D] -> [B, T, D]
        elif z_q.dim() == 5:  # [B, Q, G, T, D]
            # Sum across quantizers and groups: [B, Q, G, T, D] -> [B, T, D]
            z_q = z_q.sum(dim=1).sum(dim=1)
        
        # Transpose for decoder: [B, T, D] -> [B, D, T]
        z_q = z_q.permute(0, 2, 1)
        
        # Decode to actions
        actions_t = self.decoder(z_q)  # [B, action_dim, T_reconstructed]
        
        # [B, D, T] -> [B, T, D]
        actions = actions_t.permute(0, 2, 1)
        
        # Remove padding to get original time horizon
        if self.res_len > 0:
            actions = actions[:, :-self.res_len, :]  # [B, T_padded, D] -> [B, T, D]
        
        return actions


def train_vq_model(
    model: ActionVQModel,
    action_data: torch.Tensor,
    device: str = "cuda",
    num_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    commitment_weight: float = 0.02,
    velocity_weight: float = 0.01,
    part: str = 'wrist',
):
    """
    Train VQ-VAE model on action data.
    
    Args:
        model: ActionVQModel to train
        action_data: [num_sequences, time, action_dim]
        device: Training device
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        commitment_weight: Weight for commitment loss
    
    Returns:
        Trained model
    """
    model = model.to(device)
    model.train()
    
    # Convert to tensor if needed
    if isinstance(action_data, torch.Tensor):
        action_data = action_data.float()
    else:
        action_data = torch.from_numpy(action_data).float()
    
    # Create dataloader
    dataset = torch.utils.data.TensorDataset(action_data)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    logger.info(f"Training VQ model: {len(action_data)} sequences, {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        total_recon_loss = 0.0
        total_commit_loss = 0.0
        total_velocity_loss = 0.0
        total_loss = 0.0
        for batch_idx, (actions,) in enumerate(dataloader):
            actions = actions.to(device)
            
            # Forward pass
            reconstructed, commitment_loss, _ = model(actions)
            
            # Ensure commitment_loss is scalar
            if commitment_loss.numel() > 1:
                commitment_loss = commitment_loss.mean()
            
            # Losses need to modify here if wanna train left and right hand separately.

            recon_loss = F.mse_loss(reconstructed, actions)
            if part == 'wrist':
                velocity_loss = F.mse_loss(reconstructed, actions)
                total_loss = recon_loss + commitment_weight * commitment_loss + velocity_weight * velocity_loss
            else:
                velocity_loss = torch.tensor(0.0, device=device)
                total_loss = recon_loss + commitment_weight * commitment_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_commit_loss += commitment_loss.item()
            total_velocity_loss += velocity_loss.item()
            total_loss += total_loss.item()
        
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_commit_loss = total_commit_loss / len(dataloader)
        avg_velocity_loss = total_velocity_loss / len(dataloader)
        avg_total_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Recon Loss = {avg_recon_loss:.6f}, "
                       f"Commit Loss = {avg_commit_loss:.6f}, "
                       f"Velocity Loss = {avg_velocity_loss:.6f}, "
                       f"Total Loss = {avg_total_loss:.6f}")
    
    model.eval()
    return model


