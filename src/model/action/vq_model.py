# VQ-VAE style Model for Action Tokenization
# Adapted from https://github.com/BeingBeyond/Being-H0/beingvla/models/motion/m2m/tokenizer/model.py

import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel
from einops import rearrange
from vector_quantize_pytorch import GroupedResidualVQ, ResidualVQ, FSQ

from .encdec import Encoder, Decoder
from .vq_config import MotionVQModelConfig


class BaseVQModel(nn.Module):
    """
    VQ-VAE model for action tokenization.
    
    Supports both RVQ, GRVQ and FSQ.
    """
    
    def __init__(
        self,
        config: MotionVQModelConfig,
        motion_dim: Optional[int] = None
    ):
        super().__init__()
        """Initialize encoder/decoder and quantizer based on config."""
        self.code_dim = config.quantizer_config.codebook_dim
        self.codebook_size = config.quantizer_config.nb_code
        self.motion_dim = motion_dim if motion_dim is not None else config.motion_dim
        self.quantizer_name = config.quantizer_config.quantizer_name
        model_config = config.model_config
        self.encoder = Encoder(
            input_emb_width=self.motion_dim,
            output_emb_width=model_config.output_emb_width,
            down_t=model_config.down_t,
            stride_t=model_config.stride_t,
            width=model_config.width,
            depth=model_config.depth,
            dilation_growth_rate=model_config.dilation_growth_rate,
            activation=model_config.activate,
            norm=model_config.norm,
            num_conv_layers=model_config.num_conv_layers
        )
        self.decoder = Decoder(
            input_emb_width=self.motion_dim,
            output_emb_width=model_config.output_emb_width,
            down_t=model_config.down_t,  # Assuming symmetric
            stride_t=model_config.stride_t,
            width=model_config.width,
            depth=model_config.depth,
            dilation_growth_rate=model_config.dilation_growth_rate,
            activation=model_config.activate,
            norm=model_config.norm,
            num_conv_layers=model_config.num_conv_layers
        )

        self.quantizer = self._create_quantizer(config.quantizer_config)
    
    def _create_quantizer(self, quantizer_config):
        """Factory method for quantizer creation."""
        levels_dict = {
            256: [8, 6, 5], 512: [8, 8, 8],
            1024: [8, 5, 5, 5], 2048: [8, 8, 6, 5],
            4096: [7, 5, 5, 5, 5], 8192: [8, 6, 6, 5, 5],
            16384: [8, 8, 8, 6, 5], 65536: [8, 8, 8, 5, 5, 5]
        }
        quantizers = {
            "residualvq": lambda: ResidualVQ(
                codebook_size=quantizer_config.nb_code,
                dim=quantizer_config.codebook_dim,
                num_quantizers=quantizer_config.num_quantizers,
                shared_codebook=quantizer_config.shared_codebook
            ),
            "group_residualvq": lambda: GroupedResidualVQ(codebook_size=quantizer_config.nb_code, dim=quantizer_config.codebook_dim, 
                                               num_quantizers=quantizer_config.num_quantizers, 
                                               groups=quantizer_config.num_groups,
                                               shared_codebook=quantizer_config.shared_codebook),
            "fsq": lambda: FSQ(levels=levels_dict[quantizer_config.nb_code], dim=quantizer_config.codebook_dim)
        }
        return quantizers[quantizer_config.quantizer_name]()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).float() # (bs, T, D) -> (bs, D, T)
    
    def postprocess(self, x: torch.Tensor) -> torch.Tensor: 
        return x.permute(0, 2, 1) # (bs, D, T) ->  (bs, T, D)
    
    def encode(self, x: torch.Tensor):
        N, T, _ = x.shape
        try:
            x_encoder = self.encoder(self.preprocess(x))
        except:
            breakpoint()
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder).view(N, -1)
        return code_idx

    def forward_decoder(self, x):
        bs = x.shape[0]
        x_d = self.quantizer.get_output_from_indices(x)
        x_d = x_d.view(bs, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out
    
    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in)
        x_quant, commit_loss, perplexity = self.quantizer(x_enc)
        x_out = self.postprocess(self.decoder(x_quant))[:, :T, :]
        # The output horizon of decoder may be larger than the input's, 
        # so we need to truncate the output to the input's horizon.
        return x_out, commit_loss, perplexity


# ==================== Specialized Quantizer Models ====================
class ResidualVQModel(BaseVQModel):
    """Residual VQ variant."""
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in).permute(0, 2, 1)
        _, indices, _ = self.quantizer(x_enc)
        return indices
    
    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in).permute(0, 2, 1)
        x_quant, indices, commit_loss = self.quantizer(x_enc) # indices: B, T/4, layer
        x_out = self.postprocess(self.decoder(x_quant.permute(0, 2, 1)))[:, :T, :]
        return x_out, commit_loss.mean(), calculate_perplexity_grvq(indices, self.codebook_size)


class GroupResidualVQModel(BaseVQModel):
    """Grouped Residual VQ variant."""
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in).permute(0, 2, 1) # B, T/4, D'
        _, indices, _ = self.quantizer(x_enc) # indices: 2, B, T/4, layer
        return indices

    def forward_decoder(self, x):
        bs = x.shape[1]
        x_d = self.quantizer.get_output_from_indices(x)
        x_d = x_d.view(bs, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out
    
    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in).permute(0, 2, 1)
        x_quant, indices, commit_loss = self.quantizer(x_enc) # indices: 2, B, T/4, layer
        x_out = self.postprocess(self.decoder(x_quant.permute(0, 2, 1)))[:, :T, :]
        return x_out, commit_loss.mean(), calculate_perplexity_grvq(indices, self.codebook_size)


class FSQModel(BaseVQModel):
    """Finite Scalar Quantization Model."""

    def encode(self, x):
        x_enc = self.encoder(self.preprocess(x))
        _, code_idx, _, _, _, _ = self.quantizer(x_enc)
        return code_idx.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        x_enc = self.encoder(self.preprocess(x)) # 256, 512, 16
        x_quant, _, loss, perplexity, activate, indices = self.quantizer(x_enc)
        
        x_decoder = self.decoder(x_quant)[:, :T, :]
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity


# ==================== Loss Functions ====================
class ReconstructionLoss(nn.Module):
    """Handles motion reconstruction loss."""
    LOSS_MAP = {
        'l1': nn.L1Loss,
        'l2': nn.MSELoss,
        'l1_smooth': nn.SmoothL1Loss
    }

    def __init__(self, recons_loss: str):
        super().__init__()
        self.loss_fn = self.LOSS_MAP[recons_loss]()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


# ==================== Motion VQ Model ====================
class MotionVQModel(PreTrainedModel):
    config_class = MotionVQModelConfig

    def __init__(self, config: MotionVQModelConfig):
        super().__init__(config)
        self.use_part = config.use_part
        self.horizon = config.horizon
        self.motion_dim = config.motion_dim
        self.wrist_dim = config.wrist_dim
        self.hand_dim = config.hand_dim
        self.commit_weight = config.loss_config.commit_weight
        self.vocab_size = config.vocab_size
        if self.use_part is not None:
            if self.use_part == "wrist":
                self.model = self._create_vq_model(config, motion_dim=self.wrist_dim)
            elif self.use_part == "hand":
                self.model = self._create_vq_model(config, motion_dim=self.hand_dim)
            else:  # both
                raise NotImplementedError(f"Unsupported use_part: {self.use_part}")
        else:
            self.model = self._create_vq_model(config)
        self.loss_fn = ReconstructionLoss(config.loss_config.recons_loss)

    def _create_vq_model(self, config: MotionVQModelConfig, motion_dim: Optional[int] = None):
        """Factory method for VQ model creation."""
        model_map = {
            "residualvq": ResidualVQModel,
            "group_residualvq": GroupResidualVQModel,
            "fsq": FSQModel
        }
        return model_map.get(config.quantizer_config.quantizer_name, BaseVQModel)(config, motion_dim)
    
    def encode(self, x):
        return self.model.encode(x)

    def forward_decoder(self, x):
        return self.model.forward_decoder(x)

    def forward(self, motion: torch.Tensor, **kwargs):
        pred_motion, commit_loss, perplexity = self.model(motion.float())
        recon_loss = self.loss_fn(pred_motion, motion)
        total_loss = recon_loss + self.commit_weight * commit_loss
        
        return {
            'loss': total_loss,
            'loss_recons': recon_loss,
            'perplexity': perplexity,
            'loss_commit': commit_loss,
            'avg_pred_motion': pred_motion.mean(),
        }


def calculate_perplexity_grvq(indices, codebook_size, eps=1e-5):
    """
    Calculate perplexity for Grouped Residual VQ.
    Args:
        indices: [Groups, Batch, Length, Layers] or [Batch, Length, Layers]
        codebook_size: int
    Returns:
        perplexities: [Groups, Layers] or [Layers]
    """
    is_grouped = len(indices.shape) == 4
    if not is_grouped:
        indices = indices.unsqueeze(0)

    flat_indices = rearrange(indices, 'g b t l -> g l (b t)').long()
    G, L, N = flat_indices.shape
    counts = torch.zeros(G, L, codebook_size, device=indices.device)
    src = torch.ones_like(flat_indices, dtype=torch.float32)
    counts.scatter_add_(2, flat_indices, src) 
    probs = counts / counts.sum(dim=-1, keepdim=True)
    perplexities = torch.exp(-torch.sum(probs * torch.log(probs + eps), dim=-1))
            
    if not is_grouped:
        perplexities = perplexities.squeeze(0)
    return perplexities

