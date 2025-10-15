# VQ-VAE style Model for Action Tokenization
# Adapted from https://github.com/BeingBeyond/Being-H0/beingvla/models/motion/m2m/tokenizer/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from vector_quantize_pytorch import GroupedResidualVQ, ResidualVQ, FSQ

from .encdec import Encoder, Decoder


class BaseVQModel(nn.Module):
    """
    VQ-VAE model for action tokenization.
    
    Similar to ManoVQModel but adapted for general action sequences.
    Supports both RVQ and GRVQ (Grouped Residual VQ).
    """
    
    def __init__(
        self,
        args, # config dict
        motion_dim: Optional[int] = None
    ):
        super().__init__()
        """Initialize encoder/decoder and quantizer based on config."""
        self.code_dim = args.code_dim
        self.motion_dim = motion_dim if motion_dim is not None else args.motion_dim
        self.quantizer_name = args.quantizer_name
        
        self.encoder = Encoder(
            input_emb_width=self.motion_dim,
            output_emb_width=args.output_emb_width,
            down_t=args.down_t,
            stride_t=args.stride_t,
            width=args.width,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            activation=args.activate,
            norm=args.norm,
            num_conv_layers=args.num_conv_layers
        )
        self.decoder = Decoder(
            input_emb_width=self.motion_dim,
            output_emb_width=args.output_emb_width,
            down_t=args.down_t,  # Assuming symmetric
            stride_t=args.stride_t,
            width=args.width,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            activation=args.activate,
            norm=args.norm,
            num_conv_layers=args.num_conv_layers
        )

        self.quantizer = self._create_quantizer(args)
    
    def _create_quantizer(self, args):
        """Factory method for quantizer creation."""
        levels_dict = {
            256: [8, 6, 5], 512: [8, 8, 8],
            1024: [8, 5, 5, 5], 2048: [8, 8, 6, 5],
            4096: [7, 5, 5, 5, 5], 8192: [8, 6, 6, 5, 5],
            16384: [8, 8, 8, 6, 5], 65536: [8, 8, 8, 5, 5, 5]
        }
        quantizers = {
            "residualvq": lambda: ResidualVQ(
                codebook_size=args.nb_code,
                dim=args.code_dim,
                num_quantizers=args.num_quantizers,
                shared_codebook=args.shared_codebook
            ),
            "group_residualvq": lambda: GroupedResidualVQ(codebook_size=args.nb_code, dim=args.code_dim, 
                                               num_quantizers=args.num_quantizers, 
                                               groups=args.num_quant_groups,
                                               shared_codebook=args.shared_codebook),
            "fsq": lambda: FSQ(levels=levels_dict[args.nb_code], dim=args.code_dim)
        }
        return quantizers[args.quantizer_name]()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).float() # (bs, T, Jx3) -> (bs, Jx3, T)
    
    def postprocess(self, x: torch.Tensor) -> torch.Tensor: 
        return x.permute(0, 2, 1) # (bs, Jx3, T) ->  (bs, T, Jx3)
    
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
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(bs, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out
    
    def forward(self, x: torch.Tensor) -> tuple:
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in)
        x_quant, commit_loss, perplexity = self.quantizer(x_enc)
        x_out = self.postprocess(self.decoder(x_quant))
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
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in).permute(0, 2, 1)
        x_quant, indices, commit_loss = self.quantizer(x_enc)
        x_out = self.postprocess(self.decoder(x_quant.permute(0, 2, 1)))
        return x_out, commit_loss.mean(), torch.tensor(-1, device=x.device)


class GroupResidualVQModel(BaseVQModel):
    """Grouped Residual VQ variant."""
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in).permute(0, 2, 1) # B, T/4, D'
        _, indices, _ = self.quantizer(x_enc) # indices: 2, B, T/4, layer
        return indices

    def forward_decoder(self, x):
        bs = x.shape[1]
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(bs, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out
    
    def forward(self, x: torch.Tensor) -> tuple:
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in).permute(0, 2, 1)
        x_quant, indices, commit_loss = self.quantizer(x_enc) # indices: 2, B, T/4, layer
        x_out = self.postprocess(self.decoder(x_quant.permute(0, 2, 1)))
        return x_out, commit_loss.mean(), torch.tensor(-1, device=x.device)


class FSQModel(BaseVQModel):
    """Finite Scalar Quantization Model."""

    def encode(self, x):
        x_enc = self.encoder(self.preprocess(x))
        _, code_idx, _, _, _, _ = self.quantizer(x_enc)
        return code_idx.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> tuple:
        x_enc = self.encoder(self.preprocess(x)) # 256, 512, 16
        x_quant, _, loss, perplexity, activate, indices = self.quantizer(x_enc)
        
        x_decoder = self.decoder(x_quant)
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity


class MotionReconstructionLoss(nn.Module):
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


class MotionVQModel(PreTrainedModel):
    """Main human motion VQ-VAE model."""
    config_class = PretrainedConfig  # Can customize config class
    
    def __init__(self, args):
        config = PretrainedConfig()
        super().__init__(config)
  
        self.model = self._create_vq_model(args)
        self.motion_dim = args.motion_dim
        self.loss_fn = MotionReconstructionLoss(args.recons_loss)

        self.commit_weight = args.commit_weight

    def _create_vq_model(self, args):
        """Factory method for VQ model creation."""
        model_map = {
            "residualvq": ResidualVQModel,
            "group_residualvq": GroupResidualVQModel,
            "fsq": FSQModel
        }
        return model_map.get(args.quantizer_name, BaseVQModel)(args)
    
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
            'pred_motion': pred_motion,
        }


class PartialMotionVQModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, shape_meta, args):
        config = PretrainedConfig()
        super().__init__(config)
        self.use_part = args.use_part
        self.wrist_dim = shape_meta["obs"]["state"]["wrist"]["shape"][0]
        self.hand_dim = shape_meta["obs"]["state"]["hand"]["shape"][0]

        if args.use_part is not None:
            if args.use_part == "wrist":
                self.model = self._create_vq_model(args, dim_motion=self.wrist_dim)
            elif args.use_part == "hand":
                self.model = self._create_vq_model(args, dim_motion=self.hand_dim)
            else:  # both
                raise NotImplementedError(f"Unsupported use_part: {args.use_part}")
        else:
            self.model = self._create_vq_model(args)

        self.loss_fn = MotionReconstructionLoss(args.recons_loss)

        self.commit_weight = args.commit_weight

    def _create_vq_model(self, args, dim_motion=None):
        """Factory method for VQ model creation."""
        model_map = {
            "residualvq": ResidualVQModel,
            "group_residualvq": GroupResidualVQModel,
            "fsq": FSQModel
        }
        return model_map.get(args.quantizer_name, BaseVQModel)(args, dim_motion)
    
    def encode(self, x):
        return self.model.encode(x)

    def forward_decoder(self, x):
        return self.model.forward_decoder(x)

    def forward(self, motion: torch.Tensor, wrist_motion: torch.Tensor, hand_motion: torch.Tensor, **kwargs):
        if self.use_part is None:
            x_motion = motion
        elif self.use_part == "wrist":
            x_motion = wrist_motion
        elif self.use_part == "hand":
            x_motion = hand_motion
        
        pred_motion, commit_loss, perplexity = self.model(x_motion.float())
        recon_loss = self.loss_fn(pred_motion, x_motion)
        total_loss = recon_loss + self.commit_weight * commit_loss
        
        return {
            'loss': total_loss,
            'loss_recons': recon_loss,
            'perplexity': perplexity,
            'loss_commit': commit_loss,
            'pred_motion': pred_motion,
        }

