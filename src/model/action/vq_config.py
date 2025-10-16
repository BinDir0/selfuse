"""
VQ Model Configuration Classes

This module defines configuration classes for VQ models to properly integrate
with the transformers library's PreTrainedModel system.
"""

from transformers import PretrainedConfig
from typing import Optional, Dict, Any


class ModelArchConfig(PretrainedConfig):
    """Model architecture configuration (nested under 'model' in YAML)"""
    
    model_type = "vq_model_arch"
    
    def __init__(
        self,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 256,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        output_emb_width: int = 256,
        activate: str = "relu", # "relu", "gelu", "silu"
        norm: Optional[str] = None, # "LN", "GN", "BN"
        num_conv_layers: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.down_t = down_t
        self.stride_t = stride_t
        self.width = width
        self.depth = depth
        self.dilation_growth_rate = dilation_growth_rate
        self.output_emb_width = output_emb_width
        self.activate = activate
        self.norm = norm
        self.num_conv_layers = num_conv_layers


class QuantizerConfig(PretrainedConfig):
    """Quantizer configuration (nested under 'quantizer' in YAML)"""
    
    model_type = "vq_quantizer"
    
    def __init__(
        self,
        quantizer_name: str = "group_residualvq", # "residualvq", "group_residualvq", "fsq"
        nb_code: int = 1024, # codebook size (vocabulary size)
        codebook_dim: int = 256, # codebook embedding dimension
        num_quantizers: int = 8, # number of quantizers for Res_VQ
        num_groups: int = 1, # number of groups for Group_Res_VQ
        shared_codebook: bool = True, # whether to share codebook for Res_VQ
        **kwargs
    ):
        super().__init__(**kwargs)
        self.quantizer_name = quantizer_name
        self.nb_code = nb_code
        self.codebook_dim = codebook_dim
        self.num_quantizers = num_quantizers
        self.num_groups = num_groups
        self.shared_codebook = shared_codebook


class LossConfig(PretrainedConfig):
    """Loss configuration (nested under 'loss' in YAML)"""
    
    model_type = "vq_loss"
    
    def __init__(
        self,
        recons_loss: str = "l2", # "l1", "l2", "l1_smooth"
        commit_weight: float = 0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.recons_loss = recons_loss
        self.commit_weight = commit_weight


class MotionVQModelConfig(PretrainedConfig):
    """
    Configuration class for Motion VQ-VAE models with nested structure.
    """
    
    model_type = "vq_model"
    
    def __init__(
        self,
        # Motion dimensions
        use_part: Optional[str] = None, # "wrist", "hand", None for full state
        motion_dim: int = 24, # full motion dimension
        wrist_dim: int = 9, # wrist motion dimension
        hand_dim: int = 15, # hand motion dimension
        
        model_config: Optional[Dict[str, Any]] = None,
        quantizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        
        **kwargs
    ):
        """
        Args:
            use_part: Which part to use ("wrist", "hand", None for full state)
            motion_dim: Full motion dimension
            wrist_dim: Wrist motion dimension
            hand_dim: Hand motion dimension
            codebook_dim: Codebook embedding dimension
            nb_code: Codebook size (vocabulary size)
            mu: EMA decay for codebook updates
            model_config: Nested model architecture config (dict or ModelArchConfig)
            quantizer_config: Nested quantizer config (dict or QuantizerConfig)
            loss_config: Nested loss config (dict or LossConfig)
        """
        super().__init__(**kwargs)
        
        # Motion dimensions
        self.use_part = use_part
        self.motion_dim = motion_dim
        self.wrist_dim = wrist_dim
        self.hand_dim = hand_dim
        
        # Nested configs - handle dict, Config object, or None
        if model_config is None:
            self.model_config = ModelArchConfig()
        else: 
            self.model_config = ModelArchConfig(**model_config)
        
        if quantizer_config is None:
            self.quantizer_config = QuantizerConfig()
        else: 
            self.quantizer_config = QuantizerConfig(**quantizer_config)
        
        if loss_config is None:
            self.loss_config = LossConfig()
        else: 
            self.loss_config = LossConfig(**loss_config)
