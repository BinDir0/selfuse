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
        activation: str = "silu", # "relu", "gelu", "silu"
        norm: Optional[str] = None, # "LN", "GN", "BN"
        num_conv_layers: int = 3,
        encoder_causal: bool = True, # whether to use causal convolution for encoder
        decoder_causal: bool = False, # whether to use causal convolution for decoder
        **kwargs
    ):
        # For decoder, we do not use causal convolution by default
        # except when we need real-time streaming inference
        super().__init__(**kwargs)
        self.down_t = down_t
        self.stride_t = stride_t
        self.width = width
        self.depth = depth
        self.dilation_growth_rate = dilation_growth_rate
        self.output_emb_width = output_emb_width
        self.activation = activation
        self.norm = norm
        self.num_conv_layers = num_conv_layers
        self.encoder_causal = encoder_causal
        self.decoder_causal = decoder_causal
    
    def to_dict(self):
        """Override to only save relevant parameters."""
        return {
            'model_type': self.model_type,
            'down_t': self.down_t,
            'stride_t': self.stride_t,
            'width': self.width,
            'depth': self.depth,
            'dilation_growth_rate': self.dilation_growth_rate,
            'output_emb_width': self.output_emb_width,
            'activation': self.activation,
            'norm': self.norm,
            'num_conv_layers': self.num_conv_layers,
            'encoder_causal': self.encoder_causal,
            'decoder_causal': self.decoder_causal,
        }


class QuantizerConfig(PretrainedConfig):
    """Quantizer configuration (nested under 'quantizer' in YAML)"""
    
    model_type = "vq_quantizer"
    
    def __init__(
        self,
        quantizer_name: str = "group_residualvq", # "residualvq", "group_residualvq", "fsq"
        codebook_size: int = 1024, # codebook size (vocabulary size)
        codebook_dim: int = 256, # codebook embedding dimension
        num_quantizers: int = 8, # number of quantizers for Res_VQ
        num_groups: int = 1, # number of groups for Group_Res_VQ
        shared_codebook: bool = True, # whether to share codebook for Res_VQ
        **kwargs
    ):
        super().__init__(**kwargs)
        self.quantizer_name = quantizer_name
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_quantizers = num_quantizers
        self.num_groups = num_groups
        self.shared_codebook = shared_codebook
    
    def to_dict(self):
        """Override to only save relevant parameters."""
        return {
            'model_type': self.model_type,
            'quantizer_name': self.quantizer_name,
            'codebook_size': self.codebook_size,
            'codebook_dim': self.codebook_dim,
            'num_quantizers': self.num_quantizers,
            'num_groups': self.num_groups,
            'shared_codebook': self.shared_codebook,
        }


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
    
    def to_dict(self):
        """Override to only save relevant parameters."""
        return {
            'model_type': self.model_type,
            'recons_loss': self.recons_loss,
            'commit_weight': self.commit_weight,
        }


class MotionVQModelConfig(PretrainedConfig):
    """
    Configuration class for Motion VQ-VAE models with nested structure.
    """
    
    model_type = "vq_model"
    
    def __init__(
        self,
        # Motion dimensions
        use_part: Optional[str] = None, # "wrist", "hand", None for full state
        horizon: int = 32, # time horizon
        motion_dim: int = 48, # full motion dimension
        wrist_dim: int = 18, # bimanual wrist dimension
        hand_dim: int = 30, # bimanual hand dimension 
        
        model_config: Optional[Dict[str, Any]] = None,
        quantizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        
        **kwargs
    ):
        """
        Args:
            use_part: Which part to use ("wrist", "hand", None for full state)
            horizon: Time horizon
            motion_dim: Full motion dimension
            wrist_dim: Wrist motion dimension
            hand_dim: Hand motion dimension
            model_config: Nested model architecture config (dict or ModelArchConfig)
            quantizer_config: Nested quantizer config (dict or QuantizerConfig)
            loss_config: Nested loss config (dict or LossConfig)
        """
        super().__init__(**kwargs)
        
        # Motion dimensions
        self.use_part = use_part
        self.horizon = horizon
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

        # Vocabulary size
        if self.quantizer_config.shared_codebook:
            self.vocab_size = self.quantizer_config.codebook_size
        else:
            self.vocab_size = self.quantizer_config.codebook_size * self.quantizer_config.num_quantizers
    
    def to_dict(self):
        """
        Override to properly serialize nested PretrainedConfig objects.
        Only save relevant parameters, not the inherited defaults.
        """
        import transformers
        
        output = {
            'model_type': self.model_type,
            'use_part': self.use_part,
            'horizon': self.horizon,
            'motion_dim': self.motion_dim,
            'wrist_dim': self.wrist_dim,
            'hand_dim': self.hand_dim,
            'transformers_version': transformers.__version__,
        }
        
        # Convert nested configs to dicts (they have their own to_dict())
        if hasattr(self, 'model_config') and self.model_config is not None:
            output['model_config'] = self.model_config.to_dict()
        
        if hasattr(self, 'quantizer_config') and self.quantizer_config is not None:
            output['quantizer_config'] = self.quantizer_config.to_dict()
        
        if hasattr(self, 'loss_config') and self.loss_config is not None:
            output['loss_config'] = self.loss_config.to_dict()
        
        return output
    
    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Override to prevent filtering of nested configs.
        
        HuggingFace's default implementation compares with default config
        and filters out "unchanged" values, which breaks nested configs.
        """
        import json
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
    
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Override to properly deserialize nested PretrainedConfig objects.
        """
        # Make a copy to avoid modifying the original
        config_dict = dict(config_dict)
        
        # Extract nested configs before calling super()
        model_config_dict = config_dict.pop('model_config', None)
        quantizer_config_dict = config_dict.pop('quantizer_config', None)
        loss_config_dict = config_dict.pop('loss_config', None)
        
        # Create the main config - pass the nested dicts to __init__
        config_dict['model_config'] = model_config_dict
        config_dict['quantizer_config'] = quantizer_config_dict
        config_dict['loss_config'] = loss_config_dict
        
        # Call parent's from_dict, which will call __init__ with our config_dict
        return super().from_dict(config_dict, **kwargs)
