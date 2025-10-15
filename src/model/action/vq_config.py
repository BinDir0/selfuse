"""
VQ Model Configuration Classes

This module defines configuration classes for VQ models to properly integrate
with the transformers library's PreTrainedModel system.
"""

from transformers import PretrainedConfig
from typing import Optional, Dict, Any


class VQModelConfig(PretrainedConfig):
    """
    Configuration class for VQ-VAE models.
    
    This configuration stores all hyperparameters needed to instantiate a VQ model.
    Using this configuration class enables:
    - Automatic config saving/loading with save_pretrained()/from_pretrained()
    - Proper integration with HuggingFace Hub
    - Version control and reproducibility
    
    Example:
        >>> config = VQModelConfig(
        ...     use_part="wrist",
        ...     codebook_dim=256,
        ...     nb_code=1024
        ... )
        >>> model = PartialMotionVQModel(config=config)
        >>> model.save_pretrained("/path/to/save")  # Saves both model and config
    """
    
    model_type = "vq_model"
    
    def __init__(
        self,
        # Motion dimensions
        use_part: Optional[str] = None,  # "wrist", "hand", or None for full
        motion_dim: int = 48,
        wrist_dim: int = 18,
        hand_dim: int = 30,
        
        # Codebook config
        codebook_dim: int = 256,
        nb_code: int = 1024,
        mu: float = 0.99,
        
        # Model architecture
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 256,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        output_emb_width: int = 256,
        activate: str = "relu",
        norm: Optional[str] = None,
        num_conv_layers: int = 3,
        
        # Quantizer config
        quantizer_name: str = "group_residualvq",
        quantbeta: float = 1.0,
        num_quantizers: int = 8,
        num_groups: int = 1,
        shared_codebook: bool = True,
        
        # Loss config
        recons_loss: str = "l2",
        commit_weight: float = 0.02,
        
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
            down_t: Temporal downsampling rate
            stride_t: Temporal stride
            width: Network width
            depth: Network depth
            dilation_growth_rate: Dilation growth rate for temporal convs
            output_emb_width: Output embedding width
            activate: Activation function name
            norm: Normalization type (None, "batch", "layer")
            num_conv_layers: Number of conv layers in encoder/decoder
            quantizer_name: Quantizer type ("residualvq", "group_residualvq", "fsq")
            quantbeta: Quantization beta parameter
            num_quantizers: Number of quantizers for RVQ/GRVQ
            num_groups: Number of groups for GRVQ
            shared_codebook: Whether to share codebook across quantizers
            recons_loss: Reconstruction loss type ("l1", "l2")
            commit_weight: Commitment loss weight
        """
        super().__init__(**kwargs)
        
        # Motion dimensions
        self.use_part = use_part
        self.motion_dim = motion_dim
        self.wrist_dim = wrist_dim
        self.hand_dim = hand_dim
        
        # Codebook
        self.codebook_dim = codebook_dim
        self.nb_code = nb_code
        self.mu = mu
        
        # Model architecture
        self.down_t = down_t
        self.stride_t = stride_t
        self.width = width
        self.depth = depth
        self.dilation_growth_rate = dilation_growth_rate
        self.output_emb_width = output_emb_width
        self.activate = activate
        self.norm = norm
        self.num_conv_layers = num_conv_layers
        
        # Quantizer
        self.quantizer_name = quantizer_name
        self.quantbeta = quantbeta
        self.num_quantizers = num_quantizers
        self.num_groups = num_groups
        self.shared_codebook = shared_codebook
        
        # Loss
        self.recons_loss = recons_loss
        self.commit_weight = commit_weight
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this configuration to a Python dictionary.
        
        Returns:
            Dictionary of all attributes that define this configuration instance.
        """
        output = super().to_dict()
        return output
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "VQModelConfig":
        """
        Constructs a configuration from a Python dictionary.
        
        Args:
            config_dict: Dictionary with configuration attributes.
            
        Returns:
            VQModelConfig instance.
        """
        return cls(**config_dict)


class PartialMotionVQModelConfig(VQModelConfig):
    """
    Configuration specific to PartialMotionVQModel.
    
    Extends VQModelConfig with shape_meta information.
    """
    
    model_type = "partial_motion_vq_model"
    
    def __init__(
        self,
        shape_meta: Optional[Dict] = None,
        **kwargs
    ):
        """
        Args:
            shape_meta: Dictionary containing shape information for obs/action
            **kwargs: Arguments passed to VQModelConfig
        """
        super().__init__(**kwargs)
        self.shape_meta = shape_meta or {}
        
        # Extract dimensions from shape_meta if provided
        if shape_meta:
            if "obs" in shape_meta and "state" in shape_meta["obs"]:
                state_meta = shape_meta["obs"]["state"]
                if "wrist" in state_meta:
                    self.wrist_dim = state_meta["wrist"]["shape"][0]
                if "hand" in state_meta:
                    self.hand_dim = state_meta["hand"]["shape"][0]
                if "shape" in state_meta:
                    self.motion_dim = state_meta["shape"][0]


# Factory function to create config from OmegaConf
def create_vq_config_from_hydra(cfg, use_part: Optional[str] = None) -> VQModelConfig:
    """
    Create a VQModelConfig from Hydra/OmegaConf configuration.
    
    Args:
        cfg: Hydra configuration object
        use_part: Override use_part if needed
        
    Returns:
        VQModelConfig instance
    """
    from omegaconf import OmegaConf
    
    # Convert to dict
    if hasattr(cfg, 'model'):
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    else:
        model_cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Flatten nested structure
    config_dict = {
        "use_part": use_part or cfg.get("use_part"),
        "codebook_dim": cfg.get("codebook_dim", 256),
        "nb_code": cfg.get("nb_code", 1024),
        "mu": cfg.get("mu", 0.99),
        **model_cfg.get("model", {}),
        **model_cfg.get("quantizer", {}),
        **model_cfg.get("loss", {}),
    }
    
    return VQModelConfig(**config_dict)

