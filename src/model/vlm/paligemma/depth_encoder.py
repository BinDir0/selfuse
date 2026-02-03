import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.model.common.lora import get_layer


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
    ):
        super().__init__()
        layer = get_layer(
            use_quantize,
            use_lora,
            **config.lora if use_lora else {},
        )
        self.linear = layer(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states


class DINOv2DepthEncoder(nn.Module):
    """
    DINOv2-based depth image encoder for PaliGemma.
    
    Processes depth images (single channel), normalizes them to match ImageNet statistics,
    and extracts patch tokens compatible with SigLIP vision encoder output.
    """
    
    def __init__(
        self,
        config: dict,
    ):
        """
        Initialize DINOv2 depth encoder.
        
        Args:
            config: Config dictionary containing:
                model_name: DINOv2 model name (e.g., 'dinov2_vitb14', 'dinov2_vits14')
                freeze_backbone: Whether to freeze DINOv2 backbone weights
                image_size: Input image size (default 518 for DINOv2)
                patch_size: Patch size (default 14 for DINOv2)
                output_dim: Output dimension for depth tokens
        """
        super().__init__()
        
        # Size check: DINOv2 requires input size to be a multiple of patch_size
        model_name = config.get("model_name", "dinov2_vitb14")
        image_size = config.get("image_size", 518)
        patch_size = config.get("patch_size", 14)
        output_dim = config.get("output_dim", 768)
        if image_size % patch_size != 0:
            print(f"Warning: image_size {image_size} is not divisible by patch_size {patch_size}.")
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth_seq_len = (image_size // patch_size) ** 2
        self.output_dim = output_dim
        
        # Load model
        try:
            # Note: Requires torch.hub support
            if "dinov2_repo_path" in config: 
                self.dinov2 = torch.hub.load(
                    config["dinov2_repo_path"], model_name, source='local', pretrained=False
                )
            else: 
                self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
            if "dinov2_model_path" in config:
                self.dinov2.load_state_dict(torch.load(config["dinov2_model_path"]))
                print(f"Loaded DINOv2 model from {config['dinov2_model_path']}")
        except Exception as e:
            raise ImportError(f"Failed to load DINOv2. Error: {e}")
        
        self.dinov2_hidden_size = self.dinov2.embed_dim
        
        # Freeze backbone
        freeze_backbone = config.get("freeze_backbone", True)
        if freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            # Ensure the model is in eval mode (disables Dropout/BatchNorm updates)
            self.dinov2.eval()
        
        # Store whether backbone is frozen (for dtype handling)
        self._freeze_backbone = freeze_backbone 

    def _convert_dtype(self, dtype: torch.dtype):
        """Convert DINOv2 model to specified dtype for efficient training."""
        self.dinov2 = self.dinov2.to(dtype)
        return self
    
    @torch.compile(mode="default")
    def forward(self, depth_images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv2 depth encoder.
        
        Args:
            depth_images: Depth images, shape [B, C, H, W].
                          Values MUST be in range [0, 1].
        
        Returns:
            depth_tokens: Depth patch tokens, shape [B, num_patches, output_dim]
        """
        # forward_features returns a dict containing patch tokens
        outputs = self.dinov2.forward_features(depth_images)
        
        # Extract Patch Tokens (ignoring the CLS token)
        patch_tokens = outputs['x_norm_patchtokens']
        
        return patch_tokens

class ResNetViTDepthEncoder(nn.Module):
    """
    ResNet-ViT-based depth image encoder for PaliGemma.
    
    Processes depth images (single channel), normalizes them to match ImageNet statistics,
    and extracts patch tokens compatible with SigLIP vision encoder output.
    """
    
    def __init__(
        self,
        config: dict,
    ):
        raise NotImplementedError("ResNetViTDepthEncoder is not implemented yet.")

    def forward(self, depth_images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ResNetViTDepthEncoder is not implemented yet.")