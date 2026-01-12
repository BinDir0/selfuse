import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DINOv2DepthEncoder(nn.Module):
    """
    DINOv2-based depth image encoder for PaliGemma.
    
    Processes depth images (single channel), normalizes them to match ImageNet statistics,
    and extracts patch tokens compatible with SigLIP vision encoder output.
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        image_size: int = 518,
        patch_size: int = 14,
        output_dim: Optional[int] = None,
        freeze_backbone: bool = True,
    ):
        """
        Initialize DINOv2 depth encoder.
        
        Args:
            model_name: DINOv2 model name (e.g., 'dinov2_vitb14', 'dinov2_vits14')
            image_size: Input image size (default 518 for DINOv2)
            patch_size: Patch size (default 14 for DINOv2)
            output_dim: Output dimension for depth tokens. If None, uses DINOv2's hidden_size
            freeze_backbone: Whether to freeze DINOv2 backbone weights
        """
        super().__init__()
        
        # Size check: DINOv2 requires input size to be a multiple of patch_size
        if image_size % patch_size != 0:
            print(f"Warning: image_size {image_size} is not divisible by patch_size {patch_size}.")
        
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Load model
        try:
            # Note: Requires torch.hub support
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        except Exception as e:
            raise ImportError(f"Failed to load DINOv2. Error: {e}")
        
        self.dinov2_hidden_size = self.dinov2.embed_dim
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            # Ensure the model is in eval mode (disables Dropout/BatchNorm updates)
            self.dinov2.eval() 
        
        # Projection layer
        self.output_dim = output_dim if output_dim is not None else self.dinov2_hidden_size
        if self.output_dim != self.dinov2_hidden_size:
            self.projection = nn.Linear(self.dinov2_hidden_size, self.output_dim)
        else:
            self.projection = nn.Identity()
            
        # Define ImageNet normalization parameters
        # Registered as buffers so they automatically move to the correct device (CPU/GPU) with the model
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, depth_images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv2 depth encoder.
        
        Args:
            depth_images: Depth images, shape [B, 1, H, W] or [B, H, W].
                          Values MUST be in range [0, 1].
        
        Returns:
            depth_tokens: Depth patch tokens, shape [B, num_patches, output_dim]
        """
        # Adjust dimensions: [B, H, W] -> [B, 1, H, W]
        if depth_images.dim() == 3:
            depth_images = depth_images.unsqueeze(1)
        
        # Resize images
        # Resize before converting to 3 channels to save slightly on memory/computation
        if depth_images.shape[-2:] != (self.image_size, self.image_size):
            depth_images = F.interpolate(
                depth_images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )

        # Repeat channels: [B, 1, H, W] -> [B, 3, H, W]
        # DINOv2 expects RGB input
        if depth_images.shape[1] == 1:
            x = depth_images.repeat(1, 3, 1, 1)
        else:
            x = depth_images

        # Normalization 
        # DINOv2 expects inputs normalized with ImageNet statistics
        x = (x - self.imagenet_mean) / self.imagenet_std

        # Model forward pass
        # Use context manager to handle grad requirements based on backbone freezing
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.dinov2.parameters())):
            # forward_features returns a dict containing patch tokens
            outputs = self.dinov2.forward_features(x)
        
        # Extract Patch Tokens (ignoring the CLS token)
        patch_tokens = outputs['x_norm_patchtokens']
        
        # Projection to target dimension
        depth_tokens = self.projection(patch_tokens)
        
        return depth_tokens