import torch
import torch.nn as nn
from typing import Optional
from src.model.common.modules import GaussianFourierFeatureTransform

class ActionEncoder(nn.Module):
    """Matching pi0 appendix"""

    def __init__(self, action_dim: int, width: int, time_cond: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(action_dim, width)
        if time_cond:
            self.linear_2 = nn.Linear(2 * width, width)
        else:
            self.linear_2 = nn.Linear(width, width)
        self.nonlinearity = nn.SiLU()  # swish
        self.linear_3 = nn.Linear(width, width)
        self.time_cond = time_cond

    def forward(
        self,
        action: torch.FloatTensor,
        time_emb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Width]
        emb = self.linear_1(action)
        if self.time_cond:
            # repeat time embedding for seq_len
            # [Batch_Size, Seq_Len, Width]
            time_emb_full = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
            emb = torch.cat([time_emb_full, emb], dim=-1)
        emb = self.nonlinearity(self.linear_2(emb))
        emb = self.linear_3(emb)
        return emb


class FourierActionEncoder(nn.Module):
    """
    Action encoder with Gaussian Fourier feature embedding + MLP.
    """

    def __init__(
        self,
        action_dim: int,
        width: int,
        time_cond: bool = False,
        mlp_depth: int = 2,
        fourier_embed_dim: int = 1024,
        fourier_scale: float = 10.0,
        final_layer_norm: bool = True,
        time_emb_dim: Optional[int] = None,
    ):
        super().__init__()
        assert mlp_depth >= 0, "mlp_depth must be >= 0"
        self.time_cond = time_cond
        self.fourier = GaussianFourierFeatureTransform(
            action_dim, embed_dim=fourier_embed_dim, scale=fourier_scale
        )

        if self.time_cond:
            if time_emb_dim is None:
                raise ValueError("time_emb_dim must be provided when time_cond=True")
        else:
            time_emb_dim = 0

        mlp_input_dim = 2 * fourier_embed_dim + time_emb_dim
        if mlp_depth == 0 and mlp_input_dim != width:
            raise ValueError("mlp_depth must not be 0 if mlp_input_dim != width")
        if mlp_depth == 0: 
            self.mlp = nn.Identity()
            self.final_layer_norm = None
        else:
            layers = []
            for layer_idx in range(mlp_depth):
                in_dim = mlp_input_dim if layer_idx == 0 else width
                layers.append(nn.Linear(in_dim, width))
                if layer_idx < mlp_depth - 1:
                    layers.append(nn.SiLU())
            self.mlp = nn.Sequential(*layers)
            self.final_layer_norm = nn.LayerNorm(width) if final_layer_norm else None

    def forward(
        self,
        action: torch.FloatTensor,
        time_emb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            action: [Batch_Size, Seq_Len, Action_Dim]
            time_emb: [Batch_Size, Time_Dim]
        Returns:
            emb: [Batch_Size, Seq_Len, Width]
        """
        # Fourier features: [Batch_Size, Seq_Len, 2 * fourier_embed_dim]
        emb = self.fourier(action)
        if self.time_cond:
            if time_emb is None:
                raise ValueError("time_emb must be provided when time_cond=True")
            if time_emb.ndim == 2:
                time_emb_full = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
            else:
                time_emb_full = time_emb
            emb = torch.cat([time_emb_full, emb], dim=-1)
        emb = self.mlp(emb)
        if self.final_layer_norm is not None:
            emb = self.final_layer_norm(emb)
        return emb


class LatentConditionProjector(nn.Module):
    """
    Latent condition projector with MLP.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 3,
        final_layer_norm: bool = True,
    ):
        super().__init__()
        assert depth >= 0, "depth must be >= 0"
        if depth == 0:
            self.mlp = nn.Identity()
            self.final_layer_norm = None
        else:
            layers = []
            for layer_idx in range(depth):
                in_dim = input_dim if layer_idx == 0 else output_dim
                layers.append(nn.Linear(in_dim, output_dim))
                if layer_idx < depth - 1:
                    layers.append(nn.SiLU())
            self.mlp = nn.Sequential(*layers)
            self.final_layer_norm = nn.LayerNorm(output_dim) if final_layer_norm else None

    def forward(self, latent: torch.FloatTensor) -> torch.FloatTensor:
        emb = self.mlp(latent)
        if self.final_layer_norm is not None:
            emb = self.final_layer_norm(emb)
        return emb

        