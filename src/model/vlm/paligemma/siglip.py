import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",  # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        (
            _,
            _,
            height,
            width,
        ) = pixel_values.shape  # [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5  # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        layer = get_layer(
            use_quantize,
            use_lora,
            **config.lora if use_lora else {},
        )
        self.k_proj = layer(self.embed_dim, self.embed_dim)
        self.v_proj = layer(self.embed_dim, self.embed_dim)
        self.q_proj = layer(self.embed_dim, self.embed_dim)
        self.out_proj = layer(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_weights = None

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
    ):
        super().__init__()
        self.config = config
        layer = get_layer(
            use_quantize,
            use_lora,
            **config.lora if use_lora else {},
        )
        self.fc1 = layer(config.hidden_size, config.intermediate_size)
        self.fc2 = layer(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
        use_temporal_attention: bool = False,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.use_temporal_attention = use_temporal_attention
        if self.use_temporal_attention:
            self.temporal_residual_scale = nn.Parameter(torch.zeros(()))
        self.self_attn = SiglipAttention(
            config,
            use_quantize=use_quantize,
            use_lora=use_lora,
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(
            config,
            use_quantize=use_quantize,
            use_lora=use_lora,
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        temporal_pos_emb: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.use_temporal_attention:
            temporal_hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', b=batch_size)
            temporal_residual = temporal_hidden_states
            temporal_hidden_states = temporal_hidden_states + temporal_pos_emb.unsqueeze(0)
            temporal_hidden_states = self.layer_norm1(temporal_hidden_states)
            temporal_hidden_states, _ = self.self_attn(temporal_hidden_states, is_causal=True)
            hidden_states = rearrange(
                temporal_residual + self.temporal_residual_scale * temporal_hidden_states,
                '(b n) t d -> (b t) n d',
                b=batch_size,
            )

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, is_causal=is_causal)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class SiglipEncoder(nn.Module):
    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
        compress_to_current: bool = True,
    ):
        super().__init__()
        self.config = config
        self.compress_to_current = compress_to_current
        use_mem = bool(getattr(config, 'use_mem', False))
        temporal_interval = int(getattr(config, 'temporal_interval', 4))
        temporal_layer_indices = tuple(
            idx for idx in range(config.num_hidden_layers)
            if use_mem and temporal_interval > 0 and (idx + 1) % temporal_interval == 0
        )
        self.layers = nn.ModuleList(
            [
                SiglipEncoderLayer(
                    config,
                    use_quantize=use_quantize,
                    use_lora=use_lora,
                    use_temporal_attention=(idx in temporal_layer_indices),
                )
                for idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        batch_size: int,
        temporal_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, batch_size, temporal_pos_emb)
        if self.compress_to_current:
            hidden_states = rearrange(hidden_states, '(b t) n d -> b t n d', b=batch_size)[:, -1].clone()
        return hidden_states


def build_sinusoidal_temporal_pos_emb(positions: torch.Tensor, dim: int) -> torch.Tensor:
    positions = positions.to(dtype=torch.float32).unsqueeze(1)
    half_dim = dim // 2
    div_term = torch.exp(
        torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / max(half_dim, 1))
    )
    emb = torch.zeros(positions.shape[0], dim, dtype=torch.float32)
    if half_dim > 0:
        sinusoid = positions * div_term
        emb[:, :half_dim] = torch.sin(sinusoid)
        emb[:, half_dim : 2 * half_dim] = torch.cos(sinusoid) - 1.0
    return emb


class SiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
        compress_to_current: bool = True,
    ):
        super().__init__()
        self.config = config
        self.use_mem = bool(getattr(config, 'use_mem', False))
        self.temporal_max_frames = int(getattr(config, 'temporal_max_frames', 18))
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(
            config,
            use_quantize=use_quantize,
            use_lora=use_lora,
            compress_to_current=(compress_to_current and self.use_mem),
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        temporal_positions = torch.arange(1 - self.temporal_max_frames, 1, dtype=torch.float32)
        temporal_pos_emb = build_sinusoidal_temporal_pos_emb(temporal_positions, embed_dim)
        self.register_buffer('temporal_pos_emb', temporal_pos_emb, persistent=False)

    def encode_spatial(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states, pixel_values.shape[0], self.temporal_pos_emb[-1:])
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states

    def encode_mem(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f'MEM expects 5D input, got shape {tuple(pixel_values.shape)}')

        pixel_values = pixel_values[:, -self.temporal_max_frames:]
        batch_size, num_frames = pixel_values.shape[:2]
        hidden_states = self.embeddings(rearrange(pixel_values, 'b t c h w -> (b t) c h w'))
        hidden_states = self.encoder(hidden_states, batch_size, self.temporal_pos_emb[-num_frames:])
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.use_mem:
            if pixel_values.ndim == 4:
                pixel_values = pixel_values.unsqueeze(1)
            if pixel_values.ndim != 5:
                raise ValueError(f'MEM expects 5D input, got shape {tuple(pixel_values.shape)}')
            return self.encode_mem(pixel_values)

        if pixel_values.ndim not in (4, 5):
            raise ValueError(f'Expected 4D or 5D input, got shape {tuple(pixel_values.shape)}')
        if pixel_values.ndim == 4:
            return self.encode_spatial(pixel_values)

        batch_size, num_frames = pixel_values.shape[:2]
        hidden_states = self.encode_spatial(rearrange(pixel_values, 'b t c h w -> (b t) c h w'))
        return rearrange(hidden_states, '(b t) n d -> b (t n) d', b=batch_size, t=num_frames)


class SiglipVisionModel(nn.Module):
    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
        compress_to_current: bool = True,
    ):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(
            config,
            use_quantize=use_quantize,
            use_lora=use_lora,
            compress_to_current=compress_to_current,
        )

    @torch.compile(mode="default")
    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
