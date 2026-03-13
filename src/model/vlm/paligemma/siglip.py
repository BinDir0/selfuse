import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
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

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            is_causal=is_causal,
        )
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                SiglipEncoderLayer(
                    config,
                    use_quantize=use_quantize,
                    use_lora=use_lora,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    # Ignore copy
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


def _build_sinusoidal_temporal_pos_emb(max_frames: int, dim: int) -> torch.Tensor:
    position = torch.arange(max_frames, dtype=torch.float32).unsqueeze(1)
    half_dim = dim // 2
    div_term = torch.exp(
        torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / max(half_dim, 1))
    )
    emb = torch.zeros(max_frames, dim, dtype=torch.float32)
    if half_dim > 0:
        sinusoid = position * div_term
        emb[:, :half_dim] = torch.sin(sinusoid)
        emb[:, half_dim : 2 * half_dim] = torch.cos(sinusoid) - 1.0
    return emb


class SiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(
            config,
            use_quantize=use_quantize,
            use_lora=use_lora,
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_mem = bool(getattr(config, "use_mem", False))
        self.temporal_max_frames = int(getattr(config, "temporal_max_frames", 18))
        self.temporal_interval = int(getattr(config, "temporal_interval", 4))

        if self.use_mem:
            temporal_pos_emb = _build_sinusoidal_temporal_pos_emb(
                self.temporal_max_frames,
                embed_dim,
            )
            self.register_buffer(
                "temporal_pos_emb",
                temporal_pos_emb,
                persistent=False,
            )

    def _encode_spatial(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(inputs_embeds=hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states

    def _encode_mem(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        MEM-style short-term memory encoding.

        Every layer runs the full spatial encoder layer (LN1 -> attn -> res -> LN2 -> MLP -> res).
        Every ``temporal_interval``-th layer **additionally** applies a temporal
        attention branch (LN1 -> attn only, no MLP) using the *same* layer weights,
        added back via a residual connection.  Temporal pos-emb is injected only
        into the normed input of the temporal attention so it does not pollute the
        main residual stream.
        """
        batch_size, num_frames, channels, height, width = pixel_values.shape
        if num_frames > self.temporal_max_frames:
            pixel_values = pixel_values[:, -self.temporal_max_frames :, ...]
            num_frames = self.temporal_max_frames

        if num_frames == 1:
            return self._encode_spatial(pixel_values[:, 0, ...])

        flat_pixel_values = pixel_values.reshape(
            batch_size * num_frames, channels, height, width
        )
        hidden_states = self.embeddings(flat_pixel_values)
        num_patches = hidden_states.shape[1]
        embed_dim = hidden_states.shape[2]
        # T in [-k, 0], 0 is current frame, so need to flip the temporal pos embedding
        temporal_pos_emb = self.temporal_pos_emb[:num_frames].flip(0).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        last_temporal_layer_idx = max(
            i for i in range(len(self.encoder.layers)) if (i + 1) % self.temporal_interval == 0
        )
        for layer_idx, encoder_layer in enumerate(self.encoder.layers):

            # Every N-th layer, ADDITIONALLY add temporal attention (LN + attn, no MLP)
            if (layer_idx + 1) % self.temporal_interval == 0:
                # reshape: [B*T, n, D] -> [B*n, T, D]
                ht = hidden_states.reshape(
                    batch_size, num_frames, num_patches, embed_dim,
                )
                ht = ht.permute(0, 2, 1, 3).reshape(
                    batch_size * num_patches, num_frames, embed_dim,
                )

                # Temporal attention using the same layer's LN1 + self_attn
                residual = ht
                ht_with_pos = ht + temporal_pos_emb.unsqueeze(0)
                normed = encoder_layer.layer_norm1(ht_with_pos)
                attn_out, _ = encoder_layer.self_attn(normed, is_causal=True)
                ht = residual + attn_out

                # reshape back: [B*n, T, D] -> [B*T, n, D]
                hidden_states = ht.reshape(
                    batch_size, num_patches, num_frames, embed_dim,
                )
                hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(
                    batch_size * num_frames, num_patches, embed_dim,
                )
            # ALWAYS run the full spatial layer (LN + attn + MLP)
            hidden_states = encoder_layer(hidden_states)  # [B*T, n, D]
            # dropping representations for all patches from past timesteps to speed up the computation
            if layer_idx == last_temporal_layer_idx and num_frames > 1:
                hidden_states = hidden_states.reshape(batch_size, num_frames, num_patches, embed_dim)
                hidden_states = hidden_states[:, -1:, :, :] 
                num_frames = 1
                hidden_states = hidden_states.reshape(batch_size * num_frames, num_patches, embed_dim)

        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size, num_frames, num_patches, embed_dim
        )
        return hidden_states[:, -1, :, :]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 4:
            return self._encode_spatial(pixel_values)

        if pixel_values.ndim != 5:
            raise ValueError(
                f"Expected 4D or 5D input, got shape {tuple(pixel_values.shape)}"
            )

        if self.use_mem:
            return self._encode_mem(pixel_values)

        batch_size, num_frames, channels, height, width = pixel_values.shape
        flat_pixel_values = pixel_values.reshape(batch_size * num_frames, channels, height, width)
        hidden_states = self._encode_spatial(flat_pixel_values)
        return hidden_states.reshape(batch_size, -1, hidden_states.shape[-1])


class SiglipVisionModel(nn.Module):
    def __init__(
        self,
        config,
        use_quantize: bool = False,
        use_lora: bool = False,
    ):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(
            config,
            use_quantize=use_quantize,
            use_lora=use_lora,
        )

    @torch.compile(mode="default")
    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
