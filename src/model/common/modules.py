import math

import torch
import torch.nn as nn


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x), None # no gate for GemmaRMSNorm, use unified gate for all RMSNorm modes


class GemmaRotaryEmbedding(nn.Module):
    """
    forces RoPE to use float32 for full accuracy

    https://github.com/huggingface/transformers/pull/29402
    https://github.com/huggingface/transformers/pull/29285
    """

    def __init__(self, dim, base=10000):
        super().__init__()

        self.dim = dim  # it is set to the head_dim
        self.base = (
            base  # should be tuned based on the max_seq_len, e.g., in action expert
        )

        # Calculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        # Multiply each theta by the position (which is the argument of the sin and cos functions)
        # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )
        # emb: [Batch_Size, Seq_Len, Head_Dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


class GemmaMLP(nn.Module):
    def __init__(self, config, use_quantize=False, use_lora=False):
        super().__init__()
        del use_quantize, use_lora
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, min_period: float = 0.004, max_period: float = 4.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.half_dim = dim // 2
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B, ) or (B, 1)
        if t.ndim == 2:
            t = t.squeeze(-1)
        fraction = torch.linspace(0.0, 1.0, self.half_dim, device=t.device, dtype=torch.float64)
        # period = min * (max/min)^fraction
        period = self.min_period * (self.max_period / self.min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        emb = t[:, None] * scaling_factor[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        emb = emb.to(t.dtype)
        return emb


class TimeEncoder(nn.Module):
    """Matching pi0.5 appendix"""

    def __init__(self, time_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(time_dim, time_dim)
        self.linear_2 = nn.Linear(time_dim, time_dim)
        self.nonlinearity = nn.SiLU()

    def forward(
        self,
        time: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # [Batch_Size, Time_Dim]
        emb = self.nonlinearity(self.linear_1(time))
        emb = self.nonlinearity(self.linear_2(emb))
        return emb


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    """

    def __init__(
        self,
        input_dim,
        embed_dim=256,
        scale=10,
    ):
        super(GaussianFourierFeatureTransform, self).__init__()
        # b is a fixed random projection matrix; keep it frozen via buffer.
        self.register_buffer(
            "b", torch.randn(input_dim, embed_dim) * scale, persistent=True
        )

    def forward(self, v: torch.FloatTensor) -> torch.FloatTensor:
        assert not self.b.requires_grad, "GaussianFourierFeatureTransform.b must be frozen"
        x_proj = torch.matmul(v, self.b) * (2 * math.pi)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim: int, dim_cond: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.to_gamma = nn.Sequential(
            nn.Linear(dim_cond, dim),
            nn.Sigmoid(),
        )
        self.to_beta = nn.Linear(dim_cond, dim, bias=False)

    def _norm(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(
        self, x: torch.FloatTensor, cond: torch.FloatTensor
    ) -> torch.FloatTensor:
        output = self._norm(x)
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        gamma = self.to_gamma(cond)
        beta = self.to_beta(cond)
        return output * gamma + beta, None # no gate for adaptive RMSNorm, use unified gate for all RMSNorm modes


class AdaptiveLayerscale(nn.Module):
    def __init__(
        self, dim: int, dim_cond: int, adaln_zero_bias_init_value: float = -2.0
    ):
        super().__init__()
        adaln_zero_gamma_linear = nn.Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = adaln_zero_gamma_linear

    def forward(
        self, x: torch.FloatTensor, cond: torch.FloatTensor
    ) -> torch.FloatTensor:
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        gamma = self.to_adaln_zero_gamma(cond)
        return x * gamma.sigmoid()


class AdaLNZero(nn.Module):
    def __init__(self, dim: int, dim_cond: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.modulation = nn.Linear(dim_cond, dim * 3)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def _norm(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.FloatTensor, cond: torch.FloatTensor) -> torch.FloatTensor:
        output = self._norm(x)
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        scale, shift, gate = self.modulation(cond).chunk(3, dim=-1)
        return output * (1.0 + scale) + shift, gate


class TimeEmbedding(nn.Module):
    """Wrapper for SinusoidalPosEmb + TimeEncoder to support Hydra _target_ instantiation."""

    def __init__(self, time_hidden_size: int, min_period: float = 0.004, max_period: float = 4.0):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPosEmb(time_hidden_size, min_period=min_period, max_period=max_period),
            TimeEncoder(time_hidden_size),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)
