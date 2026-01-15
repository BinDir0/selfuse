# https://github.com/BeingBeyond/Being-H0/beingvla/models/motion/m2m/tokenizer/encdec.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple


# https://github.com/facebookresearch/encodec/blob/main/encodec/modules/conv.py
def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """
    Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


class Swish(nn.Module):
    """Swish activation function (x * sigmoid(x))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class CausalConv1d(nn.Module):
    """Causal 1D convolutional layer"""
    def __init__(
        self, in_channels: int, out_channels: int, 
        kernel_size: int, stride: int = 1, dilation: int = 1, 
        causal: bool = False, pad_mode: str = 'replicate'
    ):
        """
        A Conv1d layer with causal padding. 
        The output length is always ceil(T / stride), which is not the same as vanilla Conv1d.
        For causal convolution, we should padding the input tensor to the left to ensure the causal property.
        For integrity, we also pad the input tensor to the right to ensure the output length is ceil(T / stride).
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad the input tensor with zeros to the left to ensure the causal property
        # the output length is ceil(T / stride)
        B, C, T = x.shape
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        dilation = self.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            # Left padding for causal
            x = F.pad(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = F.pad(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class ResConv1DBlock(nn.Module):
    """Residual 1D convolutional block with optional normalization."""
    NORM_MAP = {
        "LN": nn.LayerNorm,
        "GN": lambda n_in: nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True),
        "BN": lambda n_in: nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True),
    }
    
    ACTIVATION_MAP = {
        "relu": nn.ReLU,
        "silu": Swish,
        "gelu": nn.GELU
    }
    
    def __init__(
        self, 
        n_in: int, 
        n_state: int, 
        dilation: int = 1, 
        activation: Literal['relu', 'silu', 'gelu'] = 'silu',
        norm: Optional[Literal['LN', 'GN', 'BN']] = None,
        causal: bool = False
    ):
        super().__init__()
  
        self.norm = norm
        self.norm1 = self._create_norm_layer(norm, n_in)
        self.norm2 = self._create_norm_layer(norm, n_in)

        self.conv1 = CausalConv1d(n_in, n_state, 3, 1, dilation, causal=causal)
        self.conv2 = CausalConv1d(n_state, n_in, 1, 1, causal=causal)

        self.activation1 = self.ACTIVATION_MAP[activation]()
        self.activation2 = self.ACTIVATION_MAP[activation]()
      
    def _create_norm_layer(self, norm, n_in):
        return self.NORM_MAP.get(norm, nn.Identity)(n_in)
 
    def forward(self, x):
        x_orig = x

        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)  
        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)
        x = self.conv2(x)

        return x + x_orig


class Resnet1D(nn.Module):
    """1D ResNet with configurable dilation rates."""
    def __init__(
        self, 
        n_in: int, 
        n_depth: int, 
        dilation_growth_rate: int = 1, 
        reverse_dilation: bool = True,
        activation: str = 'silu',
        norm: Optional[str] = None,
        causal: bool = False
    ):
        super().__init__()
        
        blocks = [
            ResConv1DBlock(
                n_in, n_in,
                dilation=dilation_growth_rate ** depth,
                activation=activation,
                norm=norm,
                causal=causal
            ) for depth in range(n_depth)
        ]
        
        self.model = nn.Sequential(*(blocks[::-1] if reverse_dilation else blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Encoder(nn.Module):
    """1D convolutional encoder with downsampling."""
    
    def __init__(
        self,
        input_emb_width: int = 3,
        output_emb_width: int = 512,
        down_t: int = 3,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = 'silu',
        norm: Optional[str] = None,
        num_conv_layers = 1, 
        causal: bool = False
    ):
        super().__init__()
        
        filter_t = stride_t * 2 + 1

        blocks = []
        blocks.append(CausalConv1d(input_emb_width, width, 3, 1, 1, causal=causal))
        blocks.append(Swish())
        for _ in range(num_conv_layers-1):
            blocks.append(CausalConv1d(width, width, 3, 1, 1, causal=causal))
            blocks.append(Swish())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                CausalConv1d(input_dim, width, filter_t, stride_t, causal=causal), # B, C, T -> B, C, ceil(T/stride_t)
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm, causal=causal),
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, output_emb_width, 3, 1, 1, causal=causal))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

# Do not use causal decoder for now
class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width = 3,
        output_emb_width = 512,
        down_t = 3,
        stride_t = 2,
        width = 512,
        depth = 3,
        dilation_growth_rate = 3, 
        activation='relu',
        norm=None,
        num_conv_layers=1,
        causal=False
    ):
        super().__init__()

        blocks = []
        blocks.append(CausalConv1d(output_emb_width, width, 3, 1, 1, causal=causal))
        blocks.append(Swish())
        for _ in range(num_conv_layers-1):
            blocks.append(CausalConv1d(width, width, 3, 1, 1, causal=causal))
            blocks.append(Swish())

        for _ in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm, causal=causal),
                nn.Upsample(scale_factor=stride_t, mode='nearest'),
                CausalConv1d(width, out_dim, 3, 1, 1, causal=causal)
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, width, 3, 1, 1, causal=causal))
        blocks.append(Swish())
        blocks.append(CausalConv1d(width, input_emb_width, 3, 1, 1, causal=causal))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    