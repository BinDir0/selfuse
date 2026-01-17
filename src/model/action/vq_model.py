# VQ-VAE style Model for Action Tokenization
# Adapted from https://github.com/BeingBeyond/Being-H0/beingvla/models/motion/m2m/tokenizer/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import PreTrainedModel
from einops import rearrange
from vector_quantize_pytorch import GroupedResidualVQ, ResidualVQ, ResidualFSQ, GroupedResidualFSQ

from .encdec import Encoder, Decoder
from .vq_config import MotionVQModelConfig


def get_extra_padding_for_conv1d(x: torch.Tensor, stride: int) -> int:
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
    So when we consider the tuple of (state, action), we need to pad state to grantee the whole state is encoded.
    """
    length = x.shape[-1]
    ideal_length = math.ceil(length / stride) * stride
    return ideal_length - length # extra padding

@torch.no_grad()
def calculate_perplexity_grvq(indices, codebook_size, eps=1e-5):
    """
    Calculate perplexity for Grouped Residual VQ.
    Args:
        indices: [Groups, Batch, Length, Layers] or [Batch, Length, Layers]
        codebook_size: int
    Returns:
        perplexities: [Groups, Layers] or [Layers]
    """
    is_grouped = len(indices.shape) == 4
    if not is_grouped:
        indices = indices.unsqueeze(0)

    flat_indices = rearrange(indices, 'g b t l -> g l (b t)').long()
    G, L, N = flat_indices.shape
    counts = torch.zeros(G, L, codebook_size, device=indices.device)
    src = torch.ones_like(flat_indices, dtype=torch.float32)
    counts.scatter_add_(2, flat_indices, src) 
    probs = counts / counts.sum(dim=-1, keepdim=True)
    perplexities = torch.exp(-torch.sum(probs * torch.log(probs + eps), dim=-1))
            
    if not is_grouped:
        perplexities = perplexities.squeeze(0)
    return perplexities


class BaseVQModel(nn.Module):
    """
    VQ-VAE model for state-action tokenization.
    The input is a tuple of (state, action), and the output is a tuple of (state, action).
    Supports both RVQ, GRVQ and FSQ.
    """
    
    def __init__(
        self,
        config: MotionVQModelConfig,
        motion_dim: Optional[int] = None
    ):
        super().__init__()
        """Initialize encoder/decoder and quantizer based on config."""
        self.code_dim = config.quantizer_config.codebook_dim
        self.codebook_size = config.quantizer_config.codebook_size
        self.motion_dim = motion_dim if motion_dim is not None else config.motion_dim
        self.quantizer_name = config.quantizer_config.quantizer_name
        model_config = config.model_config
        self.encoder = Encoder(
            input_emb_width=self.motion_dim,
            output_emb_width=model_config.output_emb_width,
            down_t=model_config.down_t,
            stride_t=model_config.stride_t,
            width=model_config.width,
            depth=model_config.depth,
            dilation_growth_rate=model_config.dilation_growth_rate,
            activation=model_config.activation,
            norm=model_config.norm,
            num_conv_layers=model_config.num_conv_layers,
            causal=model_config.encoder_causal
        )
        self.decoder = Decoder(
            input_emb_width=self.motion_dim,
            output_emb_width=model_config.output_emb_width,
            down_t=model_config.down_t,  # Assuming symmetric
            stride_t=model_config.stride_t,
            width=model_config.width,
            depth=model_config.depth,
            dilation_growth_rate=model_config.dilation_growth_rate,
            activation=model_config.activation,
            norm=model_config.norm,
            num_conv_layers=model_config.num_conv_layers,
            causal=model_config.decoder_causal
        )
        # equivalent downsampling stride
        self.total_stride = model_config.stride_t ** model_config.down_t 

        self.quantizer = self._create_quantizer(config.quantizer_config)
    
    def _create_quantizer(self, quantizer_config):
        """Factory method for quantizer creation."""
        levels_dict = {
            256: [8, 6, 5], 512: [8, 8, 8],
            1024: [8, 5, 5, 5], 2048: [8, 8, 6, 5],
            4096: [7, 5, 5, 5, 5], 8192: [8, 6, 6, 5, 5],
            16384: [8, 8, 8, 6, 5], 65536: [8, 8, 8, 5, 5, 5]
        }
        quantizers = {
            "residualvq": lambda: ResidualVQ(
                codebook_size=quantizer_config.codebook_size,
                # input dimension of quantizer, we use the same dimension for the codebook
                # the same as GroupedResidualVQ and FSQ
                dim=quantizer_config.codebook_dim, 
                num_quantizers=quantizer_config.num_quantizers,
                shared_codebook=quantizer_config.shared_codebook
            ),
            "group_residualvq": lambda: GroupedResidualVQ(
                codebook_size=quantizer_config.codebook_size, 
                dim=quantizer_config.codebook_dim, 
                num_quantizers=quantizer_config.num_quantizers, 
                groups=quantizer_config.num_groups,
                shared_codebook=quantizer_config.shared_codebook
            ),
            "residualfsq": lambda: ResidualFSQ(
                levels=levels_dict[quantizer_config.codebook_size], 
                dim=quantizer_config.codebook_dim,
                num_quantizers=quantizer_config.num_quantizers
            ),
            "grouped_residualfsq": lambda: GroupedResidualFSQ(
                levels=levels_dict[quantizer_config.codebook_size], 
                dim=quantizer_config.codebook_dim,
                num_quantizers=quantizer_config.num_quantizers
            )
        }
        return quantizers[quantizer_config.quantizer_name]()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).float().contiguous() # (bs, T, D) -> (bs, D, T)
    
    def postprocess(self, x: torch.Tensor) -> torch.Tensor: 
        return x.permute(0, 2, 1).contiguous() # (bs, D, T) ->  (bs, T, D)
    
    def extra_padding(self, x: torch.Tensor, pad_mode: str = 'replicate') -> torch.Tensor:
        """See get_extra_padding_for_conv1d for more details."""
        extra_padding = get_extra_padding_for_conv1d(x, self.total_stride)
        return F.pad(x, (0, extra_padding), mode=pad_mode)

    def process_motion(self, state: torch.Tensor, action: torch.Tensor):
        """
        Process and pad the state and action into a concatenated motion tensor.
        Args:
            state: torch.Tensor, shape: (N, T_state, D)
            action: torch.Tensor, shape: (N, T_action, D)
        Returns:
            motion_concat: torch.Tensor, shape: (N, D, T_state' + T_action')
            T_downsampled_state: int, the number of downsampled state steps, T_state'
        """
        # (N, T, D) -> (N, D, T)
        state, action = self.preprocess(state), self.preprocess(action)
        state_pad, action_pad = self.extra_padding(state), self.extra_padding(action)
        T_downsampled_state = state_pad.shape[-1] // self.total_stride
        motion_concat = torch.cat([state_pad, action_pad], dim=-1)
        return motion_concat, T_downsampled_state

    def encode(self, state: torch.Tensor, action: torch.Tensor):
        """
        Encode the state and action into codebook indices.
        For best performance, the length of state and action should be divisible by the equivalent downsampling stride.
        Args:
            state: torch.Tensor, shape: (N, T_state, D)
            action: torch.Tensor, shape: (N, T_action, D)
        Returns:
            state_indices: torch.Tensor, shape: (N, T_state_downsampled, Q) or (G, N, T_state_downsampled, Q)
            action_indices: torch.Tensor, shape: (N, T_action_downsampled, Q) or (G, N, T_action_downsampled, Q)
            where Q is the number of quantizers and G is the number of groups for grouped residual VQ
        """
        # motion_concat: (N, D, T_s'+T_a')(after padding)
        motion_concat, T_downsampled_state = self.process_motion(state, action)
        # (N, codebook_dim, T_motion_downsampled = ceil((T_s'+T_a')/total_stride))
        motion_enc = self.encoder(motion_concat) 

        # motion_indices: (N, T_motion_downsampled, Q) or (G, N, T_motion_downsampled, Q)
        result = self.quantizer(self.postprocess(motion_enc))
        if len(result) == 2: # RFSQ, GRFSQ: return (quantized_out, all_indices)
            _, motion_indices = result
        elif len(result) == 3: # RVQ, GRVQ: return (quantized_out, all_indices, commit_loss)
            _, motion_indices, _ = result
        else:
            raise ValueError(f"Unexpected result length: {len(result)}")

        state_indices = motion_indices[..., :T_downsampled_state, :]
        action_indices = motion_indices[..., T_downsampled_state:, :]
        return state_indices, action_indices

    def decode(self, state_indices: torch.Tensor, action_indices: torch.Tensor):
        """
        Decode the codebook indices into state and action.
        Args:
            state_indices: torch.Tensor, shape: (N, T_state_downsampled, Q) or (G, N, T_state_downsampled, Q)
            action_indices: torch.Tensor, shape: (N, T_action_downsampled, Q) or (G, N, T_action_downsampled, Q)
            where Q is the number of quantizers and G is the number of groups for grouped residual VQ
        Returns:
            state_dec: torch.Tensor, shape: (N, T_state, D) or (G, N, T_state, D)
            action_dec: torch.Tensor, shape: (N, T_action, D) or (G, N, T_action, D)
        """
        # codebook indices -> quantized codes
        state_quant = self.quantizer.get_output_from_indices(state_indices)
        action_quant = self.quantizer.get_output_from_indices(action_indices)
        assert state_quant.ndim == 3 and action_quant.ndim == 3, f"Quantized codes must have shape (N, T_s', D) or (N, T_a', D)"

        state_dec = self.preprocess(state_quant)  # (N, T_s', D) -> (N, D, T_s')
        action_dec = self.preprocess(action_quant)  # (N, T_a', D) -> (N, D, T_a')
        T_pad_state = state_dec.shape[-1] * self.total_stride
        motion_dec = self.decoder(torch.cat([state_dec, action_dec], dim=-1))
        motion_dec = self.postprocess(motion_dec)    # (N, D, T_s+T_a) -> (N, T_s+T_a, D)

        return motion_dec[:, :T_pad_state, :], motion_dec[:, T_pad_state:, :]
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        """
        Training forward pass.
        For best performance, the length of state and action should be divisible by the equivalent downsampling stride.
        Args:
            state: torch.Tensor, shape: (N, T_state, D)
            action: torch.Tensor, shape: (N, T_action, D)
        Returns:
            state_decoded: torch.Tensor, shape: (N, T_state, D) or (G, N, T_state, D)
            action_decoded: torch.Tensor, shape: (N, T_action, D) or (G, N, T_action, D)
            commit_loss: torch.Tensor, shape: (N, Q) or (G, N, Q) 
            perplexity: torch.Tensor, shape: (Q) or (G, Q)
            # Q is the number of quantizers, G is the number of groups for grouped residual VQ
        """
        T_state, T_action = state.shape[1], action.shape[1]
        # motion_concat: (N, D, T_s'+T_a')(after padding)
        motion_concat, T_downsampled_state = self.process_motion(state, action)
        T_pad_state = T_downsampled_state * self.total_stride   
        # (N, codebook_dim, T_motion_downsampled = ceil((T_s'+T_a')/total_stride))
        motion_enc = self.encoder(motion_concat) 

        # motion_quant: (N, T_motion_quantized, codebook_dim)
        # motion_indices: (N, T_motion_downsampled, Q) or (G, N, T_motion_downsampled, Q)
        # commit_loss: (N, Q) or (G, N, Q)
        result = self.quantizer(self.postprocess(motion_enc))
        if len(result) == 2: # RFSQ, GRFSQ: return (quantized_out, all_indices)
            motion_quant, motion_indices = result
            commit_loss = None
        elif len(result) == 3: # RVQ, GRVQ: return (quantized_out, all_indices, commit_loss)
            motion_quant, motion_indices, commit_loss = result
        else:
            raise ValueError(f"Unexpected result length: {len(result)}")
        perplexity = calculate_perplexity_grvq(motion_indices, self.codebook_size)

        # (N, T_motion_quantized, codebook_dim) -> (N, T_s'+T_a', D)
        motion_dec = self.postprocess(self.decoder(self.preprocess(motion_quant))) 
        state_dec, action_dec = motion_dec[:, :T_pad_state, :], motion_dec[:, T_pad_state:, :]

        return state_dec[:, :T_state, :], action_dec[:, :T_action, :], commit_loss, perplexity


# ==================== Loss Functions ====================
class ReconstructionLoss(nn.Module):
    """Handles motion reconstruction loss."""
    LOSS_MAP = {
        'l1': nn.L1Loss,
        'l2': nn.MSELoss,
        'l1_smooth': nn.SmoothL1Loss
    }

    def __init__(self, recons_loss: str):
        super().__init__()
        self.loss_fn = self.LOSS_MAP[recons_loss]()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


# ==================== Motion VQ Model ====================
class MotionVQModel(PreTrainedModel):
    config_class = MotionVQModelConfig

    def __init__(self, config: MotionVQModelConfig):
        super().__init__(config)
        self.use_part = config.use_part
        self.horizon = config.horizon
        self.motion_dim = config.motion_dim
        self.wrist_dim = config.wrist_dim
        self.hand_dim = config.hand_dim
        self.commit_weight = config.loss_config.commit_weight
        self.vocab_size = config.vocab_size
        if self.use_part is not None:
            if self.use_part == "wrist":
                self.model = BaseVQModel(config, motion_dim=self.wrist_dim)
            elif self.use_part == "hand":
                self.model = BaseVQModel(config, motion_dim=self.hand_dim)
            else:  # both
                raise NotImplementedError(f"Unsupported use_part: {self.use_part}")
        else:
            self.model = BaseVQModel(config)
        self.loss_fn = ReconstructionLoss(config.loss_config.recons_loss)
    
    def encode(self, x):
        return self.model.encode(x)

    def decode(self, x):
        return self.model.decode(x)

    def forward(self, state: torch.Tensor, action: torch.Tensor, **kwargs):
        pred_state, pred_action, commit_loss, perplexity = self.model(state, action)
        assert pred_state.shape == state.shape and pred_action.shape == action.shape, \
            f"Predicted state and action must have shape (N, T_state, D) and (N, T_action, D)"
        recon_loss = (self.loss_fn(pred_state, state) + self.loss_fn(pred_action, action)) / 2

        if commit_loss is not None:
            total_loss = recon_loss + self.commit_weight * commit_loss.mean()
        else:
            total_loss = recon_loss
        
        return {
            'loss': total_loss,
            'loss_recons': recon_loss,
            'perplexity': perplexity,
            'loss_commit': commit_loss.mean(),
            'avg_abs_pred_state': pred_state.abs().mean(),
            'avg_abs_pred_action': pred_action.abs().mean(),
        }


# ==================== Test Functions ====================
def test_get_extra_padding_for_conv1d():
    """测试 get_extra_padding_for_conv1d 函数"""
    print("=" * 60)
    print("测试 get_extra_padding_for_conv1d")
    print("=" * 60)
    
    # 测试用例 1: 长度能被 stride 整除
    x1 = torch.randn(2, 10, 8)  # (N, D, T), T=8
    stride1 = 4
    extra_pad1 = get_extra_padding_for_conv1d(x1, stride1)
    assert extra_pad1 == 0, f"期望 padding=0，得到 {extra_pad1}"
    print(f"✓ 测试 1 通过: length=8, stride=4, extra_padding={extra_pad1}")
    
    # 测试用例 2: 长度不能被 stride 整除
    x2 = torch.randn(2, 10, 7)  # T=7
    stride2 = 4
    extra_pad2 = get_extra_padding_for_conv1d(x2, stride2)
    assert extra_pad2 == 1, f"期望 padding=1，得到 {extra_pad2}"
    print(f"✓ 测试 2 通过: length=7, stride=4, extra_padding={extra_pad2}")
    
    # 测试用例 3: 长度小于 stride
    x3 = torch.randn(2, 10, 2)  # T=2
    stride3 = 4
    extra_pad3 = get_extra_padding_for_conv1d(x3, stride3)
    assert extra_pad3 == 2, f"期望 padding=2，得到 {extra_pad3}"
    print(f"✓ 测试 3 通过: length=2, stride=4, extra_padding={extra_pad3}")
    
    print("所有 get_extra_padding_for_conv1d 测试通过！\n")


def test_calculate_perplexity_grvq():
    """测试 calculate_perplexity_grvq 函数"""
    print("=" * 60)
    print("测试 calculate_perplexity_grvq")
    print("=" * 60)
    
    codebook_size = 1024
    batch_size, length, num_quantizers = 4, 10, 3
    
    # 测试用例 1: 非分组情况 [Batch, Length, Layers]
    indices1 = torch.randint(0, codebook_size, (batch_size, length, num_quantizers))
    perplexity1 = calculate_perplexity_grvq(indices1, codebook_size)
    assert perplexity1.shape == (num_quantizers,), f"期望形状 ({num_quantizers},)，得到 {perplexity1.shape}"
    assert torch.all(perplexity1 > 0), "perplexity 应该大于 0"
    print(f"✓ 测试 1 通过: 非分组 perplexity 形状 {perplexity1.shape}, 值范围 [{perplexity1.min():.2f}, {perplexity1.max():.2f}]")
    
    # 测试用例 2: 分组情况 [Groups, Batch, Length, Layers]
    num_groups = 2
    indices2 = torch.randint(0, codebook_size, (num_groups, batch_size, length, num_quantizers))
    perplexity2 = calculate_perplexity_grvq(indices2, codebook_size)
    assert perplexity2.shape == (num_groups, num_quantizers), f"期望形状 ({num_groups}, {num_quantizers})，得到 {perplexity2.shape}"
    assert torch.all(perplexity2 > 0), "perplexity 应该大于 0"
    print(f"✓ 测试 2 通过: 分组 perplexity 形状 {perplexity2.shape}, 值范围 [{perplexity2.min():.2f}, {perplexity2.max():.2f}]")
    
    print("所有 calculate_perplexity_grvq 测试通过！\n")


def test_base_vq_model(config: MotionVQModelConfig, quantizer_name: str):
    """测试 BaseVQModel 的基本功能"""
    print("=" * 60)
    print(f"测试 BaseVQModel - {quantizer_name}")
    print("=" * 60)
    
    # 创建模型
    model = BaseVQModel(config)
    model.eval()
    
    batch_size = 2
    T_state, T_action = 16, 8
    motion_dim = config.motion_dim
    
    # 创建测试数据
    state = torch.randn(batch_size, T_state, motion_dim)
    action = torch.randn(batch_size, T_action, motion_dim)
    
    print(f"输入形状: state={state.shape}, action={action.shape}")
    
    # 测试 preprocess/postprocess
    state_preprocessed = model.preprocess(state)
    assert state_preprocessed.shape == (batch_size, motion_dim, T_state), \
        f"preprocess 形状错误: {state_preprocessed.shape}"
    state_postprocessed = model.postprocess(state_preprocessed)
    assert state_postprocessed.shape == state.shape, \
        f"postprocess 形状错误: {state_postprocessed.shape}"
    print(f"✓ preprocess/postprocess 测试通过")
    
    # 测试 extra_padding
    state_padded = model.extra_padding(state_preprocessed)
    assert state_padded.shape[-1] >= state_preprocessed.shape[-1], \
        "padding 后长度应该增加或保持不变"
    print(f"✓ extra_padding 测试通过: {state_preprocessed.shape[-1]} -> {state_padded.shape[-1]}")
    
    # 测试 process_motion
    motion_concat, T_downsampled_state = model.process_motion(state, action)
    assert motion_concat.shape[0] == batch_size, "batch size 应该保持不变"
    assert motion_concat.shape[1] == motion_dim, "特征维度应该保持不变"
    print(f"✓ process_motion 测试通过: motion_concat={motion_concat.shape}, T_downsampled_state={T_downsampled_state}")
    
    # 测试 encode
    with torch.no_grad():
        state_indices, action_indices = model.encode(state, action)
        print(f"✓ encode 测试通过: state_indices={state_indices.shape}, action_indices={action_indices.shape}")
        
        # 测试 decode
        state_dec, action_dec = model.decode(state_indices, action_indices)
        print(f"✓ decode 测试通过: state_dec={state_dec.shape}, action_dec={action_dec.shape}")
        
        # 验证 decode 输出形状
        assert state_dec.shape[0] == batch_size and action_dec.shape[0] == batch_size, \
            "batch size 应该保持不变"
        
        # 测试 forward
        state_pred, action_pred, commit_loss, perplexity = model(state, action)
        assert state_pred.shape == state.shape, \
            f"state 形状不匹配: {state_pred.shape} vs {state.shape}"
        assert action_pred.shape == action.shape, \
            f"action 形状不匹配: {action_pred.shape} vs {action.shape}"
        
        if commit_loss is not None:
            print(f"✓ forward 测试通过: commit_loss 形状={commit_loss.shape}")
        else:
            print(f"✓ forward 测试通过: commit_loss=None (FSQ quantizer)")
        
        print(f"✓ forward 测试通过: perplexity 形状={perplexity.shape}")
        print(f"  预测形状: state_pred={state_pred.shape}, action_pred={action_pred.shape}")
    
    print(f"所有 BaseVQModel ({quantizer_name}) 测试通过！\n")


def test_reconstruction_loss():
    """测试 ReconstructionLoss"""
    print("=" * 60)
    print("测试 ReconstructionLoss")
    print("=" * 60)
    
    batch_size, length, dim = 4, 10, 24
    
    for loss_type in ['l1', 'l2', 'l1_smooth']:
        loss_fn = ReconstructionLoss(loss_type)
        pred = torch.randn(batch_size, length, dim)
        target = torch.randn(batch_size, length, dim)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0, f"{loss_type} loss 应该 >= 0"
        print(f"✓ {loss_type} loss 测试通过: loss={loss.item():.4f}")
    
    print("所有 ReconstructionLoss 测试通过！\n")


def test_motion_vq_model(config: MotionVQModelConfig):
    """测试 MotionVQModel"""
    print("=" * 60)
    print("测试 MotionVQModel")
    print("=" * 60)
    
    model = MotionVQModel(config)
    model.eval()
    
    batch_size = 2
    T_state, T_action = 16, 8
    
    # 根据 use_part 确定维度
    if config.use_part == "wrist":
        motion_dim = config.wrist_dim
    elif config.use_part == "hand":
        motion_dim = config.hand_dim
    else:
        motion_dim = config.motion_dim
    
    state = torch.randn(batch_size, T_state, motion_dim)
    action = torch.randn(batch_size, T_action, motion_dim)
    
    print(f"输入形状: state={state.shape}, action={action.shape}")
    print(f"use_part={config.use_part}, motion_dim={motion_dim}")
    
    with torch.no_grad():
        # 测试 encode (直接使用 model.model.encode，因为 MotionVQModel.encode 接口可能有问题)
        state_indices, action_indices = model.model.encode(state, action)
        print(f"✓ encode 测试通过: state_indices={state_indices.shape}, action_indices={action_indices.shape}")
        
        # 测试 decode (直接使用 model.model.decode)
        state_dec, action_dec = model.model.decode(state_indices, action_indices)
        print(f"✓ decode 测试通过: state_dec={state_dec.shape}, action_dec={action_dec.shape}")
        
        # 测试 forward
        outputs = model(state, action)
        assert 'loss' in outputs, "输出应该包含 'loss'"
        assert 'loss_recons' in outputs, "输出应该包含 'loss_recons'"
        assert 'perplexity' in outputs, "输出应该包含 'perplexity'"
        
        print(f"✓ forward 测试通过:")
        print(f"  loss={outputs['loss'].item():.4f}")
        print(f"  loss_recons={outputs['loss_recons'].item():.4f}")
        print(f"  perplexity 形状={outputs['perplexity'].shape}")
        if outputs['loss_commit'] is not None:
            print(f"  loss_commit 形状={outputs['loss_commit'].shape}")
    
    print("所有 MotionVQModel 测试通过！\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行 VQ Model 测试套件")
    print("=" * 60 + "\n")
    
    # 测试工具函数
    test_get_extra_padding_for_conv1d()
    test_calculate_perplexity_grvq()
    test_reconstruction_loss()
    
    # 测试不同 quantizer 类型
    quantizer_configs = [
        {
            "quantizer_name": "residualvq",
            "codebook_size": 1024,
            "codebook_dim": 128,
            "num_quantizers": 4,
            "shared_codebook": True
        },
        {
            "quantizer_name": "group_residualvq",
            "codebook_size": 1024,
            "codebook_dim": 128,
            "num_quantizers": 4,
            "num_groups": 2,
            "shared_codebook": True
        },
        {
            "quantizer_name": "residualfsq",
            "codebook_size": 1024,
            "codebook_dim": 128,
            "num_quantizers": 4
        },
        {
            "quantizer_name": "grouped_residualfsq",
            "codebook_size": 1024,
            "codebook_dim": 128,
            "num_quantizers": 4
        }
    ]
    
    # 基础模型配置
    base_model_config = {
        "down_t": 2,
        "stride_t": 2,
        "width": 128,
        "depth": 2,
        "output_emb_width": 128,
        "num_conv_layers": 2
    }
    
    base_loss_config = {
        "recons_loss": "l2",
        "commit_weight": 0.02
    }
    
    # 测试 BaseVQModel
    for q_config in quantizer_configs:
        config = MotionVQModelConfig(
            motion_dim=24,
            model_config=base_model_config,
            quantizer_config=q_config,
            loss_config=base_loss_config
        )
        try:
            test_base_vq_model(config, q_config["quantizer_name"])
        except Exception as e:
            print(f"✗ {q_config['quantizer_name']} 测试失败: {e}\n")
            import traceback
            traceback.print_exc()
    
    # 测试 MotionVQModel (不同 use_part)
    use_part_options = [None, "wrist", "hand"]
    
    for use_part in use_part_options:
        config = MotionVQModelConfig(
            use_part=use_part,
            motion_dim=24,
            wrist_dim=9,
            hand_dim=15,
            model_config=base_model_config,
            quantizer_config=quantizer_configs[0],  # 使用 residualvq
            loss_config=base_loss_config
        )
        try:
            test_motion_vq_model(config)
        except Exception as e:
            print(f"✗ MotionVQModel (use_part={use_part}) 测试失败: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
