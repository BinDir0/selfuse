import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from src.utils import create_diffusion
from src.model.common.modules import SinusoidalPosEmb, TimeEncoder


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class BaseGenerativeMLP(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = self.build_time_embed()
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()
        self.initialize_time_embed()

    def build_time_embed(self):
        raise NotImplementedError("Subclasses must implement build_time_embed")

    def initialize_time_embed(self):
        raise NotImplementedError("Subclasses must implement initialize_time_embed")

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @torch.compile(mode="default")
    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.training and self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y, use_reentrant=False)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)


class SimpleMLPAdaLN(BaseGenerativeMLP):
    def build_time_embed(self):
        return TimestepEmbedder(self.model_channels)

    def initialize_time_embed(self):
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

    @torch.compile(mode="default")
    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class FlowMatchingMLP(BaseGenerativeMLP):
    """
    Specialized MLP for Flow Matching.
    1. Output channels = input channels (only predicts velocity, no variance).
    2. Forward handles continuous time t (float in [0, 1]).
    """
    def __init__(
        self,
        in_channels,
        model_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
        time_min_period=0.004,
        time_max_period=4.0
    ):
        self.time_min_period = time_min_period
        self.time_max_period = time_max_period
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            z_channels=z_channels,
            num_res_blocks=num_res_blocks,
            grad_checkpointing=grad_checkpointing
        )

    def build_time_embed(self):
        return nn.Sequential(
            SinusoidalPosEmb(self.model_channels, self.time_min_period, self.time_max_period),
            TimeEncoder(self.model_channels)
        )

    def initialize_time_embed(self):
        nn.init.normal_(self.time_embed[1].linear_1.weight, std=0.02)
        nn.init.normal_(self.time_embed[1].linear_2.weight, std=0.02)

    @torch.compile(mode="default")
    def forward_with_cfg(self, x, t, c, cfg_scale):
        """
        Classifier-free guidance for Flow Matching.
        Assumes input x is [2B, C] where first half is conditional, second half is unconditional.
        """
        half_len = x.shape[0] // 2
        half = x[:half_len]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)  # [2B, C]
        
        cond_out, uncond_out = torch.split(model_out, half_len, dim=0)
        
        # CFG: v = v_uncond + scale * (v_cond - v_uncond)
        out = uncond_out + cfg_scale * (cond_out - uncond_out)
        
        # Return doubled batch for consistency with original API
        return torch.cat([out, out], dim=0)


# -----------------------------------------------------------------------------
# Modified DiffLoss with Flow Matching support
# -----------------------------------------------------------------------------
class DiffLoss(nn.Module):
    """Diffusion Loss with optional Flow Matching supervision"""
    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps,
        grad_checkpointing=False,
        use_ddim_sampling=False,
        use_flow_matching=False,
        flow_sig_min=0.001,
        time_min_period=0.004,  # Min period for Flow Matching time embedding
        time_max_period=4.0,  # Max period for Flow Matching time embedding
        flow_sampling="beta",  # Time sampling strategy: "uniform" or "beta"
        flow_alpha=1.5,  # Beta distribution alpha parameter
        flow_beta=1.0,  # Beta distribution beta parameter
        num_inference_steps=10,  # Euler ODE steps for Flow Matching sampling
    ):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.use_ddim_sampling = use_ddim_sampling
        self.use_flow_matching = use_flow_matching
        self.flow_sig_min = flow_sig_min
        
        # Flow matching time sampling
        self.flow_sampling = flow_sampling
        if self.flow_sampling == "beta":
            self.flow_t_max = 1.0 - flow_sig_min
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
        
        # Choose network architecture based on configuration
        if self.use_flow_matching:
            self.net = FlowMatchingMLP(
                in_channels=target_channels,
                model_channels=width,
                z_channels=z_channels,
                num_res_blocks=depth,
                grad_checkpointing=grad_checkpointing,
                time_min_period=time_min_period,
                time_max_period=time_max_period
            )
            self.train_diffusion = None
            self.gen_diffusion = None
            self.num_inference_steps = num_inference_steps
        else:
            self.net = SimpleMLPAdaLN(
                in_channels=target_channels,
                model_channels=width,
                out_channels=target_channels * 2,  # Diffusion needs double output
                z_channels=z_channels,
                num_res_blocks=depth,
                grad_checkpointing=grad_checkpointing
            )
            self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")

            # Diffusion sampling scheduler
            if self.use_ddim_sampling:
                if isinstance(num_sampling_steps, str) and num_sampling_steps.startswith("ddim"):
                    timestep_respacing = num_sampling_steps
                else:
                    timestep_respacing = f"ddim{num_sampling_steps}"
            else:
                timestep_respacing = num_sampling_steps
            self.gen_diffusion = create_diffusion(timestep_respacing=timestep_respacing, noise_schedule="cosine")

    def sample_time(self, bsz: int, device: torch.device) -> torch.FloatTensor:
        """
        Sample time steps for flow matching training.
        Uses configurable sampling strategy (uniform or beta distribution).
        
        Args:
            bsz: Batch size
            device: Target device
            
        Returns:
            torch.FloatTensor: [bsz] Time steps in [0, 1]
        """
        if self.flow_sampling == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(bsz, device=device) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":
            z = self.flow_beta_dist.sample((bsz,)).to(device)
            t = self.flow_t_max * (1 - z)  # flip and shift
        else:
            # Fallback to uniform
            t = torch.rand(bsz, device=device)
        return t

    def set_dim_weights(self, dim_weights, chunk_size):
        """Register per-dimension loss weights, tiled for flattened action chunks.

        Args:
            dim_weights: [action_dim] per-dimension weight, e.g. shape [48]
            chunk_size: temporal chunk size, e.g. 4
        """
        tiled = dim_weights.repeat_interleave(chunk_size)
        self.register_buffer('dim_weights', tiled)

    def forward(self, target, z, mask=None, t=None):
        """
        Compute loss based on configuration.
        
        Args:
            target: [B, C] Ground truth action
            z: [B, D] Condition embedding
            mask: Optional mask for valid elements
            t: Optional [B] timesteps for flow matching (if None, will sample internally)
            
        Returns:
            Loss value
        """
        if self.use_flow_matching:
            return self.flow_matching_loss(target, z, mask, t)
        else:
            return self.diffusion_loss(target, z, mask)

    @staticmethod
    def _broadcast_mask(mask, loss):
        if mask is None:
            return None
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(-1)
        return mask.to(dtype=loss.dtype)

    def diffusion_loss(self, target, z, mask=None):
        """Traditional DDPM diffusion loss"""
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(
            self.net,
            target,
            t,
            model_kwargs,
            dim_weights=getattr(self, "dim_weights", None),
        )
        loss = loss_dict["loss"]
        if mask is not None:
            mask = self._broadcast_mask(mask, loss)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss.mean()
    
    def flow_matching_loss(self, target, z, mask=None, t=None):
        """
        Flow Matching Loss (Pure Continuous Time).
        
        Args:
            target: [B, C] Ground truth action
            z: [B, D] Condition embedding
            mask: Optional mask for valid elements
            t: Optional [B] timesteps (if None, samples internally)
        
        Predicts velocity field: v_target = x1 - (1 - sigma_min) * x0
        """
        # 1. Sample continuous time t ∈ [0, 1] (if not provided)
        if t is None:
            # Sample using configured strategy (beta or uniform)
            t = self.sample_time(target.shape[0], target.device).to(dtype=target.dtype)
        
        # 2. Sample noise and construct interpolation
        x0 = torch.randn_like(target)  # Noise
        x1 = target                    # Data
        
        # OT path interpolation: x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
        t_expanded = t.view(-1, 1)
        a_t = 1 - (1 - self.flow_sig_min) * t_expanded
        b_t = t_expanded
        x_t = a_t * x0 + b_t * x1
        
        # 3. Compute target velocity
        # v_target = x1 - (1 - sigma_min) * x0
        v_target = x1 - (1 - self.flow_sig_min) * x0
        
        # 4. Model prediction (pass float t, network handles embedding scaling)
        v_pred = self.net(x_t, t, c=z)
        
        # 5. MSE Loss
        loss = (v_pred - v_target) ** 2
        # Apply per-dimension weighting
        if hasattr(self, 'dim_weights') and self.dim_weights is not None:
            loss = loss * self.dim_weights

        if mask is not None:
            mask = self._broadcast_mask(mask, loss)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
            
        return loss

    def sample(self, z, temperature=1.0, cfg=1.0, num_steps=None):
        """
        Sample from the model.
        
        Args:
            z: [B, D] Condition embedding
            temperature: Sampling temperature
            cfg: Classifier-free guidance scale
            num_steps: Number of sampling steps (for flow matching)
            
        Returns:
            Sampled actions [B, C]
        """
        if self.use_flow_matching:
            return self.flow_matching_sample(z, temperature, cfg, num_steps)
        else:
            return self.diffusion_sample(z, temperature, cfg)
    
    def diffusion_sample(self, z, temperature=1.0, cfg=1.0):
        """Traditional DDPM/DDIM sampling"""
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        if self.use_ddim_sampling:
            sampled_token_latent = self.gen_diffusion.ddim_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
                eta=0.0
            )
        else:
            sampled_token_latent = self.gen_diffusion.p_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
                temperature=temperature
            )
        return sampled_token_latent

    @torch.no_grad()
    def flow_matching_sample(self, z, temperature=1.0, cfg=1.0, num_steps=None):
        """
        Flow Matching Euler ODE Solver.
        """
        batch_size = z.shape[0]
        device = z.device
        
        if num_steps is None:
            num_steps = self.num_inference_steps
        
        # Initial noise
        x = torch.randn(batch_size, self.in_channels, device=device) * temperature
        
        # CFG setup
        if cfg != 1.0:
            x = torch.cat([x, x], dim=0)
            
            z_uncond = torch.zeros_like(z) 
            z_in = torch.cat([z, z_uncond], dim=0) 
        else:
            z_in = z

        # Euler integration loop (t from 0 to 1)
        dt = 1.0 / num_steps
        t_curr = torch.zeros(x.shape[0], device=device)
        
        for step in range(num_steps):
            # Model predicts velocity
            if cfg != 1.0:
                v_pred = self.net.forward_with_cfg(x, t_curr, z_in, cfg_scale=cfg)
            else:
                v_pred = self.net(x, t_curr, c=z_in)
            
            # x_{t+1} = x_t + v_t * dt
            x = x + v_pred * dt
            t_curr = t_curr + dt
            
        # If using CFG, extract first half (Conditional result)
        if cfg != 1.0:
            x, _ = x.chunk(2, dim=0)
            
        return x
