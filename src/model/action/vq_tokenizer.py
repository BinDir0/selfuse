# VQ-based Action Tokenizer (替代 fast_tokenizer 的 DCT+BPE)
# Interface compatible with UniversalActionProcessor

import logging
import os
import pickle
from typing import Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
from transformers.processing_utils import ProcessorMixin
from src.model.action.vq_model import ActionVQModel, train_vq_model
logger = logging.getLogger(__name__)


class VQActionProcessor(ProcessorMixin):
    """
    VQ-based action tokenizer.
    
    Interface compatible with UniversalActionProcessor:
    - __call__(actions) -> list of token sequences
    - decode(token_sequences) -> actions
    - fit() for training
    - save_pretrained() / from_pretrained() for persistence
    """
    
    attributes = ["vq_model"]
    vq_model_class = "ActionVQModel"  # Required by ProcessorMixin
    
    def __init__(
        self,
        vq_model: nn.Module,
        vocab_size: int,
        *,
        action_dim: Optional[int] = None,
        time_horizon: Optional[int] = None,
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
    ):
        """
        Args:
            vq_model: Trained VQ-VAE model (from mano_adapter style)
            vocab_size: Size of the codebook (e.g., 512, 1024)
            action_dim: Action dimension (inferred from data if None)
            time_horizon: Time horizon (inferred from data if None)
            action_mean: Normalization mean
            action_std: Normalization std
        """
        self.vq_model = vq_model
        self.vocab_size = vocab_size
        
        # Action shape info (needed for decoding)
        self.time_horizon = time_horizon
        self.action_dim = action_dim
        self.called_time_horizon = time_horizon
        self.called_action_dim = action_dim
        
        # Normalization parameters
        self.action_mean = action_mean
        self.action_std = action_std
        
    
    def __call__(self, action_chunk: np.array) -> List[List[int]]:
        """
        Encode actions to discrete token IDs.
        
        Args:
            action_chunk: [batch, timesteps, action_dim] or [timesteps, action_dim]
            
        Returns:
            List of token sequences (one per batch element)
        """
        assert action_chunk.ndim <= 3, "Only 3 dimensions supported: [batch, timesteps, action_dim]"
        if action_chunk.ndim == 2:
            action_chunk = action_chunk[None, ...]
        
        # Cache the time horizon and action dimension for decoding
        self.called_time_horizon = action_chunk.shape[-2]
        self.called_action_dim = action_chunk.shape[-1]
        
        # Normalize
        if self.action_mean is not None and self.action_std is not None:
            action_chunk = (action_chunk - self.action_mean) / (self.action_std + 1e-8)
        
        # Convert to torch tensor
        actions_tensor = torch.from_numpy(action_chunk).float()
        
        # Encode with VQ model
        with torch.no_grad():
            self.vq_model.eval()
            token_ids = self.vq_model.encode(actions_tensor)  # [batch, num_tokens]
        
        # Convert to list of lists (compatible with fast_tokenizer interface)
        token_ids = token_ids.cpu().numpy()
        tokens = [token_ids[i].tolist() for i in range(token_ids.shape[0])]
        
        return tokens
    
    def decode(
        self,
        tokens: List[List[int]],
        *,
        time_horizon: Optional[int] = None,
        action_dim: Optional[int] = None,
    ) -> np.array:
        """
        Decode token IDs back to continuous actions.
        
        Args:
            tokens: List of token sequences
            time_horizon: Override time horizon
            action_dim: Override action dimension
            
        Returns:
            Decoded actions: [batch, timesteps, action_dim]
        """
        self.time_horizon = time_horizon or self.time_horizon or self.called_time_horizon
        self.action_dim = action_dim or self.action_dim or self.called_action_dim
        
        # Cache for next call
        self.called_time_horizon = self.time_horizon
        self.called_action_dim = self.action_dim
        
        assert (
            self.time_horizon is not None and self.action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."
        
        # Convert to torch tensor
        token_tensor = torch.LongTensor(tokens)  # [batch, num_tokens]
        
        # Decode with VQ model
        with torch.no_grad():
            self.vq_model.eval()
            decoded_actions = self.vq_model.decode(token_tensor)  # [batch, timesteps, action_dim]
        
        decoded_actions = decoded_actions.cpu().numpy()
        
        # Denormalize
        if self.action_mean is not None and self.action_std is not None:
            decoded_actions = decoded_actions * self.action_std + self.action_mean
        
        return decoded_actions
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save VQ model
        torch.save(
            self.vq_model.state_dict(),
            os.path.join(save_directory, "vq_model.pth")
        )
        
        # Save full VQ model config (including architecture details)
        vq_model_config = {
            "action_dim": self.vq_model.action_dim,
            "time_horizon": self.vq_model.time_horizon,
            "codebook_size": self.vq_model.codebook_size,
            "codebook_dim": self.vq_model.codebook_dim,
            "num_quantizers": self.vq_model.num_quantizers,
            "down_t": self.vq_model.down_t,
            "quantizer_type": self.vq_model.quantizer_type,
            "num_groups": self.vq_model.num_groups,
        }
        
        # Save processor configuration
        config = {
            "vocab_size": self.vocab_size,
            "time_horizon": self.time_horizon,
            "action_dim": self.action_dim,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "vq_model_config": vq_model_config,
        }
        with open(os.path.join(save_directory, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
        
        logger.info(f"VQ tokenizer saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory: str, vq_model_class=None):
        """Load tokenizer from directory."""
        # Load configuration
        with open(os.path.join(load_directory, "config.pkl"), "rb") as f:
            config = pickle.load(f)
        
        # Get VQ model config
        vq_config = config.get("vq_model_config", {})
        
        # Create VQ model with full configuration
        if vq_model_class is None:
            from .vq_model import ActionVQModel
            vq_model = ActionVQModel(**vq_config)
        else:
            vq_model = vq_model_class(**vq_config)
        
        # Load state dict
        vq_model.load_state_dict(
            torch.load(os.path.join(load_directory, "vq_model.pth"),
                      weights_only=False)
        )
        vq_model.eval()
        
        logger.info(f"VQ tokenizer loaded from {load_directory}")
        
        return cls(
            vq_model=vq_model,
            vocab_size=config["vocab_size"],
            action_dim=config["action_dim"],
            time_horizon=config["time_horizon"],
            action_mean=config["action_mean"],
            action_std=config["action_std"],
        )
    
    @classmethod
    def fit(
        cls,
        action_data: Union[List[np.array], np.array],
        vocab_size: int = 512,
        *,
        time_horizon: Optional[int] = None,
        action_dim: Optional[int] = None,
        vq_config: dict = None,
        device: str = "cuda",
        quantizer_type: str = "residualvq",  # or "group_residualvq"
        **training_kwargs,
    ) -> "VQActionProcessor":
        """
        Train VQ tokenizer on action data.
        
        Args:
            action_data: Action sequences to train on
            vocab_size: Codebook size
            time_horizon: Time horizon (inferred if None)
            action_dim: Action dimension (inferred if None)
            vq_config: VQ model configuration
            device: Training device
            **training_kwargs: Additional training parameters
            
        Returns:
            Trained VQActionProcessor
        """
        # Convert to numpy array if needed
        if isinstance(action_data, list):
            action_data = np.array(action_data)
        
        # Infer dimensions
        if time_horizon is None:
            time_horizon = action_data.shape[1]
        if action_dim is None:
            action_dim = action_data.shape[2]
        
        # Compute normalization statistics
        action_mean = action_data.mean(axis=(0, 1))
        action_std = action_data.std(axis=(0, 1))
        
        logger.info(f"Training VQ tokenizer: vocab_size={vocab_size}, "
                   f"time_horizon={time_horizon}, action_dim={action_dim}")
        logger.info(f"Action mean: {action_mean}, std: {action_std}")
        
        # Normalize data
        action_data_norm = (action_data - action_mean) / (action_std + 1e-8)

        
        vq_config = vq_config or {}
        vq_model = ActionVQModel(
            action_dim=action_dim,
            time_horizon=time_horizon,
            codebook_size=vocab_size,
            quantizer_type=quantizer_type,
            **vq_config,
        )
        
        # Train VQ model
        logger.info("Training VQ-VAE model...")
        vq_model = train_vq_model(
            vq_model,
            action_data_norm,
            device=device,
            **training_kwargs,
        )
        
        logger.info("VQ training complete!")
        
        return cls(
            vq_model=vq_model,
            vocab_size=vocab_size,
            action_dim=action_dim,
            time_horizon=time_horizon,
            action_mean=action_mean,
            action_std=action_std,
        )


# ============ Test ============

if __name__ == "__main__":
    # Example: Train VQ tokenizer (similar to fast_tokenizer training)
    
    # 1. Load action data
    action_data = np.random.randn(1000, 30, 18)  # [num_sequences, horizon, action_dim]
    
    # 2a. Train tokenizer with RVQ (default)
    tokenizer_rvq = VQActionProcessor.fit(
        action_data,
        vocab_size=512,
        quantizer_type="residualvq",  # RVQ
        vq_config={
            "codebook_dim": 256,
            "num_quantizers": 4,
        },
        num_epochs=100,
        batch_size=128,
        part="wrist",
    )
    
    # 2b. Train tokenizer with GRVQ (like Being-H0)
    tokenizer_grvq = VQActionProcessor.fit(
        action_data,
        vocab_size=8192,  # Being-H0 uses 8K
        quantizer_type="group_residualvq",  # GRVQ
        vq_config={
            "codebook_dim": 256,
            "num_quantizers": 4,
            "num_groups": 2,  # Split into 2 groups
        },
        num_epochs=100,
        batch_size=128,
        part="wrist",
    )
    
    # 3. Save tokenizer
    tokenizer_grvq.save_pretrained("output/vq_tokenizer_grvq")
    
    # 4. Load and use
    tokenizer = VQActionProcessor.from_pretrained("output/vq_tokenizer_grvq")
    
    # Encode (must match training action_dim)
    test_actions = np.random.randn(2, 30, 18)  # [batch, time, action_dim]
    tokens = tokenizer(test_actions)
    print(f"Encoded tokens shape: {[len(t) for t in tokens]}")
    print(f"Sample tokens: {tokens[0][:10]}")  # Show first 10 tokens
    
    # Decode
    reconstructed = tokenizer.decode(tokens)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Reconstruction error (L1): {np.mean(np.abs(test_actions - reconstructed)):.6f}")


