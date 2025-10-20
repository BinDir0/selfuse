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
from src.model.action.vq_model import MotionVQModel
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
        vq_model: Union[nn.Module, dict],
        wrist_dim: int,
        hand_dim: int,
    ):
        """
        Args:
            vq_model: Trained VQ-VAE model (from mano_adapter style)
                      Can be a single model or dict with 'wrist' and 'hand' keys for bimanual
            vocab_size: Size of the codebook (e.g., 512, 1024)
            action_dim: Action dimension (inferred from data if None)
            time_horizon: Time horizon (inferred from data if None)
            action_mean: Normalization mean
            action_std: Normalization std
            wrist_dim: Dimension of wrist data (for bimanual mode)
            hand_dim: Dimension of hand data (for bimanual mode)
        """
        self.vq_model = vq_model
        
        # Bimanual dimensions
        self.wrist_dim = wrist_dim
        self.hand_dim = hand_dim
        
        # Metadata storage for bimanual encoding
        self.vq_meta = None
          
    def __call__(self, action_chunk: np.array) -> List[np.ndarray]:
        """
        Encode bimanual actions to discrete token IDs in time-interleaved order.
        
        Args:
            action_chunk: [T, D] where D = 2*wrist_dim + 2*hand_dim
                         Layout: [left_wrist, right_wrist, left_hand, right_hand]
            
        Returns:
            List containing single element: 1D flattened discrete tokens (np.ndarray)
        """

        
        if isinstance(action_chunk, np.ndarray):
            action_chunk_tensor = torch.from_numpy(action_chunk).float()
        elif isinstance(action_chunk, torch.Tensor):
            action_chunk_tensor = action_chunk
        
        # Define dimension slices
        w = self.wrist_dim
        h = self.hand_dim
        
        # Split data into 4 parts - expecting [T, D] input
        wrist_left_data = action_chunk_tensor[:, :w]                      # [T, w]
        wrist_right_data = action_chunk_tensor[:, w:2*w]                  # [T, w]
        hand_left_data = action_chunk_tensor[:, 2*w:2*w+h]                # [T, h]
        hand_right_data = action_chunk_tensor[:, 2*w+h:2*w+2*h]           # [T, h]
        
        # Add batch dimension for VQ model: [T, D] -> [B=1, T, D]
        wrist_left_data = wrist_left_data.unsqueeze(0)            # [1, T, w]
        wrist_right_data = wrist_right_data.unsqueeze(0)          # [1, T, w]
        hand_left_data = hand_left_data.unsqueeze(0)              # [1, T, h]
        hand_right_data = hand_right_data.unsqueeze(0)            # [1, T, h]
        
        # Encode each part using VQ model's encode method: returns [G, B, T, L]
        wrist_left_raw = self.vq_model["wrist"].encode(wrist_left_data)
        wrist_right_raw = self.vq_model['wrist'].encode(wrist_right_data)
        hand_left_raw = self.vq_model['hand'].encode(hand_left_data)
        hand_right_raw = self.vq_model['hand'].encode(hand_right_data)
        
        # Flatten with time-interleaved order and get metadata
        # Squeeze batch dimension and convert to numpy
        WL = np.asarray(wrist_left_raw).squeeze(1)   # [G, T, L]
        WR = np.asarray(wrist_right_raw).squeeze(1)  # [G, T, L]
        HL = np.asarray(hand_left_raw).squeeze(1)    # [G, T, L]
        HR = np.asarray(hand_right_raw).squeeze(1)   # [G, T, L]
        
        G, T, Lw = WL.shape
        Lh = HL.shape[2]
        self.vq_meta = {"G": G, "T": T, "Lw": Lw, "Lh": Lh}
        seq = []
        # Iterate through time (outer loop for time locality)
        for t in range(T):
            # At each timestep, add: left_wrist, right_wrist, left_hand, right_hand
            # For each part, iterate through groups
            
            # Left wrist at time t
            for g in range(G):
                seq.extend(WL[g, t].tolist())
            
            # Right wrist at time t
            for g in range(G):
                seq.extend(WR[g, t].tolist())
            
            # Left hand at time t
            for g in range(G):
                seq.extend(HL[g, t].tolist())
            
            # Right hand at time t
            for g in range(G):
                seq.extend(HR[g, t].tolist())
        
        # Return as list (compatible with Fast tokenizer interface)
        return [np.asarray(seq, dtype=np.int64)]
    
    @torch.no_grad()
    def decode(
        self,
        tokens_1d: np.ndarray,
        meta: Optional[dict] = None,
    ) -> np.array:
        """
        Decode time-interleaved VQ tokens back to bimanual continuous actions.
        
        Args:
            tokens_1d: 1D array of VQ token IDs in time-interleaved format
            meta: Dictionary with keys 'G', 'T', 'Lw', 'Lh' (uses self.vq_meta if None)
            
        Returns:
            Decoded actions: [T, D] where D = 2*wrist_dim + 2*hand_dim
        """
        # Use cached metadata if not provided
        if meta is None:
            meta = self.vq_meta
            if meta is None:
                raise ValueError("No metadata provided and no cached metadata found")
        
        G = int(meta["G"])
        T = int(meta["T"])
        Lw = int(meta["Lw"])
        Lh = int(meta["Lh"])
        
        # Calculate tokens per timestep
        wrist_per_t = 2 * G * Lw  # left + right wrist
        hand_per_t = 2 * G * Lh   # left + right hand
        tokens_per_t = wrist_per_t + hand_per_t
        
        # Check if we have enough tokens
        expected = T * tokens_per_t
        if len(tokens_1d) < expected:
            logger.warning(f"Short sequence: {len(tokens_1d)} < {expected}")
            T = len(tokens_1d) // tokens_per_t
        
        # Reconstruct [G, T, L] format for wrist and hand
        wrist_ids_left = np.zeros((G, T, Lw), dtype=np.int64)
        wrist_ids_right = np.zeros((G, T, Lw), dtype=np.int64)
        hand_ids_left = np.zeros((G, T, Lh), dtype=np.int64)
        hand_ids_right = np.zeros((G, T, Lh), dtype=np.int64)
        
        for t in range(T):
            start = t * tokens_per_t
            timestep_tokens = tokens_1d[start : start + tokens_per_t]
            
            # Extract wrist tokens (first wrist_per_t tokens)
            wrist_tokens_t = timestep_tokens[:wrist_per_t]
            
            # Left wrist: first G*Lw tokens
            for g in range(G):
                for l in range(Lw):
                    idx = g * Lw + l
                    if idx < len(wrist_tokens_t):
                        wrist_ids_left[g, t, l] = wrist_tokens_t[idx]
            
            # Right wrist: next G*Lw tokens
            for g in range(G):
                for l in range(Lw):
                    idx = G * Lw + g * Lw + l
                    if idx < len(wrist_tokens_t):
                        wrist_ids_right[g, t, l] = wrist_tokens_t[idx]
            
            # Extract hand tokens (remaining tokens)
            hand_tokens_t = timestep_tokens[wrist_per_t:]
            
            # Left hand: first G*Lh tokens
            for g in range(G):
                for l in range(Lh):
                    idx = g * Lh + l
                    if idx < len(hand_tokens_t):
                        hand_ids_left[g, t, l] = hand_tokens_t[idx]
            
            # Right hand: next G*Lh tokens
            for g in range(G):
                for l in range(Lh):
                    idx = G * Lh + g * Lh + l
                    if idx < len(hand_tokens_t):
                        hand_ids_right[g, t, l] = hand_tokens_t[idx]
        
        # Decode each part using VQ model
        # Convert numpy arrays to torch tensors and move to model device
        device = next(self.vq_model["wrist"].parameters()).device
        wrist_ids_left_tensor = torch.from_numpy(wrist_ids_left).long().to(device)
        wrist_ids_right_tensor = torch.from_numpy(wrist_ids_right).long().to(device)
        hand_ids_left_tensor = torch.from_numpy(hand_ids_left).long().to(device)
        hand_ids_right_tensor = torch.from_numpy(hand_ids_right).long().to(device)
        
        acts_wrist_left = self.vq_model["wrist"].forward_decoder(wrist_ids_left_tensor)
        acts_wrist_right = self.vq_model["wrist"].forward_decoder(wrist_ids_right_tensor)
        acts_hand_left = self.vq_model["hand"].forward_decoder(hand_ids_left_tensor)
        acts_hand_right = self.vq_model["hand"].forward_decoder(hand_ids_right_tensor)
        
        # Concatenate: [left_wrist, right_wrist, left_hand, right_hand]
        # Each part is [T, dim], concatenate along last axis to get [T, 2*wrist_dim + 2*hand_dim]
        # Convert tensors to numpy (detach first to remove gradients)
        decoded_actions = np.concatenate([
            acts_wrist_left.detach().cpu().numpy() if isinstance(acts_wrist_left, torch.Tensor) else acts_wrist_left,
            acts_wrist_right.detach().cpu().numpy() if isinstance(acts_wrist_right, torch.Tensor) else acts_wrist_right,
            acts_hand_left.detach().cpu().numpy() if isinstance(acts_hand_left, torch.Tensor) else acts_hand_left,
            acts_hand_right.detach().cpu().numpy() if isinstance(acts_hand_right, torch.Tensor) else acts_hand_right
        ], axis=-1)
        
        return decoded_actions
    
    @classmethod
    def from_pretrained(cls, load_directory: str, wrist_dim: int = 9, hand_dim: int = 15):
        """
        Load VQ tokenizer from directory.
        
        Args:
            load_directory: Path to directory containing 'wrist' and 'hand' subdirectories
            wrist_dim: Dimension of wrist data (default: 9)
            hand_dim: Dimension of hand data (default: 15)
            
        Returns:
            VQActionProcessor instance with loaded wrist and hand models
        """
        
        # Load wrist and hand models
        vq_model = {
            "wrist": MotionVQModel.from_pretrained(
                os.path.join(load_directory, "wrist")
            ),
            "hand": MotionVQModel.from_pretrained(
                os.path.join(load_directory, "hand")
            )
        }
        
        logger.info(f"VQ tokenizer loaded from {load_directory}")
        
        return cls(
            vq_model=vq_model,
            wrist_dim=wrist_dim,
            hand_dim=hand_dim,
        )

    

