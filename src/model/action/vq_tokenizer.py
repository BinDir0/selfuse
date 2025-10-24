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
            wrist_dim: Dimension of wrist data (for bimanual mode)
            hand_dim: Dimension of hand data (for bimanual mode)
        """
        self.vq_model = vq_model
        
        # Bimanual dimensions
        self.wrist_dim = wrist_dim
        self.hand_dim = hand_dim
        self.vocab_size = sum([self.vq_model[k].vocab_size for k in self.vq_model.keys()])
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
        # Data layout: [left_trans(3), right_trans(3), left_rot(6), right_rot(6), hand_left(15), hand_right(15)]
        T_original = action_chunk_tensor.shape[0]  # Save original time horizon
        
        # Reconstruct wrist_left = [left_trans + left_rot], wrist_right = [right_trans + right_rot]
        wrist_left_data = torch.cat([
            action_chunk_tensor[:, 0:3],    # left translation
            action_chunk_tensor[:, 6:12]    # left rotation
        ], dim=-1)  # [T, 9]
        
        wrist_right_data = torch.cat([
            action_chunk_tensor[:, 3:6],    # right translation
            action_chunk_tensor[:, 12:18]   # right rotation
        ], dim=-1)  # [T, 9]
        
        hand_left_data = action_chunk_tensor[:, 18:18+h]                # [T, h]
        hand_right_data = action_chunk_tensor[:, 18+h:18+2*h]           # [T, h]
        print(f"[VQ ENCODE] Ground Truth:")
        print(f"[VQ ENCODE] wrist_left[0:3, :3]: {wrist_left_data[0:3, :3]}, wrist_right[0:3, :3]: {wrist_right_data[0:3, :3]}")
        
        # Add batch dimension for VQ model: [T, D] -> [B=1, T, D]
        wrist_left_data = wrist_left_data.unsqueeze(0)            # [1, T, w]
        wrist_right_data = wrist_right_data.unsqueeze(0)          # [1, T, w]
        hand_left_data = hand_left_data.unsqueeze(0)              # [1, T, h]
        hand_right_data = hand_right_data.unsqueeze(0)            # [1, T, h]
        
        # Encode each part using VQ model's encode method: returns [G, B, T/4, L]
        print(f"[VQ ENCODE] Input shapes - wrist_left: {wrist_left_data.shape}, hand_left: {hand_left_data.shape}")
        wrist_left_raw = self.vq_model["wrist"].encode(wrist_left_data)
        wrist_right_raw = self.vq_model['wrist'].encode(wrist_right_data)
        hand_left_raw = self.vq_model['hand'].encode(hand_left_data)
        hand_right_raw = self.vq_model['hand'].encode(hand_right_data)
        
        print(f"[VQ ENCODE] Raw output shapes - wrist_left: {wrist_left_raw.shape}, hand_left: {hand_left_raw.shape}")
        
        # Flatten with time-interleaved order and get metadata
        # Squeeze batch dimension and convert to numpy
        WL = np.asarray(wrist_left_raw).squeeze(1)   # [G, T/4, L]
        WR = np.asarray(wrist_right_raw).squeeze(1)  # [G, T/4, L]
        HL = np.asarray(hand_left_raw).squeeze(1)    # [G, T/4, L]
        HR = np.asarray(hand_right_raw).squeeze(1)   # [G, T/4, L]
        
        print(f"[VQ ENCODE] Final shapes - WL: {WL.shape}, HL: {HL.shape}")
        
        G, T, Lw = WL.shape
        Lh = HL.shape[2]
        self.vq_meta = {"G": G, "T": T, "Lw": Lw, "Lh": Lh, "T_original": T_original}
        print(f"[VQ ENCODE] Sample tokens - WL[0,0,:5]: {WL[0, 0, :5]}, WR[0,0,:5]: {WR[0, 0, :5]}")
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
        return [seq]
    
    @torch.no_grad()
    def decode(
        self,
        tokens: list[list[int]] | list[int],
    ) -> np.array:
        """
        Decode time-interleaved VQ tokens back to bimanual continuous actions.
        
        Args:
            tokens: List of 1D arrays of VQ token IDs, or single 1D array
            
        Returns:
            Decoded actions: [batch, T, D]
        """
        # Handle both list and single array inputs
        is_batch = isinstance(tokens, list)
        if not is_batch:
            tokens = [tokens]
        
        # Use cached metadata
        meta = self.vq_meta
        G = int(meta["G"])
        T = int(meta["T"])
        Lw = int(meta["Lw"])
        Lh = int(meta["Lh"])
        T_original = int(meta["T_original"]) # Original time horizon before downsampling
        
        # Calculate tokens per timestep
        wrist_per_t = 2 * G * Lw
        hand_per_t = 2 * G * Lh
        tokens_per_t = wrist_per_t + hand_per_t
        
        all_decoded = []
        device = next(self.vq_model["wrist"].parameters()).device
        
        for tokens_1d in tokens:
            # Check if we have enough tokens
            expected = T * tokens_per_t
            current_T = T

            if len(tokens_1d) < expected:
                logger.warning(f"Short sequence: {len(tokens_1d)} < {expected}")
                current_T = len(tokens_1d) // tokens_per_t
            
            # Reconstruct [G, T, L] format for wrist and hand
            wrist_ids_left = np.zeros((G, current_T, Lw), dtype=np.int64)
            wrist_ids_right = np.zeros((G, current_T, Lw), dtype=np.int64)
            hand_ids_left = np.zeros((G, current_T, Lh), dtype=np.int64)
            hand_ids_right = np.zeros((G, current_T, Lh), dtype=np.int64)
            
            for t in range(current_T):
                start = t * tokens_per_t
                timestep_tokens = tokens_1d[start : start + tokens_per_t]
                
                # Extract wrist tokens
                wrist_tokens_t = timestep_tokens[:wrist_per_t]
                for g in range(G):
                    for l in range(Lw):
                        idx = g * Lw + l
                        if idx < len(wrist_tokens_t):
                            wrist_ids_left[g, t, l] = wrist_tokens_t[idx]
                
                for g in range(G):
                    for l in range(Lw):
                        idx = G * Lw + g * Lw + l
                        if idx < len(wrist_tokens_t):
                            wrist_ids_right[g, t, l] = wrist_tokens_t[idx]
                
                # Extract hand tokens
                hand_tokens_t = timestep_tokens[wrist_per_t:]
                for g in range(G):
                    for l in range(Lh):
                        idx = g * Lh + l
                        if idx < len(hand_tokens_t):
                            hand_ids_left[g, t, l] = hand_tokens_t[idx]
                
                for g in range(G):
                    for l in range(Lh):
                        idx = G * Lh + g * Lh + l
                        if idx < len(hand_tokens_t):
                            hand_ids_right[g, t, l] = hand_tokens_t[idx]
            
            # Decode each part
            # forward_decoder expects (G, B, T/4, L), so add batch dim
            print(f"[VQ DECODE] Sample tokens - WL[0,0,:5]: {wrist_ids_left[0, 0, :5]}, WR[0,0,:5]: {wrist_ids_right[0, 0, :5]}")
            wrist_ids_left_tensor = torch.from_numpy(wrist_ids_left).long().to(device).unsqueeze(1)  # (G, 1, T/4, L)
            wrist_ids_right_tensor = torch.from_numpy(wrist_ids_right).long().to(device).unsqueeze(1)
            hand_ids_left_tensor = torch.from_numpy(hand_ids_left).long().to(device).unsqueeze(1)
            hand_ids_right_tensor = torch.from_numpy(hand_ids_right).long().to(device).unsqueeze(1)
            
            # forward_decoder returns (B=1, T_upsampled, D), squeeze batch dim and truncate to T_original
            acts_wrist_left = self.vq_model["wrist"].forward_decoder(wrist_ids_left_tensor).squeeze(0)[:T_original]  # (T_original, D)
            acts_wrist_right = self.vq_model["wrist"].forward_decoder(wrist_ids_right_tensor).squeeze(0)[:T_original]
            acts_hand_left = self.vq_model["hand"].forward_decoder(hand_ids_left_tensor).squeeze(0)[:T_original]
            acts_hand_right = self.vq_model["hand"].forward_decoder(hand_ids_right_tensor).squeeze(0)[:T_original]
            
            # Convert to numpy
            wl_np = acts_wrist_left.detach().cpu().numpy() if isinstance(acts_wrist_left, torch.Tensor) else acts_wrist_left
            wr_np = acts_wrist_right.detach().cpu().numpy() if isinstance(acts_wrist_right, torch.Tensor) else acts_wrist_right
            hl_np = acts_hand_left.detach().cpu().numpy() if isinstance(acts_hand_left, torch.Tensor) else acts_hand_left
            hr_np = acts_hand_right.detach().cpu().numpy() if isinstance(acts_hand_right, torch.Tensor) else acts_hand_right
            
            print(f"[VQ DECODE] Decoded parts:")
            print(f"  wrist_left[0:3, :3]: {wl_np[0:3, :3]}, wrist_right[0:3, :3]: {wr_np[0:3, :3]}")
            
            # Reassemble in original interleaved format:
            # [left_trans(3), right_trans(3), left_rot(6), right_rot(6), hand_left(15), hand_right(15)]
            decoded_actions = np.concatenate([
                wl_np[:, 0:3],   # left translation
                wr_np[:, 0:3],   # right translation
                wl_np[:, 3:9],   # left rotation
                wr_np[:, 3:9],   # right rotation
                hl_np,           # hand_left (15)
                hr_np            # hand_right (15)
            ], axis=-1)  # (T_original, 48)
            
            print(f"[VQ DECODE] Decoded shape: {decoded_actions.shape}")
            
            all_decoded.append(decoded_actions)
        
        # Return stacked for batch, single for non-batch
        result = np.stack(all_decoded)
        print(f"[VQ DECODE] Final output shape: {result.shape}")
        return result
    
    @classmethod
    def from_pretrained(cls, load_directory: str, wrist_dim: int = None, hand_dim: int = None):
        """
        Load VQ tokenizer from directory.
        
        Args:
            load_directory: Path to directory containing 'wrist' and 'hand' subdirectories
            wrist_dim: Dimension of wrist data (if None, read from config)
            hand_dim: Dimension of hand data (if None, read from config)
            
        Returns:
            VQActionProcessor instance with loaded wrist and hand models
        """
        import json
        
        # Load wrist and hand models
        vq_model = {
            "wrist": MotionVQModel.from_pretrained(
                os.path.join(load_directory, "wrist")
            ),
            "hand": MotionVQModel.from_pretrained(
                os.path.join(load_directory, "hand")
            )
        }
        
        # Read dimensions from config if not provided
        if wrist_dim is None or hand_dim is None:
            wrist_config_path = os.path.join(load_directory, "wrist", "config.json")
            with open(wrist_config_path, 'r') as f:
                config = json.load(f)
                if wrist_dim is None:
                    wrist_dim = config.get("wrist_dim", 9)
                if hand_dim is None:
                    hand_dim = config.get("hand_dim", 15)
        
        logger.info(f"VQ tokenizer loaded from {load_directory} (wrist_dim={wrist_dim}, hand_dim={hand_dim})")
        
        return cls(
            vq_model=vq_model,
            wrist_dim=wrist_dim,
            hand_dim=hand_dim,
        )

    
    def setup_tokenizer_gemma_mappings(self, usable_token_ids: list, start_idx: int = 0):
        """
        Build mappings between VQ action tokens and Gemma tokenizer token IDs for this processor.
        
        Args:
            usable_token_ids: List of usable Gemma token IDs
            start_idx: Starting index in usable_token_ids to use
            
        Returns:
            Tuple of (token_id2gemma_token_id, gemma_token_id2token_id, end_idx)
            - token_id2gemma_token_id: {part: {vq_id: gemma_id}}
            - gemma_token_id2token_id: {gemma_id: (part, vq_id)}
            - end_idx: Next index to use in usable_token_ids
        """
        # Initialize mappings for this VQ processor (two-level: part -> vq_id)
        token_id2gemma_token_id = {part_name: {} for part_name in sorted(self.vq_model.keys())}
        gemma_token_id2token_id = {}
        
        replace_idx = start_idx
        for part_name in sorted(self.vq_model.keys()):  # "hand", "wrist"
            model = self.vq_model[part_name]
            vocab_sz = model.vocab_size
            for vq_id in range(vocab_sz):
                gemma_id = usable_token_ids[replace_idx]
                replace_idx += 1
                token_id2gemma_token_id[part_name][vq_id] = gemma_id
                gemma_token_id2token_id[gemma_id] = (part_name, vq_id)
        
        return token_id2gemma_token_id, gemma_token_id2token_id, replace_idx

    def map_hand_tokens2gemma(self, hand_tokens_1d, mapping):
        """
        Map VQ token IDs to Gemma token IDs while preserving time-interleaved order.
        
        Args:
            vq_ids_1d: 1D array of VQ token IDs [t0_wrist, t0_hand, t1_wrist, t1_hand, ...]
            mapping: vq_token_id2gemma_token_id dict
            
        Returns:
            1D array of Gemma token IDs [t0_wrist, t0_hand, t1_wrist, t1_hand, ...]
        """
        G = int(self.vq_meta["G"])
        T = int(self.vq_meta["T"])
        Lw = int(self.vq_meta["Lw"])
        Lh = int(self.vq_meta["Lh"])
        wrist_per_t = 2 * G * Lw  # left + right wrist
        hand_per_t = 2 * G * Lh   # left + right hand
        tokens_per_t = wrist_per_t + hand_per_t
        
        # Step 1: Separate and map wrist/hand tokens, then re-interleave
        mapped_interleaved = []
        
        for t in range(T):
            start = t * tokens_per_t
            
            # Extract and map wrist tokens for this timestep
            wrist_start = start
            wrist_end = start + wrist_per_t
            for vq in hand_tokens_1d[wrist_start:wrist_end]:
                mapped_interleaved.append(mapping['wrist'][int(vq)])
            
            # Extract and map hand tokens for this timestep
            hand_start = start + wrist_per_t
            hand_end = start + tokens_per_t
            for vq in hand_tokens_1d[hand_start:hand_end]:
                mapped_interleaved.append(mapping['hand'][int(vq)])
        
        return np.asarray(mapped_interleaved, dtype=np.int64)