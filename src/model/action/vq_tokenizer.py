# VQ-based Action Tokenizer (替代 fast_tokenizer 的 DCT+BPE)
# Interface compatible with UniversalActionProcessor

import logging
import os
import pickle
from typing import Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers.processing_utils import ProcessorMixin
from src.model.action.vq_model import MotionVQModel
logger = logging.getLogger(__name__)

# support partial bimanual encoding/decoding
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
        down_ratio = 2**vq_model["wrist"].config.model_config.down_t
        self.vq_meta = {
            "Gw": vq_model["wrist"].config.quantizer_config.num_groups,
            "Gh": vq_model["hand"].config.quantizer_config.num_groups,
            "T": (vq_model["wrist"].horizon + down_ratio - 1) // down_ratio, # round up to the nearest integer
            "Lw": vq_model["wrist"].config.quantizer_config.num_quantizers,
            "Lh": vq_model["hand"].config.quantizer_config.num_quantizers,
            "T_original": vq_model["wrist"].horizon,
        }
          
    def __call__(self, action_chunk: Union[list, np.array, torch.Tensor]) -> List[np.ndarray]:
        """
        Encode bimanual actions to discrete token IDs in time-interleaved order.
        
        Args:
            action_chunk: np.ndarray or torch.Tensor with shape [T, D] or [B, T, D]
                or list of np.ndarray or torch.Tensor with shape [T, D]
                where D = 2*wrist_dim + 2*hand_dim
                Layout: [wrist_trans, wrist_rot, left_hand, right_hand]
            
        Returns:
            List of 1D flattened discrete tokens 
        """
        if isinstance(action_chunk, list): 
            if isinstance(action_chunk[0], np.ndarray):
                action_chunk_tensor = np.stack(action_chunk, axis=0)
                action_chunk_tensor = torch.from_numpy(action_chunk_tensor).float()
            elif isinstance(action_chunk[0], torch.Tensor):
                action_chunk_tensor = torch.stack(action_chunk, axis=0)
            else:
                raise ValueError(f"Unsupported type: {type(action_chunk[0])}")
        elif isinstance(action_chunk, np.ndarray):
            action_chunk_tensor = torch.from_numpy(action_chunk).float()
        elif isinstance(action_chunk, torch.Tensor):
            action_chunk_tensor = action_chunk
        else: 
            raise ValueError(f"Unsupported type: {type(action_chunk)}")
        
        if action_chunk_tensor.ndim == 2: 
            action_chunk_tensor = action_chunk_tensor.unsqueeze(0)
        device = next(self.vq_model["wrist"].parameters()).device
        action_chunk_tensor = action_chunk_tensor.to(device)
        
        # Define dimension slices
        w = self.wrist_dim
        h = self.hand_dim
        
        # Split data into 4 parts - expecting [B, T, D] input
        # Data layout: [left_trans(3), right_trans(3), left_rot(6), right_rot(6), hand_left(15), hand_right(15)]
        T_original = action_chunk_tensor.shape[1]  # Save original time horizon
        
        # Reconstruct wrist_left = [left_trans + left_rot], wrist_right = [right_trans + right_rot]
        wrist_left_data = torch.cat([
            action_chunk_tensor[..., 0:3],    # left translation
            action_chunk_tensor[..., 6:12]    # left rotation
        ], dim=-1)  # [B, T, 9]
        
        wrist_right_data = torch.cat([
            action_chunk_tensor[..., 3:6],    # right translation
            action_chunk_tensor[..., 12:18]   # right rotation
        ], dim=-1)  # [B, T, 9]
        
        hand_left_data = action_chunk_tensor[..., 18:18+h]                # [B, T, h]
        hand_right_data = action_chunk_tensor[..., 18+h:18+2*h]           # [B, T, h]
        print(f"[VQ ENCODE] Ground Truth:")
        print(f"[VQ ENCODE] wrist_left[0, 0, :3]: {wrist_left_data[0, 0, :3]}, wrist_right[0, 0, :3]: {wrist_right_data[0, 0, :3]}")
        
        # Encode each part using VQ model's encode method: returns [G, B, T/4, L]
        print(f"[VQ ENCODE] Input shapes - wrist_left: {wrist_left_data.shape}, hand_left: {hand_left_data.shape}")
        wrist_left_raw = self.vq_model["wrist"].encode(wrist_left_data) # [G, B, T/4, Lw]   
        wrist_right_raw = self.vq_model['wrist'].encode(wrist_right_data) # [G, B, T/4, Lw]
        hand_left_raw = self.vq_model['hand'].encode(hand_left_data) # [G, B, T/4, Lh]
        hand_right_raw = self.vq_model['hand'].encode(hand_right_data) # [G, B, T/4, Lh]
        
        print(f"[VQ ENCODE] Raw output shapes - wrist_left: {wrist_left_raw.shape}, hand_left: {hand_left_raw.shape}")
        
        # Flatten with time-interleaved order and get metadata
        Gw, B, T, Lw = wrist_left_raw.shape
        Gh, B, T, Lh = hand_left_raw.shape
        # assert T == self.vq_meta["T"], f"T mismatch: {T} != {self.vq_meta['T']}"
        assert Gw == self.vq_meta["Gw"], f"Gw mismatch: {Gw} != {self.vq_meta['Gw']}"
        assert Gh == self.vq_meta["Gh"], f"Gh mismatch: {Gh} != {self.vq_meta['Gh']}"
        assert Lw == self.vq_meta["Lw"], f"Lw mismatch: {Lw} != {self.vq_meta['Lw']}"
        assert Lh == self.vq_meta["Lh"], f"Lh mismatch: {Lh} != {self.vq_meta['Lh']}"
        # assert T_original == self.vq_meta["T_original"], f"T_original mismatch: {T_original} != {self.vq_meta['T_original']}"
        print(f"[VQ ENCODE] Sample tokens - WL[0,0,0,:5]: {wrist_left_raw[0, 0, 0, :5]}, WR[0,0,0,:5]: {wrist_right_raw[0, 0, 0, :5]}")

        WL = rearrange(wrist_left_raw, 'gw b t lw -> b t (gw lw)').contiguous()   # [Gw, B, T/4, Lw] -> [B, T/4, Gw*Lw]
        WR = rearrange(wrist_right_raw, 'gw b t lw -> b t (gw lw)').contiguous()  # [Gw, B, T/4, Lw] -> [B, T/4, Gw*Lw]
        HL = rearrange(hand_left_raw, 'gh b t lh -> b t (gh lh)').contiguous()    # [Gh, B, T/4, Lh] -> [B, T/4, Gh*Lh]
        HR = rearrange(hand_right_raw, 'gh b t lh -> b t (gh lh)').contiguous()   # [Gh, B, T/4, Lh] -> [B, T/4, Gh*Lh]
        print(f"[VQ ENCODE] Final shapes - WL: {WL.shape}, HL: {HL.shape}, HR: {HR.shape}, WR: {WR.shape}")
        total_tokens = torch.cat([WL, WR, HL, HR], dim=-1).flatten(start_dim=1) # [B, T/4 * (Gw*Lw + Gh*Lh)*2]
        return total_tokens.tolist()
    
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
            Decoded actions: np.ndarray[batch, T, D]
        """
        # Handle both list and single array inputs
        assert isinstance(tokens, (list, np.ndarray, torch.Tensor)), f"Unsupported type: {type(tokens)}"
        if isinstance(tokens, np.ndarray):
            tokens = torch.from_numpy(tokens)
        elif isinstance(tokens, list):
            if isinstance(tokens[0], list):
                tokens = torch.tensor(tokens)
            elif isinstance(tokens[0], np.ndarray):
                tokens = np.stack(tokens, axis=0)
                tokens = torch.from_numpy(tokens)
            elif isinstance(tokens[0], torch.Tensor):
                tokens = torch.stack(tokens, axis=0)
            else:
                raise ValueError(f"Unsupported type: {type(tokens[0])}")

        device = next(self.vq_model["wrist"].parameters()).device
        tokens = tokens.to(device).long()
        
        if tokens.ndim == 1: 
            tokens = tokens.unsqueeze(0)
        assert tokens.ndim == 2, f"Tokens must have shape [B, N], got {tokens.shape}"
        
        B, N = tokens.shape
        # Use cached metadata
        Gw = self.vq_meta["Gw"]
        Gh = self.vq_meta["Gh"]
        T = self.vq_meta["T"]
        Lw = self.vq_meta["Lw"]
        Lh = self.vq_meta["Lh"]
        T_original = self.vq_meta["T_original"] # Original time horizon before downsampling
        
        # Calculate tokens per timestep
        wrist_per_t = Gw * Lw
        hand_per_t = Gh * Lh
        
        tokens = rearrange(tokens, 'b (t n1) -> b t n1', t = T) # [B, T*(Gw*Lw + Gh*Lh)*2] -> [B, T, (Gw*Lw + Gh*Lh)*2]
        WL = tokens[..., :wrist_per_t] # [B, T, Gw*Lw]
        WR = tokens[..., wrist_per_t:wrist_per_t*2] # [B, T, Gw*Lw]
        HL = tokens[..., wrist_per_t*2:wrist_per_t*2+hand_per_t] # [B, T, Gh*Lh]
        HR = tokens[..., wrist_per_t*2+hand_per_t:] # [B, T, Gh*Lh]

        WL = rearrange(WL, 'b t (gw lw) -> gw b t lw', gw = Gw) # [B, T, Gw*Lw] -> [Gw, B, T, Lw]
        WR = rearrange(WR, 'b t (gw lw) -> gw b t lw', gw = Gw) # [B, T, Gw*Lw] -> [Gw, B, T, Lw]
        HL = rearrange(HL, 'b t (gh lh) -> gh b t lh', gh = Gh) # [B, T, Gh*Lh] -> [Gh, B, T, Lh]
        HR = rearrange(HR, 'b t (gh lh) -> gh b t lh', gh = Gh) # [B, T, Gh*Lh] -> [Gh, B, T, Lh]

        wrist_left_data = self.vq_model["wrist"].forward_decoder(WL)[:, :T_original, :] # [Gw, B, T, Lw] -> [B, T_original, wrist_dim]
        wrist_right_data = self.vq_model["wrist"].forward_decoder(WR)[:, :T_original, :] # [Gw, B, T, Lw] -> [B, T_original, wrist_dim]
        hand_left_data = self.vq_model["hand"].forward_decoder(HL)[:, :T_original, :] # [Gh, B, T, Lh] -> [B, T_original, hand_dim]
        hand_right_data = self.vq_model["hand"].forward_decoder(HR)[:, :T_original, :] # [Gh, B, T, Lh] -> [B, T_original, hand_dim]

        decoded_actions = torch.cat([
            wrist_left_data[..., :3], 
            wrist_right_data[..., :3], 
            wrist_left_data[..., 3:9], 
            wrist_right_data[..., 3:9], 
            hand_left_data, 
            hand_right_data
        ], dim=-1) # [B, T_original, total_dim]
        
        return decoded_actions.cpu().numpy()
    
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
        Gw = int(self.vq_meta["Gw"])
        Gh = int(self.vq_meta["Gh"])
        T = int(self.vq_meta["T"])
        Lw = int(self.vq_meta["Lw"])
        Lh = int(self.vq_meta["Lh"])
        bimanual_wrist_per_t = 2 * Gw * Lw  # left + right wrist
        bimanual_hand_per_t = 2 * Gh * Lh   # left + right hand
        tokens_per_t = bimanual_wrist_per_t + bimanual_hand_per_t
        
        # Step 1: Separate and map wrist/hand tokens, then re-interleave
        mapped_interleaved = []
        
        for t in range(T):
            start = t * tokens_per_t
            
            # Extract and map wrist tokens for this timestep
            wrist_start = start
            wrist_end = start + bimanual_wrist_per_t
            for vq in hand_tokens_1d[wrist_start:wrist_end]:
                mapped_interleaved.append(mapping['wrist'][int(vq)])
            
            # Extract and map hand tokens for this timestep
            hand_start = start + bimanual_wrist_per_t
            hand_end = start + tokens_per_t
            for vq in hand_tokens_1d[hand_start:hand_end]:
                mapped_interleaved.append(mapping['hand'][int(vq)])
        
        return np.asarray(mapped_interleaved, dtype=np.int64)
