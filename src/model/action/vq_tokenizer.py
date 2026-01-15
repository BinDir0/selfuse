# VQ-based Action Tokenizer (Replacement for fast_tokenizer's DCT+BPE)
# Interface compatible with UniversalActionProcessor
#
# Variable Naming Convention:
# ============================
# - B: batch size
# - T: time steps (after downsampling)
# - T_original: original time steps (before downsampling)
# - D: feature dimension
# - G: groups (number of groups for Grouped Residual VQ)
# - L: layers/quantizers (number of quantizer layers)
# - Gw: wrist groups (number of wrist groups)
# - Gh: hand groups (number of hand groups)
# - Lw: wrist quantizers (number of wrist quantizer layers)
# - Lh: hand quantizers (number of hand quantizer layers)
# - w: wrist_dim (wrist data dimension, typically 9: 3 translation + 6 rotation)
# - h: hand_dim (hand data dimension, typically 15)
#
# Data Layout Specification:
# ==========================
# Input action data layout: [B, T_original, D_total]
#   D_total = 2*wrist_dim + 2*hand_dim = 2*9 + 2*15 = 48
#   Specific order: [left_trans(3), right_trans(3), left_rot(6), right_rot(6), 
#                    hand_left(15), hand_right(15)]
#
# Encoded token layout: [B, T, tokens_per_timestep]
#   tokens_per_timestep = 2*(Gw*Lw) + 2*(Gh*Lh)
#   Time-interleaved order: [t0_wrist_left, t0_wrist_right, t0_hand_left, t0_hand_right,
#                            t1_wrist_left, t1_wrist_right, t1_hand_left, t1_hand_right, ...]

import logging
import os
from typing import Union, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers.processing_utils import ProcessorMixin
from src.model.action.vq_model import MotionVQModel

logger = logging.getLogger(__name__)


class VQActionProcessor(ProcessorMixin):
    """
    VQ-based action tokenizer for bimanual manipulation tasks.
    
    This processor encodes continuous bimanual actions (wrist poses + hand poses) 
    into discrete token sequences using VQ-VAE models. It supports separate models
    for wrist and hand data, enabling fine-grained control.
    
    Interface compatible with UniversalActionProcessor:
    - __call__(actions) -> list of token sequences
    - decode(token_sequences) -> actions
    - fit() for training
    - save_pretrained() / from_pretrained() for persistence
    
    Attributes:
        vq_model: Dict[str, MotionVQModel] - Dictionary with 'wrist' and 'hand' keys
        wrist_dim: int - Dimension of wrist data (typically 9: 3 trans + 6 rot)
        hand_dim: int - Dimension of hand data (typically 15)
        vocab_size: int - Total vocabulary size (sum of wrist and hand vocab sizes)
        vq_meta: Dict[str, int] - Metadata about VQ model structure
    """
    
    def __init__(
        self,
        vq_model: Dict[str, MotionVQModel],
        wrist_dim: int,
        hand_dim: int,
    ):
        """
        Initialize VQ action processor.
        
        Args:
            vq_model: Dictionary with 'wrist' and 'hand' keys, each containing a MotionVQModel.
                      The models should be trained separately for wrist and hand data.
            wrist_dim: Dimension of wrist data (typically 9: 3 translation + 6 rotation)
            hand_dim: Dimension of hand data (typically 15 for MANO hand parameters)
        """
        if not isinstance(vq_model, dict):
            raise ValueError("vq_model must be a dictionary with 'wrist' and 'hand' keys")
        if "wrist" not in vq_model or "hand" not in vq_model:
            raise ValueError("vq_model must contain both 'wrist' and 'hand' keys")
        
        self.vq_model = vq_model
        self.wrist_dim = wrist_dim
        self.hand_dim = hand_dim
        
        # Calculate total vocabulary size
        self.vocab_size = sum([model.vocab_size for model in self.vq_model.values()])
        
        # Extract and cache VQ model metadata for efficient encoding/decoding
        self._initialize_vq_metadata()
    
    def _initialize_vq_metadata(self):
        """
        Initialize and cache VQ model metadata (static configuration only).
        
        Note: Time-related metadata (T, T_original) are computed dynamically during encoding
        to support variable-length sequences and different state/action lengths.
        """
        wrist_model = self.vq_model["wrist"]
        hand_model = self.vq_model["hand"]
        
        # Extract quantizer configuration
        wrist_quantizer_config = wrist_model.config.quantizer_config
        hand_quantizer_config = hand_model.config.quantizer_config
        
        # Cache static metadata only (time-related metadata computed dynamically)
        self.vq_meta = {
            "Gw": wrist_quantizer_config.num_groups,  # Wrist groups
            "Gh": hand_quantizer_config.num_groups,   # Hand groups
            "Lw": wrist_quantizer_config.num_quantizers,  # Wrist quantizer layers
            "Lh": hand_quantizer_config.num_quantizers,   # Hand quantizer layers
            "stride_t": wrist_model.config.model_config.stride_t,
            "down_t": wrist_model.config.model_config.down_t,
        }
        
        # Calculate downsampling ratio for reference
        self.vq_meta["down_ratio"] = self.vq_meta["stride_t"] ** self.vq_meta["down_t"]
    
    def _unify_input_to_tensor(
        self, 
        action_chunk: Union[List, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        unify input to torch.Tensor with shape [B, T, D].
        
        Args:
            action_chunk: Input in various formats:
                - List of arrays/tensors with shape [T, D]
                - np.ndarray with shape [T, D] or [B, T, D]
                - torch.Tensor with shape [T, D] or [B, T, D]
        
        Returns:
            unified tensor with shape [B, T, D]
        """
        if isinstance(action_chunk, list):
            if len(action_chunk) == 0:
                raise ValueError("Empty list provided")
            if isinstance(action_chunk[0], np.ndarray):
                action_tensor = np.stack(action_chunk, axis=0)
                action_tensor = torch.from_numpy(action_tensor).float()
            elif isinstance(action_chunk[0], torch.Tensor):
                action_tensor = torch.stack(action_chunk, axis=0)
            else:
                raise ValueError(f"Unsupported list element type: {type(action_chunk[0])}")
        elif isinstance(action_chunk, np.ndarray):
            action_tensor = torch.from_numpy(action_chunk).float()
        elif isinstance(action_chunk, torch.Tensor):
            action_tensor = action_chunk.float()
        else:
            raise ValueError(f"Unsupported input type: {type(action_chunk)}")
        
        # Ensure batch dimension exists
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)
        
        if action_tensor.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got {action_tensor.ndim}D")
        
        return action_tensor
    
    def _split_bimanual_actions(
        self, 
        action_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split bimanual action tensor into wrist and hand parts, with left and right concatenated.
        
        Args:
            action_tensor: [B, T_original, D_total] where D_total = 2*wrist_dim + 2*hand_dim
                Layout: [left_trans(3), right_trans(3), left_rot(6), right_rot(6),
                        hand_left(15), hand_right(15)]
        
        Returns:
            Tuple of (wrist_bimanual, hand_bimanual) tensors
            - wrist_bimanual: [B, T_original, 2*wrist_dim] = [left_wrist, right_wrist] concatenated
            - hand_bimanual: [B, T_original, 2*hand_dim] = [left_hand, right_hand] concatenated
        """
        B, T_original, D_total = action_tensor.shape
        
        # Expected total dimension
        expected_dim = 2 * self.wrist_dim + 2 * self.hand_dim
        if D_total != expected_dim:
            raise ValueError(
                f"Expected action dimension {expected_dim} (2*{self.wrist_dim} + 2*{self.hand_dim}), "
                f"got {D_total}"
            )
        
        # Extract components
        # Left translation: indices 0:3
        left_trans = action_tensor[..., 0:3]
        # Right translation: indices 3:6
        right_trans = action_tensor[..., 3:6]
        # Left rotation: indices 6:12
        left_rot = action_tensor[..., 6:12]
        # Right rotation: indices 12:18
        right_rot = action_tensor[..., 12:18]
        # Left hand: indices 18:18+hand_dim
        hand_left = action_tensor[..., 18:18+self.hand_dim]
        # Right hand: indices 18+hand_dim:18+2*hand_dim
        hand_right = action_tensor[..., 18+self.hand_dim:18+2*self.hand_dim]
        
        # Reconstruct wrist data (translation + rotation) and concatenate left and right
        wrist_left = torch.cat([left_trans, left_rot], dim=-1)   # [B, T_original, wrist_dim]
        wrist_right = torch.cat([right_trans, right_rot], dim=-1)  # [B, T_original, wrist_dim]
        wrist_bimanual = torch.cat([wrist_left, wrist_right], dim=-1)  # [B, T_original, 2*wrist_dim]
        
        # Concatenate left and right hand data
        hand_bimanual = torch.cat([hand_left, hand_right], dim=-1)  # [B, T_original, 2*hand_dim]
        
        return wrist_bimanual, hand_bimanual
    
    def _encode_with_vq_model(
        self, 
        model: MotionVQModel, 
        state_data: torch.Tensor,
        action_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Encode state and action data using VQ model with joint tokenization.
        
        Args:
            model: MotionVQModel instance
            state_data: [B, T_state_original, D] state data (can have variable length)
            action_data: [B, T_action_original, D] action data (can have variable length)
        
        Returns:
            Tuple of (state_indices, action_indices, T_state_original, T_action_original)
            - state_indices: [G, B, T_state_downsampled, L] quantized state token indices
            - action_indices: [G, B, T_action_downsampled, L] quantized action token indices
            - T_state_original: Original state time horizon
            - T_action_original: Original action time horizon
        """
        # Store original time horizons
        T_state_original = state_data.shape[1]
        T_action_original = action_data.shape[1]
        
        # Jointly encode state and action: returns (state_indices, action_indices)
        # Note: state_indices and action_indices may have different time dimensions
        state_indices, action_indices = model.model.encode(state_data, action_data)
        
        return state_indices, action_indices, T_state_original, T_action_original
    
    def _decode_with_vq_model(
        self, 
        model: MotionVQModel, 
        state_indices: torch.Tensor,
        action_indices: torch.Tensor,
        T_state_original: int,
        T_action_original: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode state and action indices using VQ model.
        
        Note: The model's decode returns padded values, so we need to slice using T_original.
        
        Args:
            model: MotionVQModel instance
            state_indices: [G, B, T_state_downsampled, L] quantized state token indices
            action_indices: [G, B, T_action_downsampled, L] quantized action token indices
            T_state_original: Original state time horizon (before padding/downsampling)
            T_action_original: Original action time horizon (before padding/downsampling)
        
        Returns:
            Tuple of (state_decoded, action_decoded)
            - state_decoded: [B, T_state_original, D] decoded state data (sliced from padded output)
            - action_decoded: [B, T_action_original, D] decoded action data (sliced from padded output)
        """
        # Jointly decode state and action indices: returns (state_decoded, action_decoded)
        # Note: The model returns padded values, so we need to slice to get original length
        state_decoded_padded, action_decoded_padded = model.model.decode(state_indices, action_indices)
        
        # Slice to original time horizon
        state_decoded = state_decoded_padded[:, :T_state_original, :]
        action_decoded = action_decoded_padded[:, :T_action_original, :]
        
        return state_decoded, action_decoded
    
    def __call__(
        self, 
        state_chunk: Union[List, np.ndarray, torch.Tensor],
        action_chunk: Union[List, np.ndarray, torch.Tensor]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Encode bimanual state and action pairs to discrete token IDs with joint tokenization.
        
        Supports variable-length sequences and different state/action time horizons.
        
        Args:
            state_chunk: Input states in various formats:
                - List of arrays/tensors with shape [T_state, D]
                - np.ndarray with shape [T_state, D] or [B, T_state, D]
                - torch.Tensor with shape [T_state, D] or [B, T_state, D]
                where D = 2*wrist_dim + 2*hand_dim
                Layout: [left_trans(3), right_trans(3), left_rot(6), right_rot(6),
                        hand_left(15), hand_right(15)]
            action_chunk: Input actions in same format as state_chunk, but can have different T_action
        
        Returns:
            Tuple of (state_tokens, action_tokens, metadata)
            - state_tokens: List of 1D flattened discrete state token sequences (one per batch item)
            - action_tokens: List of 1D flattened discrete action token sequences (one per batch item)
            - metadata: Dict containing time information:
                - "T_state_original": Original state time horizon
                - "T_action_original": Original action time horizon
            Each sequence length depends on the original time horizon and downsampling ratio.
        """
        # Unify inputs to tensors (state and action can have different time lengths)
        state_tensor = self._unify_input_to_tensor(state_chunk)
        action_tensor = self._unify_input_to_tensor(action_chunk)
        
        # Verify batch size and feature dimension match
        if state_tensor.shape[0] != action_tensor.shape[0]:
            raise ValueError(
                f"Batch size mismatch: state={state_tensor.shape[0]}, "
                f"action={action_tensor.shape[0]}"
            )
        if state_tensor.shape[2] != action_tensor.shape[2]:
            raise ValueError(
                f"Feature dimension mismatch: state={state_tensor.shape[2]}, "
                f"action={action_tensor.shape[2]}"
            )
        # Time dimensions can differ, so we don't check them
        
        # Move to device
        device = next(self.vq_model["wrist"].parameters()).device
        state_tensor = state_tensor.to(device)
        action_tensor = action_tensor.to(device)
        
        # Split into wrist and hand parts (left and right concatenated in feature dimension)
        state_wrist_bimanual, state_hand_bimanual = self._split_bimanual_actions(state_tensor)
        action_wrist_bimanual, action_hand_bimanual = self._split_bimanual_actions(action_tensor)
        
        # Jointly encode wrist and hand parts using VQ models
        # Returns: (state_indices, action_indices, T_state_original, T_action_original)
        wrist_state_idx, wrist_action_idx, T_state_original, T_action_original = \
            self._encode_with_vq_model(self.vq_model["wrist"], state_wrist_bimanual, action_wrist_bimanual)
        hand_state_idx, hand_action_idx, _, _ = \
            self._encode_with_vq_model(self.vq_model["hand"], state_hand_bimanual, action_hand_bimanual)
        
        # Verify batch size and groups match metadata
        Gw, B, T_state_wrist_down, Lw = wrist_state_idx.shape
        Gh, B_check, T_state_hand_down, Lh = hand_state_idx.shape
        
        assert B == B_check, f"Batch size mismatch: {B} != {B_check}"
        assert Gw == self.vq_meta["Gw"], f"Wrist groups mismatch: {Gw} != {self.vq_meta['Gw']}"
        assert Gh == self.vq_meta["Gh"], f"Hand groups mismatch: {Gh} != {self.vq_meta['Gh']}"
        assert Lw == self.vq_meta["Lw"], f"Wrist quantizers mismatch: {Lw} != {self.vq_meta['Lw']}"
        assert Lh == self.vq_meta["Lh"], f"Hand quantizers mismatch: {Lh} != {self.vq_meta['Lh']}"
        
        # Flatten groups and quantizers: [G, B, T, L] -> [B, T, G*L]
        def flatten_indices(indices, groups):
            """Flatten indices from [G, B, T, L] -> [B, T, G*L]"""
            return rearrange(indices, 'g b t l -> b t (g l)').contiguous()
        
        # Flatten state and action indices
        state_wrist_flat = flatten_indices(wrist_state_idx, Gw)  # [B, T_state_wrist_down, Gw*Lw]
        state_hand_flat = flatten_indices(hand_state_idx, Gh)  # [B, T_state_hand_down, Gh*Lh]
        action_wrist_flat = flatten_indices(wrist_action_idx, Gw)  # [B, T_action_wrist_down, Gw*Lw]
        action_hand_flat = flatten_indices(hand_action_idx, Gh)  # [B, T_action_hand_down, Gh*Lh]
        
        # Flatten time dimension for each part type
        # [B, T_state_downsampled, Gw*Lw + Gh*Lh]
        state_tokens_flat = torch.cat([state_wrist_flat, state_hand_flat], dim=-1).flatten(start_dim=1)
        # [B, T_action_downsampled, Gw*Lw + Gh*Lh]
        action_tokens_flat = torch.cat([action_wrist_flat, action_hand_flat], dim=-1).flatten(start_dim=1)
        
        # Prepare metadata with original time horizons
        metadata = {
            "T_state_original": T_state_original,
            "T_action_original": T_action_original,
        }
        
        # Convert to list of lists
        return state_tokens_flat.tolist(), action_tokens_flat.tolist(), metadata
    
    def _unify_tokens_to_tensor(
        self,
        tokens: Union[List[List[int]], List[int], np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Unify token sequences to torch.Tensor with shape [B, N].
        
        Args:
            tokens: Token sequences in various formats:
                - List of lists of integers: [[token1, token2, ...], ...]
                - List of integers: [token1, token2, ...] (single sequence)
                - np.ndarray: shape [B, N] or [N]
                - torch.Tensor: shape [B, N] or [N]
        
        Returns:
            Unified tensor with shape [B, N]
        """
        if isinstance(tokens, np.ndarray):
            tokens_tensor = torch.from_numpy(tokens)
        elif isinstance(tokens, list):
            if len(tokens) == 0:
                raise ValueError("Empty token list provided")
            if isinstance(tokens[0], list):
                tokens_tensor = torch.tensor(tokens)
            elif isinstance(tokens[0], np.ndarray):
                tokens_tensor = torch.from_numpy(np.stack(tokens, axis=0))
            elif isinstance(tokens[0], (int, np.integer)):
                tokens_tensor = torch.tensor(tokens).unsqueeze(0)
            elif isinstance(tokens[0], torch.Tensor):
                tokens_tensor = torch.stack(tokens, axis=0)
            else:
                raise ValueError(f"Unsupported token list element type: {type(tokens[0])}")
        elif isinstance(tokens, torch.Tensor):
            tokens_tensor = tokens
        else:
            raise ValueError(f"Unsupported token type: {type(tokens)}")
        
        # Ensure batch dimension exists
        if tokens_tensor.ndim == 1:
            tokens_tensor = tokens_tensor.unsqueeze(0)
        
        if tokens_tensor.ndim != 2:
            raise ValueError(f"Expected 1D or 2D token tensor, got {tokens_tensor.ndim}D")
        
        return tokens_tensor
    
    def _split_interleaved_tokens(
        self,
        tokens_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape flattened tokens back to [G, B, T, L] format for decoding.
        
        Args:
            tokens_tensor: [B, N] flattened tokens, where N = T_downsampled * (Gw*Lw + Gh*Lh)
        
        Returns:
            Tuple of (wrist_tokens, hand_tokens)
            - wrist_tokens: [B, T_downsampled, Gw*Lw]
            - hand_tokens: [B, T_downsampled, Gh*Lh]
        """
        B, N = tokens_tensor.shape
        
        # Get metadata
        Gw = self.vq_meta["Gw"]
        Gh = self.vq_meta["Gh"]
        Lw = self.vq_meta["Lw"]
        Lh = self.vq_meta["Lh"]
        
        # Tokens per timestep
        tokens_per_timestep = Gw * Lw + Gh * Lh
        
        # Infer time dimension from total length
        # N = T_downsampled * tokens_per_timestep
        if N % tokens_per_timestep != 0:
            raise ValueError(f"Token length {N} is not divisible by tokens_per_timestep {tokens_per_timestep} ")
        T_downsampled = N // tokens_per_timestep
        
        # Reshape: [B, T_downsampled * tokens_per_timestep] -> [B, T_downsampled, tokens_per_timestep]
        tokens_reshaped = rearrange(tokens_tensor, 'b (t n) -> b t n', t=T_downsampled)
        
        # Reshape back to [G, B, T_downsampled, L] format for decoding
        wrist_tokens = tokens_reshaped[:, :, :Gw*Lw]
        hand_tokens = tokens_reshaped[:, :, Gw*Lw:]
        
        return wrist_tokens, hand_tokens
    
    def _reconstruct_bimanual_data(
        self,
        wrist_bimanual_data: torch.Tensor,
        hand_bimanual_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct bimanual data from wrist and hand parts to original format.
        
        Args:
            wrist_bimanual_data: [B, T_original, 2*wrist_dim] = [left_wrist, right_wrist] concatenated
            hand_bimanual_data: [B, T_original, 2*hand_dim] = [left_hand, right_hand] concatenated
        
        Returns:
            Reconstructed data: [B, T_original, D_total]
            Layout: [left_trans(3), right_trans(3), left_rot(6), right_rot(6),
                    hand_left(15), hand_right(15)]
        """
        # Split wrist data into left and right
        wrist_left = wrist_bimanual_data[..., :self.wrist_dim]  # [B, T_original, wrist_dim]
        wrist_right = wrist_bimanual_data[..., self.wrist_dim:]  # [B, T_original, wrist_dim]
        
        # Split hand data into left and right
        hand_left = hand_bimanual_data[..., :self.hand_dim]  # [B, T_original, hand_dim]
        hand_right = hand_bimanual_data[..., self.hand_dim:]  # [B, T_original, hand_dim]
        
        # Reconstruct original format
        return torch.cat([
            wrist_left[..., :3],      # left translation
            wrist_right[..., :3],     # right translation
            wrist_left[..., 3:9],     # left rotation
            wrist_right[..., 3:9],    # right rotation
            hand_left,                # left hand
            hand_right                # right hand
        ], dim=-1)  # [B, T_original, D_total]
    
    @torch.no_grad()
    def decode(
        self,
        state_tokens: Union[List[List[int]], List[int], np.ndarray, torch.Tensor],
        action_tokens: Union[List[List[int]], List[int], np.ndarray, torch.Tensor],
        T_state_original: int = None,
        T_action_original: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode VQ tokens back to bimanual continuous state and action pairs.
        
        Supports variable-length sequences. Tokens structure: [wrist_tokens, hand_tokens]
        where wrist_tokens = [left, right] flattened, hand_tokens = [left, right] flattened.
        
        Args:
            state_tokens: State token sequences in various formats:
                - List of lists of integers: [[token1, token2, ...], ...]
                - List of integers: [token1, token2, ...] (single sequence)
                - np.ndarray: shape [B, N] or [N]
                - torch.Tensor: shape [B, N] or [N]
                where N = T_state_wrist_down * 2*Gw*Lw + T_state_hand_down * 2*Gh*Lh
            action_tokens: Action token sequences in same format as state_tokens
                where N = T_action_wrist_down * 2*Gw*Lw + T_action_hand_down * 2*Gh*Lh
            T_state_original: Original state time horizon (required for proper slicing).
                If None, will try to infer from tokens, but may be inaccurate.
            T_action_original: Original action time horizon (required for proper slicing).
                If None, will try to infer from tokens, but may be inaccurate.
        
        Returns:
            Tuple of (decoded_states, decoded_actions)
            - decoded_states: np.ndarray[B, T_state_original, D_total]
            - decoded_actions: np.ndarray[B, T_action_original, D_total]
            where D_total = 2*wrist_dim + 2*hand_dim
            Note: T_state_original and T_action_original may differ
        """
        # Unify tokens to tensors
        state_tokens_tensor = self._unify_tokens_to_tensor(state_tokens)
        action_tokens_tensor = self._unify_tokens_to_tensor(action_tokens)
        
        # Verify batch size matches
        if state_tokens_tensor.shape[0] != action_tokens_tensor.shape[0]:
            raise ValueError(
                f"Batch size mismatch: state={state_tokens_tensor.shape[0]}, "
                f"action={action_tokens_tensor.shape[0]}"
            )
        
        # Move to device and ensure integer type
        device = next(self.vq_model["wrist"].parameters()).device
        state_tokens_tensor = state_tokens_tensor.to(device).long()
        action_tokens_tensor = action_tokens_tensor.to(device).long()
        
        # Split tokens into wrist and hand parts
        # Structure: [wrist_tokens, hand_tokens]
        state_wrist_idx, state_hand_idx = self._split_interleaved_tokens(state_tokens_tensor)
        action_wrist_idx, action_hand_idx = self._split_interleaved_tokens(action_tokens_tensor)
        
        # Infer T_original if not provided
        if T_state_original is None or T_action_original is None:
            raise ValueError("T_state_original and T_action_original must be provided")
        
        # Decode wrist and hand parts
        state_wrist_decoded, action_wrist_decoded = self._decode_with_vq_model(
            self.vq_model["wrist"],
            state_wrist_idx,
            action_wrist_idx,
            T_state_original,
            T_action_original
        )
        
        state_hand_decoded, action_hand_decoded = self._decode_with_vq_model(
            self.vq_model["hand"],
            state_hand_idx,
            action_hand_idx,
            T_state_original,
            T_action_original
        )
        
        # Reconstruct original format
        decoded_states = self._reconstruct_bimanual_data(state_wrist_decoded, state_hand_decoded)
        decoded_actions = self._reconstruct_bimanual_data(action_wrist_decoded, action_hand_decoded)
        
        return decoded_states.cpu().numpy(), decoded_actions.cpu().numpy()
    
    @classmethod
    def from_pretrained(
        cls, 
        load_directory: str, 
        wrist_dim: int = None, 
        hand_dim: int = None
    ) -> "VQActionProcessor":
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
        wrist_model_path = os.path.join(load_directory, "wrist")
        hand_model_path = os.path.join(load_directory, "hand")
        
        if not os.path.exists(wrist_model_path):
            raise FileNotFoundError(f"Wrist model not found at {wrist_model_path}")
        if not os.path.exists(hand_model_path):
            raise FileNotFoundError(f"Hand model not found at {hand_model_path}")
        
        vq_model = {
            "wrist": MotionVQModel.from_pretrained(wrist_model_path),
            "hand": MotionVQModel.from_pretrained(hand_model_path)
        }
        
        # Read dimensions from config if not provided
        if wrist_dim is None or hand_dim is None:
            wrist_config_path = os.path.join(wrist_model_path, "config.json")
            if os.path.exists(wrist_config_path):
                with open(wrist_config_path, 'r') as f:
                    config = json.load(f)
                    if wrist_dim is None:
                        wrist_dim = config.get("wrist_dim", 9)
                    if hand_dim is None:
                        hand_dim = config.get("hand_dim", 15)
            else:
                logger.warning(f"Config file not found at {wrist_config_path}, using defaults")
                if wrist_dim is None:
                    wrist_dim = 9
                if hand_dim is None:
                    hand_dim = 15
        
        logger.info(
            f"VQ tokenizer loaded from {load_directory} "
            f"(wrist_dim={wrist_dim}, hand_dim={hand_dim})"
        )
        
        return cls(
            vq_model=vq_model,
            wrist_dim=wrist_dim,
            hand_dim=hand_dim,
        )
    
    def setup_tokenizer_gemma_mappings(
        self, 
        usable_token_ids: List[int], 
        start_idx: int = 0
    ) -> Tuple[Dict[str, Dict[int, int]], Dict[int, int], int]:
        """
        Build mappings between VQ action tokens and Gemma tokenizer token IDs.
        
        Args:
            usable_token_ids: List of usable Gemma token IDs
            start_idx: Starting index in usable_token_ids to use
        
        Returns:
            Tuple of (token_id2gemma_token_id, gemma_token_id2token_id, end_idx)
            - token_id2gemma_token_id: {part_name: {vq_id: gemma_id}}
            - gemma_token_id2token_id: {gemma_id: vq_id}
            - end_idx: Next index to use in usable_token_ids
        """
        # Initialize mappings (two-level: part -> vq_id -> gemma_id)
        token_id2gemma_token_id = {
            part_name: {} 
            for part_name in sorted(self.vq_model.keys())
        }
        gemma_token_id2token_id = {}
        
        replace_idx = start_idx
        for part_name in sorted(self.vq_model.keys()):  # "hand", "wrist"
            model = self.vq_model[part_name]
            vocab_size = model.vocab_size
            
            for vq_id in range(vocab_size):
                if replace_idx >= len(usable_token_ids):
                    raise ValueError(
                        f"Not enough usable token IDs. Need {vocab_size} for {part_name}, "
                        f"but only {len(usable_token_ids) - start_idx} available starting from {start_idx}"
                    )
                
                gemma_id = usable_token_ids[replace_idx]
                token_id2gemma_token_id[part_name][vq_id] = gemma_id
                gemma_token_id2token_id[gemma_id] = vq_id
                replace_idx += 1
        
        return token_id2gemma_token_id, gemma_token_id2token_id, replace_idx
    
    def map_motion_tokens2gemma(
        self, 
        motion_tokens_1d: np.ndarray, 
        mapping: Dict[str, Dict[int, int]]
    ) -> np.ndarray:
        """
        Map VQ token IDs to Gemma token IDs while preserving time-interleaved order.
        
        Args:
            motion_tokens_1d: 1D array of VQ token IDs with time-interleaved order:
                [t0_wrist_left, t0_wrist_right, t0_hand_left, t0_hand_right,
                 t1_wrist_left, t1_wrist_right, t1_hand_left, t1_hand_right, ...]
            mapping: Dictionary mapping VQ token IDs to Gemma token IDs:
                {part_name: {vq_id: gemma_id}}
        
        Returns:
            1D array of Gemma token IDs with same time-interleaved order
        """
        wrist_tokens, hand_tokens = self._split_interleaved_tokens(
            torch.from_numpy(motion_tokens_1d).unsqueeze(0)
        )
        wrist_tokens = wrist_tokens.squeeze(0)
        hand_tokens = hand_tokens.squeeze(0)
        
        mapped_wrist_tokens = [mapping['wrist'][int(token)] for token in wrist_tokens]
        mapped_hand_tokens = [mapping['hand'][int(token)] for token in hand_tokens]
        
        return np.concatenate([mapped_wrist_tokens, mapped_hand_tokens])
