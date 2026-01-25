"""
Sampling utility functions for token sampling during text generation.
"""

from typing import Optional, Tuple
import torch


def sample_token(
    logits: torch.FloatTensor,
    temperature: float = 1.0,
    top_k: int = 10,
    top_p: float = 1.0,
    allowed_token_ids: Optional[torch.LongTensor] = None,
) -> torch.LongTensor:
    """
    Sample the next token from logits.
    
    Args:
        logits (torch.FloatTensor): Logits of shape [B, vocab_size] or [B, seq_len, vocab_size].
            If [B, seq_len, vocab_size], only the last position's logits are used.
        temperature (float): Temperature parameter controlling sampling randomness.
            Higher values make sampling more random, lower values make it more deterministic.
        top_k (int): Top-k sampling, only sample from the k tokens with highest probabilities.
        top_p (float): Nucleus sampling, only sample from tokens whose cumulative probability reaches top_p.
        allowed_token_ids (Optional[torch.LongTensor]): Allowed token ID range for sampling.
            If provided, can be:
            - A boolean mask tensor of shape [vocab_size]
            - An integer range [min_id, max_id]
            - A list or set of integer token IDs
    
    Returns:
        torch.LongTensor: Sampled token IDs of shape [B]
    """
    # Handle logits shape
    if logits.dim() == 3:
        # [B, seq_len, vocab_size] -> [B, vocab_size]
        logits = logits[:, -1, :]
    elif logits.dim() != 2:
        raise ValueError(f"logits must be 2D or 3D tensor, got shape: {logits.shape}")
    
    batch_size, vocab_size = logits.shape
    
    # Apply token ID range restriction
    if allowed_token_ids is not None:
        if isinstance(allowed_token_ids, tuple) and len(allowed_token_ids) == 2:
            # Assume (min_id, max_id) range
            min_id, max_id = allowed_token_ids[0], allowed_token_ids[1]
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
            mask[min_id:max_id+1] = True
            allowed_token_ids = mask
        elif isinstance(allowed_token_ids, (list, set)):
            # Convert to boolean mask
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
            mask[list(allowed_token_ids)] = True
            allowed_token_ids = mask
        elif isinstance(allowed_token_ids, torch.Tensor):
            if allowed_token_ids.dtype == torch.bool:
                # Already a boolean mask
                if allowed_token_ids.shape[0] != vocab_size:
                    raise ValueError(
                        f"allowed_token_ids boolean mask length ({allowed_token_ids.shape[0]}) "
                        f"must equal vocab_size ({vocab_size})"
                    )
            elif allowed_token_ids.dtype in (torch.int64, torch.int32, torch.int16):
                # Integer range or list
                if allowed_token_ids.numel() == 2 and allowed_token_ids.dim() == 1:
                    # Assume [min_id, max_id] range
                    min_id, max_id = allowed_token_ids[0].item(), allowed_token_ids[1].item()
                    mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
                    mask[min_id:max_id+1] = True
                    allowed_token_ids = mask
                else:
                    # Assume token ID list
                    mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
                    mask[allowed_token_ids] = True
                    allowed_token_ids = mask
            else:
                raise ValueError(f"Unsupported allowed_token_ids type: {allowed_token_ids.dtype}")
        else: 
            raise ValueError(f"Unsupported allowed_token_ids type: {type(allowed_token_ids)}")
        
        # Set logits of disallowed tokens to negative infinity
        logits = logits.masked_fill(~allowed_token_ids, float('-inf'))
    
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        # Use greedy sampling when temperature = 0
        return logits.argmax(dim=-1)
    
    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Top-k sampling
    if top_k > 0 and top_k < vocab_size:
        # Get top-k indices and values
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        # Create new probability distribution, keeping only top-k
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(1, top_k_indices, top_k_probs)
        # Renormalize
        probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    # Top-p (nucleus) sampling
    if top_p < 1.0:
        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        # Compute cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Find positions where cumulative probability exceeds top_p
        sorted_indices_to_remove = cumsum_probs > top_p
        # Keep the first position that exceeds top_p (keep at least one token)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Create mask marking tokens to remove
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        # Set probabilities of removed tokens to 0
        probs = probs.masked_fill(indices_to_remove, 0.0)
        # Renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Sample from probability distribution
    sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return sampled_indices

