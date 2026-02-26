"""Tests for joint model attention functions."""
import torch
from src.model.moe.joint_model import (
    forward_mixture_scaled_dot_product_attention,
    forward_insulation_scaled_dot_product_attention,
)


def test_attention_functions():
    """
    Test function to verify that forward_mixture_scaled_dot_product_attention and 
    forward_insulation_scaled_dot_product_attention produce the same output and 
    that forward_insulation prevents gradient flow from non-vlm parts to vlm parts.
    """
    print("Testing attention functions...")
    
    # Set up test parameters
    batch_size = 2
    num_heads_q = 8
    num_heads_kv = 8
    vlm_seq_len = 4
    other_seq_len = 3
    head_dim = 64
    attn_softclamp = 50.0
    attention_dropout = 0.0
    
    # Create test data
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Create query, key, value states for different mixtures
    query_states_all = {
        "vlm": torch.randn(batch_size, num_heads_q, vlm_seq_len, head_dim, device=device, dtype=dtype, requires_grad=True),
        "action": torch.randn(batch_size, num_heads_q, other_seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    }
    
    key_states_all = {
        "vlm": torch.randn(batch_size, num_heads_kv, vlm_seq_len, head_dim, device=device, dtype=dtype, requires_grad=True),
        "action": torch.randn(batch_size, num_heads_kv, other_seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    }
    
    value_states_all = {
        "vlm": torch.randn(batch_size, num_heads_kv, vlm_seq_len, head_dim, device=device, dtype=dtype, requires_grad=True),
        "action": torch.randn(batch_size, num_heads_kv, other_seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    }
    
    # Create attention mask that reflects the insulation constraint
    # VLM can only attend to VLM, others can attend to both VLM and others
    full_seq_len = vlm_seq_len + other_seq_len
    attention_mask = torch.zeros(batch_size, num_heads_q, full_seq_len, full_seq_len, device=device, dtype=dtype)
    
    # Set mask to -inf for positions that should not be attended to
    # VLM (first vlm_seq_len positions) cannot attend to others (positions vlm_seq_len:)
    attention_mask[:, :, :vlm_seq_len, vlm_seq_len:] = torch.finfo(dtype).min
    print(attention_mask[0][0])
    
    # Others can attend to everything (no additional masking needed)
    
    # Test 1: Compare outputs
    print("Test 1: Comparing outputs...")
    
    # Forward pass with mixture attention
    output_mixture = forward_mixture_scaled_dot_product_attention(
        query_states_all, key_states_all, value_states_all, attention_mask,
        attn_softclamp, attention_dropout, training=False
    )
    
    # Forward pass with insulation attention
    output_insulation = forward_insulation_scaled_dot_product_attention(
        query_states_all, key_states_all, value_states_all, attention_mask,
        attn_softclamp, attention_dropout, training=False
    )
    
    # Check if outputs are the same
    output_diff = torch.abs(output_mixture - output_insulation).max().item()
    print(f"Maximum difference between outputs: {output_diff}")
    if output_diff < 1e-5:
        print("✓ Outputs are identical!")
    else:
        print("✗ Outputs differ!")
    
    # Test 2: Check gradient flow
    print("\nTest 2: Checking gradient flow...")
    
    # Create a simple loss function
    loss_mixture = output_mixture.sum()
    loss_insulation = output_insulation.sum()
    
    # Backward pass for mixture attention
    loss_mixture.backward(retain_graph=True)
    vlm_grad_mixture = query_states_all["vlm"].grad.clone()
    action_grad_mixture = query_states_all["action"].grad.clone()
    
    # Clear gradients
    for tensor in query_states_all.values():
        tensor.grad = None
    for tensor in key_states_all.values():
        tensor.grad = None
    for tensor in value_states_all.values():
        tensor.grad = None
    
    # Backward pass for insulation attention
    loss_insulation.backward(retain_graph=True)
    vlm_grad_insulation = query_states_all["vlm"].grad.clone()
    action_grad_insulation = query_states_all["action"].grad.clone()
    
    # Check if vlm gradients are the same
    vlm_grad_diff = torch.abs(vlm_grad_mixture - vlm_grad_insulation).max().item()
    print(f"Maximum difference in VLM gradients: {vlm_grad_diff}")
    if vlm_grad_diff < 1e-5:
        print("✓ VLM gradients are identical!")
    else:
        print("✗ VLM gradients differ!")
    
    # Check if action gradients are the same
    action_grad_diff = torch.abs(action_grad_mixture - action_grad_insulation).max().item()
    print(f"Maximum difference in action gradients: {action_grad_diff}")
    if action_grad_diff < 1e-5:
        print("✓ Action gradients are identical!")
    else:
        print("✗ Action gradients differ!")
    
    # Test 3: Check gradient isolation
    print("\nTest 3: Checking gradient isolation...")
    
    # Clear all gradients
    for tensor in query_states_all.values():
        tensor.grad = None
    for tensor in key_states_all.values():
        tensor.grad = None
    for tensor in value_states_all.values():
        tensor.grad = None
    
    # Test gradient flow from action to vlm in insulation mode
    # We'll create a loss that only depends on action part of the output
    action_output_insulation = output_insulation[:, :, vlm_seq_len:, :]  # Only action part
    loss_action_only = action_output_insulation.sum()
    loss_action_only.backward()
    
    # Check if vlm parameters received gradients
    vlm_has_grad = query_states_all["vlm"].grad is not None and query_states_all["vlm"].grad.abs().sum() > 0
    action_has_grad = query_states_all["action"].grad is not None and query_states_all["action"].grad.abs().sum() > 0
    
    print(f"VLM received gradients: {vlm_has_grad}")
    print(f"Action received gradients: {action_has_grad}")
    
    if not vlm_has_grad and action_has_grad:
        print("✓ Gradient isolation working correctly - VLM isolated from action gradients!")
    else:
        print("✗ Gradient isolation not working as expected!")
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_attention_functions()
