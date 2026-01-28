import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pathlib
import math


def plot_l1_loss_as_bar(data, step, output_dir): 
    """
    Plot average L1 loss for each dimension as a bar chart and save to file.

    Args:
        data: np.ndarray of shape (..., D)
        step: int, Step number
        output_dir: Output directory to save the plot
    """
    print(f"Plotting L1 loss as bar for step {step}, data shape: {data.shape}")
    mean_l1_loss = np.mean(data.reshape(-1, data.shape[-1]), axis=0)
    plt.figure(figsize=(12, 8))
    plt.bar(range(mean_l1_loss.shape[0]), mean_l1_loss)
    plt.xlabel('Dimension')
    plt.ylabel('Average L1 Loss')
    plt.title('Average L1 Loss for Each Dimension')
    save_dir = pathlib.Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'l1_loss_as_bar_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_histogram(data, key, output_dir):
    """
    Plot histogram of token lengths with statistical annotations and save to file.
    
    Args:
        data: List of token lengths
        key: Key name for the tokenizer
        output_dir: Output directory to save the plot
    """
    # Calculate statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    p95 = np.percentile(data, 95)
    p99 = np.percentile(data, 99)
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    n, bins, patches = plt.hist(data, bins=300, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add vertical lines for statistics
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2, label=f'Mean + Std: {mean_val + std_val:.2f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2, label=f'Mean - Std: {mean_val - std_val:.2f}')
    plt.axvline(p95, color='green', linestyle='--', linewidth=2, label=f'95th percentile: {p95:.2f}')
    plt.axvline(p99, color='purple', linestyle='--', linewidth=2, label=f'99th percentile: {p99:.2f}')
    
    # Add labels and title
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.title(f'Token Length Distribution for {key}\n'
              f'Mean: {mean_val:.2f}, Std: {std_val:.2f}, 95th: {p95:.2f}, 99th: {p99:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f'{key}_token_length_histogram.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / f'{key}_token_length_histogram.pdf', bbox_inches='tight')
    
    # Print statistics
    print(f"Token length statistics for {key}:")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Standard deviation: {std_val:.2f}")
    print(f"  95th percentile: {p95:.2f}")
    print(f"  99th percentile: {p99:.2f}")
    print(f"  Min: {np.min(data):.2f}")
    print(f"  Max: {np.max(data):.2f}")
    print(f"  Total samples: {len(data)}")


def plot_attention_maps(attention_maps, output_dir, step=None, layer_idx=None, token_segments=None):
    """
    Plot multi-head attention maps with each head in a separate subplot.
    
    Args:
        attention_maps: np.ndarray of shape [num_heads, seq_len, seq_len]
            Attention weights after softmax (values in [0, 1])
        output_dir: Output directory to save the plot
        step: Optional step number for filename
        layer_idx: Optional layer index for filename
        token_segments: Optional dict mapping token type names to their lengths.
            Example: {'image': 256, 'prompt': 10, 'state': 5, 'answer': 3, 'action': 20}
            The segments will be drawn in the order provided (dict insertion order).
    """
    if len(attention_maps.shape) != 3:
        raise ValueError(f"Expected attention_maps shape [num_heads, seq_len, seq_len], got {attention_maps.shape}")
    
    num_heads, seq_len, _ = attention_maps.shape
    total_subplots = num_heads + 1  # Individual heads + averaged map
    
    # Calculate optimal grid layout (prefer more columns than rows)
    # We want ncols >= nrows, and ncols * nrows >= total_subplots
    ncols = math.ceil(math.sqrt(total_subplots))  # Bias towards wider layout
    nrows = math.ceil(total_subplots / ncols)
    
    # Create figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    
    # Flatten axes array for easier indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])
    
    # Use classic attention map colormap
    # 'hot': black -> red -> yellow -> white (most classic for attention)
    # 'Reds': white -> red (clean and intuitive)
    # 'YlOrRd': yellow -> orange -> red
    # 'viridis': purple(0.0) -> blue -> cyan/green(0.5) -> yellow(1.0) (perceptually uniform)
    cmap = 'viridis'
    
    # Calculate segment boundaries if token_segments provided
    segment_boundaries = None
    segment_labels = None
    segment_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    if token_segments is not None:
        segment_boundaries = []
        segment_labels = []
        cumsum = 0
        for segment_name, segment_len in token_segments.items():
            segment_boundaries.append(cumsum)
            segment_labels.append(segment_name)
            cumsum += segment_len
        segment_boundaries.append(cumsum)  # Add final boundary
        
        # Verify total length matches seq_len
        if cumsum != seq_len:
            print(f"Warning: Total segment length ({cumsum}) doesn't match seq_len ({seq_len})")
    
    # Use PowerNorm for better visibility of different scales
    # Set threshold: values below 1/seq_len are set to exactly threshold (mapped to colormap min)
    threshold = 1.0 / seq_len
    gamma = 0.5  # Power for PowerNorm (0.5 = sqrt, smaller = more compression)
    
    # Use the colormap directly without modification
    norm = colors.PowerNorm(gamma=gamma, vmin=threshold, vmax=1.0)
    
    # Plot individual attention heads
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        head_data = attention_maps[head_idx].copy()
        # Set values below threshold to exactly threshold (will map to colormap minimum)
        head_data_clipped = np.clip(head_data, threshold, 1.0)
        
        im = ax.imshow(head_data_clipped, cmap=cmap, aspect='auto', norm=norm)
        ax.set_title(f'Head {head_idx + 1}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Key Position', fontsize=8)
        ax.set_ylabel('Query Position', fontsize=8)
        ax.tick_params(labelsize=7)
        
        # Add segment boundaries and labels
        if segment_boundaries is not None:
            _add_segment_annotations(ax, segment_boundaries, segment_labels, segment_colors, seq_len)
        
        # Add colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(f'Attention (≥{threshold:.4f})', fontsize=7)
    
    # Plot averaged attention map
    avg_attention = np.mean(attention_maps, axis=0)
    ax = axes[num_heads]
    avg_data = avg_attention.copy()
    # Set values below threshold to exactly threshold (will map to colormap minimum)
    avg_data_clipped = np.clip(avg_data, threshold, 1.0)
    
    im = ax.imshow(avg_data_clipped, cmap=cmap, aspect='auto', norm=norm)
    ax.set_title('Average over Heads', fontsize=10, fontweight='bold', color='red')
    ax.set_xlabel('Key Position', fontsize=8)
    ax.set_ylabel('Query Position', fontsize=8)
    ax.tick_params(labelsize=7)
    
    # Add segment boundaries and labels
    if segment_boundaries is not None:
        _add_segment_annotations(ax, segment_boundaries, segment_labels, segment_colors, seq_len)
    
    # Add colorbar for averaged map
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(f'Attention (≥{threshold:.4f})', fontsize=7)
    
    # Hide unused subplots
    for idx in range(total_subplots, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    title_parts = ['Multi-Head Attention Maps']
    if layer_idx is not None:
        title_parts.append(f'Layer {layer_idx}')
    if step is not None:
        title_parts.append(f'Step {step}')
    fig.suptitle(' - '.join(title_parts), fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save the plot
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build filename
    filename_parts = ['attention_maps']
    if layer_idx is not None:
        filename_parts.append(f'layer{layer_idx}')
    if step is not None:
        filename_parts.append(f'step{step}')
    filename = '_'.join(filename_parts)
    
    plt.savefig(output_path / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / f'{filename}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Attention maps saved to {output_path / filename}.png")
    print(f"  Shape: {attention_maps.shape}")
    print(f"  Num heads: {num_heads}, Sequence length: {seq_len}")
    print(f"  Value range: [{attention_maps.min():.4f}, {attention_maps.max():.4f}]")
    print(f"  Average attention entropy: {-np.sum(attention_maps * np.log(attention_maps + 1e-10), axis=-1).mean():.4f}")


def _add_segment_annotations(ax, boundaries, labels, colors, seq_len):
    """
    Add segment boundary lines and labels to an attention map plot.
    
    Args:
        ax: Matplotlib axis object
        boundaries: List of boundary positions (including 0 and seq_len)
        labels: List of segment labels
        colors: List of colors for each segment
        seq_len: Total sequence length
    """
    # Draw vertical and horizontal lines at boundaries
    for i, boundary in enumerate(boundaries[1:-1], start=1):  # Skip first (0) and last (seq_len)
        ax.axvline(x=boundary - 0.5, color='white', linewidth=1.5, linestyle='--', alpha=0.8)
        ax.axhline(y=boundary - 0.5, color='white', linewidth=1.5, linestyle='--', alpha=0.8)
    
    # Add text labels for each segment
    for i, label in enumerate(labels):
        start_pos = boundaries[i]
        end_pos = boundaries[i + 1]
        mid_pos = (start_pos + end_pos) / 2
        color = colors[i % len(colors)]
        
        # Add label on top (for key/column dimension)
        ax.text(
            mid_pos, -0.02 * seq_len, label,
            ha='center', va='top', fontsize=7, fontweight='bold',
            color=color, rotation=0,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8)
        )
        
        # Add label on left (for query/row dimension)
        ax.text(
            -0.02 * seq_len, mid_pos, label,
            ha='right', va='center', fontsize=7, fontweight='bold',
            color=color, rotation=0,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8)
        )


def plot_multilayer_attention_maps(attention_maps, output_dir, step=None, token_segments=None, layer_plot_every=4):
    """
    Plot multi-layer multi-head attention maps.
    Each layer is plotted separately, plus an averaged map across all layers.
    
    Args:
        attention_maps: np.ndarray of shape [num_layers, num_heads, seq_len, seq_len]
            Attention weights after softmax (values in [0, 1])
        output_dir: Output directory to save the plots
        step: Optional step number for filename
        token_segments: Optional dict mapping token type names to their lengths.
            Example: {'image': 256, 'prompt': 10, 'state': 5, 'answer': 3, 'action': 20}
    """
    if len(attention_maps.shape) != 4:
        raise ValueError(
            f"Expected attention_maps shape [num_layers, num_heads, seq_len, seq_len], got {attention_maps.shape}"
        )

    num_layers, num_heads, seq_len, _ = attention_maps.shape

    print(f"\nPlotting multi-layer attention maps:")
    print(f"  Total shape: {attention_maps.shape}")
    print(f"  Num layers: {num_layers}, Num heads: {num_heads}, Sequence length: {seq_len}")
    print(f"  Value range: [{attention_maps.min():.4f}, {attention_maps.max():.4f}]")
    print()

    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot each layer separately
    for layer_idx in range(0, num_layers, layer_plot_every):
        print(f"Plotting layer {layer_idx + 1}/{num_layers}...")
        layer_attention = attention_maps[layer_idx]  # [num_heads, seq_len, seq_len]
        plot_attention_maps(
            attention_maps=layer_attention,
            output_dir=output_dir,
            step=step,
            layer_idx=layer_idx,
            token_segments=token_segments
        )

    # Plot averaged attention map across all layers
    print(f"\nPlotting average across all {num_layers} layers...")
    print(f"\nPlotting average across all {math.ceil(num_layers / layer_plot_every)} layers...")
    avg_attention = np.mean(attention_maps, axis=0)  # [num_heads, seq_len, seq_len]
    plot_attention_maps(
        attention_maps=avg_attention,
        output_dir=output_dir,
        step=step,
        layer_idx="avg",
        token_segments=token_segments
    )

    print(f"\nAll attention maps saved to {output_path}")
    print(f"  Total files: {num_layers + 1} ({num_layers} layers + 1 averaged)")


def test_attention_visualization(output_dir, seq_len=100, num_heads=8, num_layers=4):
    """
    Test function to visualize typical attention patterns.
    Creates various characteristic attention patterns for testing the visualization.
    
    Args:
        output_dir: Output directory to save the test plots
        seq_len: Sequence length for attention maps
        num_heads: Number of attention heads
        num_layers: Number of layers (for multilayer test)
    """
    print(f"\n{'='*60}")
    print(f"Testing Attention Visualization")
    print(f"{'='*60}")
    print(f"Parameters:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Define token segments for testing
    token_segments = {
        'image': seq_len // 2,
        'prompt': seq_len // 6,
        'state': seq_len // 10,
        'action': seq_len - seq_len // 2 - seq_len // 6 - seq_len // 10
    }
    
    print(f"Token segments: {token_segments}")
    print()
    
    # Create typical attention patterns
    attention_patterns = []
    pattern_names = []
    
    # Pattern 1: Uniform attention (all positions equal weight)
    uniform = np.ones((seq_len, seq_len)) / seq_len
    attention_patterns.append(uniform)
    pattern_names.append("Uniform")
    
    # Pattern 2: Diagonal attention (self-attention)
    diagonal = np.eye(seq_len, dtype=np.float64)  # Use float64 for precision
    # Each row sums to 1 already, no need to normalize
    # Verify it's exactly 0 and 1
    assert np.all((diagonal == 0) | (diagonal == 1)), "Diagonal should only contain 0 and 1"
    assert diagonal.max() == 1.0, f"Diagonal max should be 1.0, got {diagonal.max()}"
    assert diagonal.min() == 0.0, f"Diagonal min should be 0.0, got {diagonal.min()}"
    attention_patterns.append(diagonal)
    pattern_names.append("Diagonal")
    
    # Pattern 3: Local attention (attend to nearby tokens)
    local = np.zeros((seq_len, seq_len))
    window_size = 10
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        local[i, start:end] = 1.0
    local = local / local.sum(axis=-1, keepdims=True)
    attention_patterns.append(local)
    pattern_names.append("Local")
    
    # Pattern 4: Global attention (first token attends to all, others attend to first)
    global_attn = np.zeros((seq_len, seq_len))
    global_attn[0, :] = 1.0 / seq_len  # First token attends to all
    global_attn[1:, 0] = 0.7  # Other tokens attend strongly to first token
    global_attn[1:, 1:] = 0.3 / (seq_len - 1)  # Weak attention to others
    global_attn = global_attn / global_attn.sum(axis=-1, keepdims=True)
    attention_patterns.append(global_attn)
    pattern_names.append("Global")
    
    # Pattern 5: Block attention (attend within segments)
    block = np.zeros((seq_len, seq_len))
    cumsum = 0
    for segment_name, segment_len in token_segments.items():
        block[cumsum:cumsum+segment_len, cumsum:cumsum+segment_len] = 1.0 / segment_len
        cumsum += segment_len
    attention_patterns.append(block)
    pattern_names.append("Block")
    
    # Pattern 6: Cross-segment attention (image tokens attend to action tokens)
    cross = np.zeros((seq_len, seq_len))
    image_len = token_segments['image']
    action_start = sum(list(token_segments.values())[:3])  # After image, prompt, state
    action_end = seq_len
    # Image tokens attend to action tokens
    cross[:image_len, action_start:action_end] = 1.0 / (action_end - action_start)
    # Other tokens uniform
    cross[image_len:, :] = 1.0 / seq_len
    cross = cross / cross.sum(axis=-1, keepdims=True)
    attention_patterns.append(cross)
    pattern_names.append("Cross-Segment")
    
    # Pattern 7: Sparse attention (few strong connections)
    sparse = np.ones((seq_len, seq_len)) * 0.01 / seq_len
    for i in range(0, seq_len, seq_len // 10):
        sparse[i, :] = 1.0 / seq_len
        sparse[:, i] += 0.5 / seq_len
    sparse = sparse / sparse.sum(axis=-1, keepdims=True)
    attention_patterns.append(sparse)
    pattern_names.append("Sparse")
    
    # Pattern 8: Causal attention (only attend to previous tokens)
    causal = np.tril(np.ones((seq_len, seq_len)))
    causal = causal / causal.sum(axis=-1, keepdims=True)
    attention_patterns.append(causal)
    pattern_names.append("Causal")
    
    # Ensure we have exactly num_heads patterns (repeat or truncate)
    while len(attention_patterns) < num_heads:
        attention_patterns.extend(attention_patterns[:num_heads - len(attention_patterns)])
        pattern_names.extend(pattern_names[:num_heads - len(pattern_names)])
    attention_patterns = attention_patterns[:num_heads]
    pattern_names = pattern_names[:num_heads]
    
    # Stack into [num_heads, seq_len, seq_len]
    single_layer_attention = np.stack(attention_patterns, axis=0)
    
    print("="*60)
    print("Test 1: Single Layer Attention Maps")
    print("="*60)
    print(f"Patterns: {', '.join(pattern_names)}")
    print(f"\nAttention data range: [{single_layer_attention.min():.4f}, {single_layer_attention.max():.4f}]")
    for i, name in enumerate(pattern_names):
        pattern_min = single_layer_attention[i].min()
        pattern_max = single_layer_attention[i].max()
        print(f"  {name}: min={pattern_min:.4f}, max={pattern_max:.4f}")
    print()
    
    plot_attention_maps(
        attention_maps=single_layer_attention,
        output_dir=output_dir,
        step=0,
        layer_idx=0,
        token_segments=token_segments
    )
    
    print()
    print("="*60)
    print("Test 2: Multi-Layer Attention Maps")
    print("="*60)
    print(f"Creating {num_layers} layers with varying patterns...")
    print()
    
    # Create multi-layer attention with slight variations
    multilayer_attention = []
    for layer_idx in range(num_layers):
        # For multi-layer test, keep patterns clean
        layer_patterns = []
        for i, pattern in enumerate(attention_patterns):
            if pattern_names[i] == "Diagonal":
                # Keep diagonal pattern clean (no noise)
                layer_patterns.append(pattern.copy())
            else:
                # Add small noise to other patterns for variation
                noisy_pattern = pattern + np.random.normal(0, 0.01, pattern.shape)
                noisy_pattern = np.clip(noisy_pattern, 0, 1)
                noisy_pattern = noisy_pattern / noisy_pattern.sum(axis=-1, keepdims=True)
                layer_patterns.append(noisy_pattern)
        multilayer_attention.append(np.stack(layer_patterns, axis=0))
    
    multilayer_attention = np.stack(multilayer_attention, axis=0)  # [num_layers, num_heads, seq_len, seq_len]
    
    plot_multilayer_attention_maps(
        attention_maps=multilayer_attention,
        output_dir=output_dir,
        step=0,
        token_segments=token_segments
    )
    
    print()
    print("="*60)
    print("Test Complete!")
    print("="*60)
    print(f"All test visualizations saved to: {output_dir}")
    print(f"Generated files:")
    print(f"  - Single layer: attention_maps_layer0_step0.png/pdf")
    print(f"  - Multi-layer: attention_maps_layer{{0..{num_layers-1}}}_step0.png/pdf")
    print(f"  - Multi-layer average: attention_maps_layeravg_step0.png/pdf")
    print("="*60)


def main():
    """Main function for testing attention visualization from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test attention map visualization with various typical patterns"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/test_attention_visualization",
        help="Output directory for test visualizations (default: outputs/test_attention_visualization)"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=100,
        help="Sequence length for attention maps (default: 100)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of layers for multi-layer test (default: 4)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Attention Map Visualization Test")
    print("="*70 + "\n")
    
    test_attention_visualization(
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print(f"Check the output directory: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
