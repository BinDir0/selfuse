import argparse
import math
import pathlib

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
try:
    import cv2
except Exception:
    cv2 = None


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


def plot_attention_maps(
    attention_maps,
    output_dir,
    step=None,
    layer_idx=None,
    token_segments=None,
    filename_prefix="attention_maps",
):
    """
    Plot multi-head attention maps with each head in a separate subplot.
    
    Args:
        attention_maps: np.ndarray of shape [num_heads, seq_len, seq_len]
            Attention weights after softmax (values in [0, 1])
        output_dir: Output directory to save the plot
        step: Optional step number for filename
        layer_idx: Optional layer index for filename
        token_segments: Optional dict mapping token type names to their lengths.
            Example: {'image': 256, 'text': 128, 'state': 5, 'action': 20, 'action_expert': 32, 'padding': 64}
            The segments will be drawn in the order provided (dict insertion order).
    """
    if len(attention_maps.shape) != 3:
        raise ValueError(f"Expected attention_maps shape [num_heads, seq_len, seq_len], got {attention_maps.shape}")
    
    num_heads, q_len, k_len = attention_maps.shape
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
    segment_colors = None
    draw_q = True
    draw_k = True
    color_map = {
        "image": "red",
        "text": "blue",
        "state": "green",
        "action": "orange",
        "action_expert": "purple",
        "padding": "gray",
    }
    default_colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]

    if token_segments is not None:
        segment_boundaries = []
        segment_labels = []
        segment_colors = []
        cumsum = 0
        if isinstance(token_segments, dict) and "segments" in token_segments:
            draw_q = bool(token_segments.get("draw_q", True))
            draw_k = bool(token_segments.get("draw_k", True))
            items = token_segments["segments"]
        else:
            items = token_segments
        for segment_name, segment_len in items:
            segment_boundaries.append(cumsum)
            segment_labels.append(segment_name)
            segment_colors.append(color_map.get(segment_name, default_colors[len(segment_colors) % len(default_colors)]))
            cumsum += int(segment_len)
        segment_boundaries.append(cumsum)  # Add final boundary

        # Verify total length matches seq_len
        expected_len = k_len if (draw_k and not draw_q) else q_len
        if draw_q and draw_k:
            expected_len = q_len
        if cumsum != expected_len:
            print(f"Warning: Total segment length ({cumsum}) doesn't match expected length ({expected_len})")
    
    # Use PowerNorm for better visibility of different scales
    # Set threshold: values below 1/seq_len are set to exactly threshold (mapped to colormap min)
    threshold = 1.0 / max(q_len, k_len)
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
            _add_segment_annotations(
                ax,
                segment_boundaries,
                segment_labels,
                segment_colors,
                q_len=q_len,
                k_len=k_len,
                draw_q=draw_q,
                draw_k=draw_k,
            )
        
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
        _add_segment_annotations(
            ax,
            segment_boundaries,
            segment_labels,
            segment_colors,
            q_len=q_len,
            k_len=k_len,
            draw_q=draw_q,
            draw_k=draw_k,
        )
    
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
    filename_parts = [filename_prefix]
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
    print(f"  Num heads: {num_heads}, Query length: {q_len}, Key length: {k_len}")
    print(f"  Value range: [{attention_maps.min():.4f}, {attention_maps.max():.4f}]")
    print(f"  Average attention entropy: {-np.sum(attention_maps * np.log(attention_maps + 1e-10), axis=-1).mean():.4f}")


def _add_segment_annotations(ax, boundaries, labels, colors, q_len, k_len, draw_q=True, draw_k=True):
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
        if draw_k:
            ax.axvline(x=boundary - 0.5, color='white', linewidth=0.3, linestyle='--', alpha=0.5)
        if draw_q:
            ax.axhline(y=boundary - 0.5, color='white', linewidth=0.3, linestyle='--', alpha=0.5)
    
    # Add text labels for each segment
    visible_idx = 0
    for i, label in enumerate(labels):
        start_pos = boundaries[i]
        end_pos = boundaries[i + 1]
        segment_len = end_pos - start_pos
        if label == "padding":
            continue
        if label == "text" and segment_len < 5:
            continue
        mid_pos = (start_pos + end_pos) / 2
        color = colors[i % len(colors)]

        # Add label on top (for key/column dimension), stagger to reduce overlap
        if draw_k:
            stagger = (visible_idx % 3) * 0.035 * q_len
            top_y = -0.015 * q_len - stagger
            ax.text(
                mid_pos, top_y, label,
                ha='center', va='top', fontsize=3, fontweight='bold',
                color=color, rotation=0,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor=color, alpha=0.8)
            )

        # Add label on left (for query/row dimension), stagger to reduce overlap
        if draw_q:
            stagger = (visible_idx % 3) * 0.035 * k_len
            left_x = -0.015 * k_len - stagger
            ax.text(
                left_x, mid_pos, label,
                ha='right', va='center', fontsize=3, fontweight='bold',
                color=color, rotation=0,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor=color, alpha=0.8)
            )
        visible_idx += 1


def plot_multilayer_attention_maps(
    attention_maps,
    output_dir,
    step=None,
    token_segments=None,
    layer_plot_every=4,
    filename_prefix="attention_maps",
):
    """
    Plot multi-layer multi-head attention maps.
    Each layer is plotted separately, plus an averaged map across all layers.
    
    Args:
        attention_maps: np.ndarray of shape [num_layers, num_heads, seq_len, seq_len]
            Attention weights after softmax (values in [0, 1])
        output_dir: Output directory to save the plots
        step: Optional step number for filename
        token_segments: Optional dict mapping token type names to their lengths.
            Example: {'image': 256, 'text': 128, 'state': 5, 'action': 20, 'action_expert': 32, 'padding': 64}
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
            token_segments=token_segments,
            filename_prefix=filename_prefix,
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
        token_segments=token_segments,
        filename_prefix=filename_prefix,
    )

    print(f"\nAll attention maps saved to {output_path}")
    print(f"  Total files: {num_layers + 1} ({num_layers} layers + 1 averaged)")


def _load_config(config_path):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load(config_path)
    return cfg


def _filter_padding_tokens(attention_maps, labels):
    labels_len = len(labels)
    q_len, k_len = attention_maps.shape[-2:]
    match_q = labels_len == q_len
    match_k = labels_len == k_len
    assert match_q or match_k, "labels_len must match q_len or k_len"

    keep_indices = np.array(
        [idx for idx, label in enumerate(labels) if label != "padding"]
    )
    assert len(keep_indices) > 0, "No tokens left after filtering padding"
    filtered = attention_maps
    if match_q:
        filtered = filtered[..., keep_indices, :]
    if match_k:
        filtered = filtered[..., :, keep_indices]

    filtered_labels = [labels[idx] for idx in keep_indices]
    filtered_segments = _compress_segments(filtered_labels)
    return filtered, filtered_segments, match_q, match_k

def _compress_segments(labels):
    if not labels:
        return []
    segments = []
    current = labels[0]
    length = 1
    for label in labels[1:]:
        if label == current:
            length += 1
        else:
            segments.append((current, length))
            current = label
            length = 1
    segments.append((current, length))
    return segments

def _build_token_segments(
    input_ids,
    total_len,
    image_token_id,
    state_token_id,
    action_token_id,
    pad_token_id,
    other_name='answer',
):
    input_ids = np.array(input_ids).astype(np.int64)
    vlm_len = input_ids.shape[0]
    labels = []
    for _, token in enumerate(input_ids):
        if image_token_id is not None and token == image_token_id:
            labels.append("image")
        elif state_token_id is not None and token == state_token_id:
            labels.append("state")
        elif action_token_id is not None and token == action_token_id:
            labels.append("action")
        elif pad_token_id is not None and token == pad_token_id:
            labels.append("padding")
        else:
            labels.append("text")

    for _ in range(vlm_len, total_len):
        labels.append(other_name)
    return _compress_segments(labels)


def _load_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _plot_causal_mask(causal_mask, output_dir, step=None, token_segments=None):
    if causal_mask.ndim == 3:
        causal_mask = causal_mask[0]
    # Causal mask convention: 0 means allowed, -inf means blocked.
    # Use absolute magnitude to detect "blocked" entries robustly.
    mask = (np.abs(causal_mask) < 1e4).astype(np.float32)
    plot_attention_maps(
        attention_maps=mask[None, ...],
        output_dir=output_dir,
        step=step,
        layer_idx="causal",
        token_segments=token_segments,
        filename_prefix="causal_mask",
    )


def _prepare_rgb_image(pixel_values):
    img = np.array(pixel_values)
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32)
    if img.min() < 0.0:
        img = img * 0.5 + 0.5
    img = np.clip(img, 0.0, 1.0)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img


def _resize_heatmap(heatmap, target_h, target_w):
    if cv2 is not None:
        return cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if heatmap.shape[0] == 0 or heatmap.shape[1] == 0:
        return heatmap
    scale_h = max(1, target_h // heatmap.shape[0])
    scale_w = max(1, target_w // heatmap.shape[1])
    resized = np.repeat(np.repeat(heatmap, scale_h, axis=0), scale_w, axis=1)
    return resized[:target_h, :target_w]


def _attention_to_image(attn_avg, query_indices, image_key_indices):
    if query_indices is None:
        query_indices = np.arange(attn_avg.shape[0])
    if len(query_indices) == 0 or len(image_key_indices) == 0:
        return None
    attn_slice = attn_avg[np.ix_(query_indices, image_key_indices)]
    return attn_slice.sum(axis=0)


def _image_grid_shape(num_image_tokens, pixel_values, patch_size=None):
    if pixel_values is not None:
        img = np.array(pixel_values)
        if img.ndim == 4:
            _, _, h, w = img.shape
        elif img.ndim == 3:
            if img.shape[0] in (1, 3):
                h, w = img.shape[1], img.shape[2]
            else:
                h, w = img.shape[0], img.shape[1]
        else:
            h, w = None, None
        if h is not None and w is not None and patch_size:
            gh, gw = h // patch_size, w // patch_size
            if gh * gw == num_image_tokens:
                return gh, gw
    return None, None


def _overlay_heatmap(image_rgb, heatmap, output_path, cmap="jet", alpha=0.5, gamma=0.7):
    heatmap = heatmap.astype(np.float32)
    if heatmap.size == 0:
        return
    hmin, hmax = float(heatmap.min()), float(heatmap.max())
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    if gamma is not None:
        heatmap = heatmap ** gamma
    heatmap = _resize_heatmap(heatmap, image_rgb.shape[0], image_rgb.shape[1])
    # Overlay heatmap on top of the original image.
    fig_w = 6.0
    fig_h = fig_w * (image_rgb.shape[0] / max(1, image_rgb.shape[1]))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(np.clip(image_rgb, 0.0, 1.0))
    ax.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
    ax.axis("off")
    plt.tight_layout(pad=0.0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def visualize_image_attention(
    attn_weights,
    action_expert_attn_weights,
    input_ids,
    attention_mask,
    pixel_values,
    image_token_id,
    sep_token_id,
    n_actions,
    output_dir,
    step=None,
    patch_size=None,
    exclude_image_queries=True,
):
    input_ids = np.array(input_ids).astype(np.int64)
    attention_mask = np.array(attention_mask).astype(np.int64) if attention_mask is not None else None
    attn_avg = attn_weights.mean(axis=(0, 1)) # average attention weights over heads and layers
    image_key_indices = np.where(input_ids == image_token_id)[0]
    if image_key_indices.size == 0:
        return
    num_frames = pixel_values.shape[0] if np.array(pixel_values).ndim == 4 else 1
    tokens_per_image = image_key_indices.size
    if num_frames > 1 and image_key_indices.size % num_frames == 0:
        tokens_per_image = image_key_indices.size // num_frames
        image_key_indices = image_key_indices[:tokens_per_image]

    grid_h, grid_w = _image_grid_shape(tokens_per_image, pixel_values, patch_size=patch_size)
    if grid_h is None or grid_w is None:
        return

    vlm_len = input_ids.shape[0]
    valid_vlm_queries = np.arange(vlm_len)
    if attention_mask is not None:
        valid_vlm_queries = valid_vlm_queries[attention_mask != 0]
    if exclude_image_queries and image_token_id is not None:
        # Avoid image self-attention dominating the image-key heatmap.
        non_image = input_ids[valid_vlm_queries] != image_token_id
        if np.any(non_image):
            valid_vlm_queries = valid_vlm_queries[non_image]
    assert valid_vlm_queries.size > 0, "No valid VLM queries found"
    action_query_indices = np.array([], dtype=np.int64)
    if sep_token_id is not None:
        sep_positions = np.where(input_ids == sep_token_id)[0]
        if sep_positions.size > 0:
            sep_pos = int(sep_positions[-1])
            action_query_indices = np.arange(sep_pos + 1, vlm_len)
        else:
            print("Warning: sep_token_id not found in input_ids; skipping action query indices.")

    action_expert_query_indices = np.array([], dtype=np.int64)
    if n_actions is not None and n_actions > 0:
        start = vlm_len
        end = min(vlm_len + int(n_actions), attn_avg.shape[0])
        action_expert_query_indices = np.arange(start, end)

    image_rgb = _prepare_rgb_image(pixel_values)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_heat = _attention_to_image(attn_avg, valid_vlm_queries, image_key_indices)
    if total_heat is not None:
        heatmap = total_heat.reshape(grid_h, grid_w)
        print(heatmap.tolist())
        suffix = f"_step{step}" if step is not None else ""
        _overlay_heatmap(image_rgb, heatmap, output_dir / f"image_attention_all{suffix}.png")

    if action_query_indices.size > 0:
        action_heat = _attention_to_image(attn_avg, action_query_indices, image_key_indices)
        if action_heat is not None:
            heatmap = action_heat.reshape(grid_h, grid_w)
            suffix = f"_step{step}" if step is not None else ""
            _overlay_heatmap(image_rgb, heatmap, output_dir / f"image_attention_action{suffix}.png")

    if action_expert_query_indices.size > 0:
        expert_heat = _attention_to_image(attn_avg, action_expert_query_indices, image_key_indices)
        if expert_heat is not None:
            heatmap = expert_heat.reshape(grid_h, grid_w)
            suffix = f"_step{step}" if step is not None else ""
            _overlay_heatmap(image_rgb, heatmap, output_dir / f"image_attention_action_expert{suffix}.png")
    elif action_expert_attn_weights is not None:
        expert_attn_avg = action_expert_attn_weights.mean(axis=(0, 1))
        expert_query_len = expert_attn_avg.shape[0]
        if n_actions is not None and int(n_actions) > 0:
            expert_query_len = min(expert_query_len, int(n_actions))
        expert_query_indices = np.arange(expert_query_len)
        expert_heat = _attention_to_image(expert_attn_avg, expert_query_indices, image_key_indices)
        if expert_heat is not None:
            heatmap = expert_heat.reshape(grid_h, grid_w)
            suffix = f"_step{step}" if step is not None else ""
            _overlay_heatmap(image_rgb, heatmap, output_dir / f"image_attention_action_expert{suffix}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attention maps and causal masks from saved npz"
    )
    parser.add_argument("--npz", type=str, required=True, help="Path to saved npz file")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment yaml")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--layer_plot_every", type=int, default=4, help="Plot every N layers")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    processor = instantiate(cfg.vla_processor)
    image_token_id = processor.image_token_id
    state_token_id = processor.state_token_id
    action_token_id = processor.action_token_id
    sep_token_id = getattr(processor, "sep_token_id", None)
    pad_token_id = cfg.pad_token_id

    data = _load_npz(args.npz)
    for key, value in data.items():
        print(f"{key}: {value.shape}")
    attn_weights = data.get("attn_weights")
    action_expert_attn_weights = data.get("action_expert_attn_weights")
    input_ids = data["input_ids"]
    attention_mask = data.get("attention_mask")
    causal_mask = data.get("causal_mask")
    n_actions = data.get("n_actions")
    pixel_values = data.get("pixel_values")
    if n_actions is not None:
        n_actions = int(np.array(n_actions).reshape(-1)[0])

    step = data.get("update_step")
    if step is not None:
        step = int(np.array(step).reshape(-1)[0])

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(pathlib.Path(args.npz).parent / "attention_visualization")

    attention_map_items = []
    if attn_weights is not None:
        attention_map_items.append(("attn_weights", attn_weights))
    if action_expert_attn_weights is not None:
        attention_map_items.append(("action_expert_attn_weights", action_expert_attn_weights))

    token_segments_for_causal = None
    token_labels_for_causal = None
    for key, attention_maps in attention_map_items:
        total_len = attention_maps.shape[-1]
        token_segments = _build_token_segments(
            input_ids=input_ids,
            total_len=total_len,
            image_token_id=image_token_id,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
            pad_token_id=pad_token_id,
            other_name='action_expert' if 'action_expert' in key else 'answer', 
        )
        token_labels = []
        for segment_name, segment_len in token_segments:
            token_labels.extend([segment_name] * int(segment_len))
        filtered_attention_maps, filtered_segments, match_q, match_k = _filter_padding_tokens(
            attention_maps, token_labels
        )
        print(f"filtered_segments: {filtered_segments}")
        token_labels_for_causal = token_labels
        token_segments_for_causal = token_segments
        token_segments_for_plot = {
            "segments": filtered_segments,
            "draw_q": match_q,
            "draw_k": match_k,
        }
        print(f"{key} token_segments: drawing matched axes only.")
        plot_multilayer_attention_maps(
            attention_maps=filtered_attention_maps,
            output_dir=output_dir,
            step=step,
            token_segments=token_segments_for_plot,
            layer_plot_every=args.layer_plot_every,
            filename_prefix=key,
        )
    if causal_mask is not None:
        token_segments_for_plot = None
        if token_labels_for_causal is not None:
            labels_len = len(token_labels_for_causal)
            q_len = causal_mask.shape[-2]
            k_len = causal_mask.shape[-1]
            if labels_len == q_len and labels_len == k_len:
                token_segments_for_plot = token_segments_for_causal
        _plot_causal_mask(causal_mask, output_dir, step=step, token_segments=token_segments_for_plot)
    if pixel_values is not None and attn_weights is not None:
        patch_size = cfg.get("patch_size")
        visualize_image_attention(
            attn_weights=attn_weights,
            action_expert_attn_weights=action_expert_attn_weights,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_token_id=image_token_id,
            sep_token_id=sep_token_id,
            n_actions=n_actions,
            output_dir=output_dir,
            step=step,
            patch_size=patch_size,
        )


if __name__ == "__main__":
    main()
