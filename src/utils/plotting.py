import matplotlib.pyplot as plt
import numpy as np
import pathlib


def plot_l1_loss_as_bar(data, step, output_dir): 
    """
    Plot average L1 loss for each dimension as a bar chart and save to file.

    Args:
        data: np.ndarray of shape (..., D)
        step: int, Step number
        output_dir: Output directory to save the plot
    """
    mean_l1_loss = np.mean(data.reshape(-1, data.shape[-1]), axis=0)
    plt.figure(figsize=(12, 8))
    plt.bar(range(mean_l1_loss.shape[0]), mean_l1_loss)
    plt.xlabel('Dimension')
    plt.ylabel('Average L1 Loss')
    plt.title('Average L1 Loss for Each Dimension')
    plt.savefig(output_dir / f'l1_loss_as_bar_{step}.png', dpi=300, bbox_inches='tight')
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
