import os

import torch
import numpy as np

from src.utils.metric import (
    get_action_accuracy,
    compute_smoothness_metrics,
    compute_error_heatmap,
    compute_covariance_matrix,
    compute_loss_over_time,
    compute_trajectory_metrics,
    compute_per_dimension_metrics,
    compute_all_metrics,
    plot_smoothness_analysis,
    plot_smoothness_error_heatmaps,
    plot_error_heatmap,
    plot_covariance_matrices,
    plot_loss_over_time,
    plot_trajectory_comparison,
    plot_all_visualizations,
)


def main():
    """
    Test function with random data to verify all metrics and visualizations work correctly.
    """
    print("Generating random test data...")
    # Create output directory
    output_dir = 'outputs/eval'
    os.makedirs(output_dir, exist_ok=True)
    print(f"   Saving visualizations to: {output_dir}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate random data
    batch_size = 32
    horizon = 32
    action_dim = 48

    # Generate smooth GT trajectory (sine waves with different frequencies)
    t = torch.linspace(0, 4 * np.pi, horizon).unsqueeze(0).unsqueeze(-1)  # [1, H, 1]
    gt = torch.zeros(batch_size, horizon, action_dim)
    for d in range(action_dim):
        freq = 0.5 + d * 0.2
        phase = d * 0.5
        gt[:, :, d] = torch.sin(freq * t.squeeze(-1) + phase) + torch.randn(batch_size, horizon) * 0.1

    # Generate pred with some error (add noise and slight phase shift)
    pred = gt.clone()
    # Add systematic error: larger errors at certain time steps and dimensions
    error_pattern = torch.zeros(batch_size, horizon, action_dim)
    error_pattern[:, 5:10, 2:5] = 0.3  # Larger errors in middle time steps for some dimensions
    error_pattern[:, -5:, :] = 0.2  # Larger errors at the end
    pred = pred + error_pattern + torch.randn(batch_size, horizon, action_dim) * 0.15

    print(f"Data shape: GT={gt.shape}, Pred={pred.shape}")

    # Define action dimension names
    action_dim_names = [f'Joint_{i}' for i in range(action_dim)]

    # Test all metric computation functions
    print("\n=== Testing Metric Computation Functions ===")

    # 1. Action accuracy
    print("1. Computing action accuracy...")
    thresholds = [0.1, 0.2, 0.3]
    accuracy = get_action_accuracy(gt, pred, thresholds=thresholds)
    print(f"   Accuracy at thresholds {thresholds}: {accuracy.cpu().numpy()}")

    # 2. Smoothness metrics
    print("2. Computing smoothness metrics...")
    smoothness = compute_smoothness_metrics(gt, pred)
    print(f"   First diff error: {smoothness['first_diff_error'].item():.4f}")
    print(f"   Second diff error: {smoothness['second_diff_error'].item():.4f}")
    print(f"   First diff error heatmap shape: {smoothness['first_diff_error_heatmap'].shape}")
    print(f"   Second diff error heatmap shape: {smoothness['second_diff_error_heatmap'].shape}")

    # 3. Error heatmap
    print("3. Computing error heatmap...")
    error_heatmap = compute_error_heatmap(gt, pred)
    print(f"   Error heatmap shape: {error_heatmap.shape}")
    print(f"   Mean error: {torch.mean(error_heatmap).item():.4f}")

    # 4. Covariance matrices
    print("4. Computing covariance matrices...")
    cov_dict = compute_covariance_matrix(gt, pred)
    print(f"   GT cov shape: {cov_dict['gt_cov'].shape}")
    print(f"   Pred cov shape: {cov_dict['pred_cov'].shape}")
    print(f"   Error cov shape: {cov_dict['error_cov'].shape}")

    # 5. Loss over time
    print("5. Computing loss over time...")
    loss_over_time = compute_loss_over_time(gt, pred, loss_type='l1')
    print(f"   Loss over time shape: {loss_over_time.shape}")
    print(f"   Mean loss: {torch.mean(loss_over_time).item():.4f}")

    # 6. Trajectory metrics
    print("6. Computing trajectory metrics...")
    traj_metrics = compute_trajectory_metrics(gt, pred)
    print(f"   Endpoint error shape: {traj_metrics['endpoint_error'].shape}")
    print(f"   Mean endpoint error: {torch.mean(traj_metrics['endpoint_error']).item():.4f}")
    print(f"   Mean trajectory length error: {torch.mean(traj_metrics['trajectory_length_error']).item():.4f}")

    # 7. Per dimension metrics
    print("7. Computing per dimension metrics...")
    dim_metrics = compute_per_dimension_metrics(gt, pred)
    print(f"   MAE per dim shape: {dim_metrics['mae_per_dim'].shape}")
    print(f"   MAE per dim: {dim_metrics['mae_per_dim'].cpu().numpy()}")

    # 8. All metrics
    print("8. Computing all metrics...")
    all_metrics = compute_all_metrics(gt, pred)
    print(f"   Overall MAE: {all_metrics['overall_mae'].item():.4f}")
    print(f"   Number of metrics: {len(all_metrics)}")

    # Test all visualization functions
    print("\n=== Testing Visualization Functions ===")

    # 1. Smoothness analysis
    print("1. Plotting smoothness analysis...")
    plot_smoothness_analysis(
        gt, pred,
        save_path=os.path.join(output_dir, 'smoothness_analysis.png'),
        action_dim_names=action_dim_names,
    )

    # 2. Smoothness error heatmaps
    print("2. Plotting smoothness error heatmaps...")
    plot_smoothness_error_heatmaps(
        gt, pred,
        save_path=os.path.join(output_dir, 'smoothness_error_heatmaps.png'),
        action_dim_names=action_dim_names,
    )

    # 3. Error heatmap
    print("3. Plotting error heatmap...")
    plot_error_heatmap(
        gt, pred,
        save_path=os.path.join(output_dir, 'error_heatmap.png'),
        action_dim_names=action_dim_names,
    )

    # 4. Covariance matrices
    print("4. Plotting covariance matrices...")
    plot_covariance_matrices(
        gt, pred,
        save_path=os.path.join(output_dir, 'covariance_matrices.png'),
        action_dim_names=action_dim_names,
    )

    # 5. Loss over time
    print("5. Plotting loss over time...")
    plot_loss_over_time(
        gt, pred,
        save_path=os.path.join(output_dir, 'loss_over_time.png'),
        loss_type='l1',
    )

    # 6. Trajectory comparison
    print("6. Plotting trajectory comparison...")
    plot_trajectory_comparison(
        gt, pred,
        save_path=os.path.join(output_dir, 'trajectory_comparison.png'),
        action_dim_names=action_dim_names,
        num_samples=5,
    )

    # 7. All visualizations
    print("7. Plotting all visualizations...")
    plot_all_visualizations(
        gt, pred,
        output_dir=output_dir,
        action_dim_names=action_dim_names,
        prefix='all_',
    )

    print("\n=== Test Complete ===")
    print(f"All visualizations saved to: {output_dir}")
    print("Check the output directory for generated plots.")


if __name__ == "__main__":
    main()
