"""
Embedding 分布分析工具

该模块提供了分析词向量分布的统计函数
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Union
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_embedding_distribution(
    embeddings: Union[torch.Tensor, np.ndarray],
    sample_size: int = 1000,
    plot: bool = True,
    save_path: Optional[str] = None,
    bins: int = 50
) -> Dict[str, Any]:
    """
    分析 embedding 矩阵的统计分布特性
    
    Args:
        embeddings: 词向量矩阵 [vocab_size, D]
        sample_size: 用于可视化的采样数量（如果vocab_size很大）
        plot: 是否生成可视化图表
        save_path: 图表保存路径（如果为None且plot=True，则显示图表）
        bins: 直方图的bin数量
    
    Returns:
        包含各种统计信息的字典，包括：
        - global_stats: 全局统计信息（所有元素）
        - per_vector_stats: 每个词向量的统计信息
        - per_dimension_stats: 每个维度的统计信息
        - distribution_tests: 分布检验结果
    """
    
    # 转换为 numpy 数组
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    vocab_size, embedding_dim = embeddings.shape
    
    # 将所有embedding值展平
    all_values = embeddings.flatten()
    
    # ========== 1. 全局统计信息 ==========
    global_stats = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'total_elements': all_values.size,
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'var': float(np.var(all_values)),
        'min': float(np.min(all_values)),
        'max': float(np.max(all_values)),
        'median': float(np.median(all_values)),
        'q25': float(np.percentile(all_values, 25)),
        'q75': float(np.percentile(all_values, 75)),
        'skewness': float(stats.skew(all_values)),
        'kurtosis': float(stats.kurtosis(all_values)),
    }
    
    # ========== 2. 每个词向量的统计信息 ==========
    per_vector_means = np.mean(embeddings, axis=1)
    per_vector_stds = np.std(embeddings, axis=1)
    per_vector_norms = np.linalg.norm(embeddings, axis=1)
    
    per_vector_stats = {
        'means': {
            'mean': float(np.mean(per_vector_means)),
            'std': float(np.std(per_vector_means)),
            'min': float(np.min(per_vector_means)),
            'max': float(np.max(per_vector_means)),
        },
        'stds': {
            'mean': float(np.mean(per_vector_stds)),
            'std': float(np.std(per_vector_stds)),
            'min': float(np.min(per_vector_stds)),
            'max': float(np.max(per_vector_stds)),
        },
        'norms': {
            'mean': float(np.mean(per_vector_norms)),
            'std': float(np.std(per_vector_norms)),
            'min': float(np.min(per_vector_norms)),
            'max': float(np.max(per_vector_norms)),
        }
    }
    
    # ========== 3. 每个维度的统计信息 ==========
    per_dim_means = np.mean(embeddings, axis=0)
    per_dim_stds = np.std(embeddings, axis=0)
    
    per_dimension_stats = {
        'means': {
            'mean': float(np.mean(per_dim_means)),
            'std': float(np.std(per_dim_means)),
            'min': float(np.min(per_dim_means)),
            'max': float(np.max(per_dim_means)),
        },
        'stds': {
            'mean': float(np.mean(per_dim_stds)),
            'std': float(np.std(per_dim_stds)),
            'min': float(np.min(per_dim_stds)),
            'max': float(np.max(per_dim_stds)),
        }
    }
    
    # ========== 4. 分布检验 ==========
    # 使用采样来加速大规模数据的检验
    if all_values.size > 5000:
        sample_indices = np.random.choice(all_values.size, 5000, replace=False)
        sample_values = all_values[sample_indices]
    else:
        sample_values = all_values
    
    # 正态性检验 (Shapiro-Wilk test)
    if len(sample_values) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(sample_values)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
    
    # Kolmogorov-Smirnov 检验（与标准正态分布比较）
    normalized_values = (sample_values - np.mean(sample_values)) / (np.std(sample_values) + 1e-8)
    ks_stat, ks_p = stats.kstest(normalized_values, 'norm')
    
    # Anderson-Darling 检验
    anderson_result = stats.anderson(sample_values, dist='norm')
    
    # 判断分布类型
    is_normal = shapiro_p > 0.05 and ks_p > 0.05
    is_uniform = np.abs(global_stats['skewness']) < 0.1 and np.abs(global_stats['kurtosis']) < 0.1
    
    distribution_tests = {
        'shapiro_wilk': {
            'statistic': float(shapiro_stat),
            'p_value': float(shapiro_p),
            'is_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None
        },
        'kolmogorov_smirnov': {
            'statistic': float(ks_stat),
            'p_value': float(ks_p),
            'is_normal': ks_p > 0.05
        },
        'anderson_darling': {
            'statistic': float(anderson_result.statistic),
            'critical_values': anderson_result.critical_values.tolist(),
            'significance_levels': anderson_result.significance_level.tolist()
        },
        'distribution_type': {
            'likely_normal': is_normal,
            'likely_uniform': is_uniform,
            'description': _get_distribution_description(global_stats, is_normal, is_uniform)
        }
    }
    
    # ========== 5. 可视化 ==========
    if plot:
        _plot_embedding_analysis(
            embeddings=embeddings,
            all_values=all_values,
            per_vector_norms=per_vector_norms,
            per_vector_means=per_vector_means,
            per_dim_means=per_dim_means,
            per_dim_stds=per_dim_stds,
            sample_size=sample_size,
            bins=bins,
            save_path=save_path
        )
    
    # 返回完整的统计信息
    results = {
        'global_stats': global_stats,
        'per_vector_stats': per_vector_stats,
        'per_dimension_stats': per_dimension_stats,
        'distribution_tests': distribution_tests,
    }
    
    return results


def _get_distribution_description(global_stats: Dict, is_normal: bool, is_uniform: bool) -> str:
    """生成分布类型的描述"""
    skew = global_stats['skewness']
    kurt = global_stats['kurtosis']
    
    descriptions = []
    
    if is_normal:
        descriptions.append("Approximately Normal")
    elif is_uniform:
        descriptions.append("Approximately Uniform")
    else:
        if np.abs(skew) < 0.5:
            descriptions.append("Symmetric")
        elif skew > 0.5:
            descriptions.append("Right-skewed")
        else:
            descriptions.append("Left-skewed")
        
        if kurt > 3:
            descriptions.append("Leptokurtic")
        elif kurt < -1:
            descriptions.append("Platykurtic")
    
    return ", ".join(descriptions) if descriptions else "Unknown Distribution"


def _plot_embedding_analysis(
    embeddings: np.ndarray,
    all_values: np.ndarray,
    per_vector_norms: np.ndarray,
    per_vector_means: np.ndarray,
    per_dim_means: np.ndarray,
    per_dim_stds: np.ndarray,
    sample_size: int,
    bins: int,
    save_path: Optional[str]
):
    """生成 embedding 分析的可视化图表"""
    
    # Set default font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(20, 12))
    
    # 采样数据用于可视化
    if len(all_values) > sample_size:
        sample_indices = np.random.choice(len(all_values), sample_size, replace=False)
        sampled_values = all_values[sample_indices]
    else:
        sampled_values = all_values
    
    # 1. Global value distribution
    ax1 = plt.subplot(3, 4, 1)
    ax1.hist(sampled_values, bins=bins, alpha=0.7, edgecolor='black', density=True)
    ax1.set_title('Global Value Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)
    
    # Add normal distribution fitting curve
    mu, sigma = np.mean(sampled_values), np.std(sampled_values)
    x = np.linspace(sampled_values.min(), sampled_values.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
    ax1.legend()
    
    # 2. Q-Q plot (normality test)
    ax2 = plt.subplot(3, 4, 2)
    stats.probplot(sampled_values, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax3 = plt.subplot(3, 4, 3)
    ax3.boxplot(sampled_values, vert=True)
    ax3.set_title('Box Plot', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative Distribution Function (CDF)
    ax4 = plt.subplot(3, 4, 4)
    sorted_values = np.sort(sampled_values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax4.plot(sorted_values, cdf, linewidth=2)
    ax4.set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.grid(True, alpha=0.3)
    
    # 5. Per-vector norm distribution
    ax5 = plt.subplot(3, 4, 5)
    ax5.hist(per_vector_norms, bins=bins, alpha=0.7, edgecolor='black', color='green')
    ax5.set_title('Vector Norm Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('L2 Norm')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)
    
    # 6. Per-vector mean distribution
    ax6 = plt.subplot(3, 4, 6)
    ax6.hist(per_vector_means, bins=bins, alpha=0.7, edgecolor='black', color='orange')
    ax6.set_title('Vector Mean Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Mean')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    
    # 7. Mean per dimension
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(per_dim_means, linewidth=1.5, color='purple')
    ax7.set_title('Mean per Dimension', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Dimension Index')
    ax7.set_ylabel('Mean')
    ax7.grid(True, alpha=0.3)
    
    # 8. Std per dimension
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(per_dim_stds, linewidth=1.5, color='brown')
    ax8.set_title('Std per Dimension', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Dimension Index')
    ax8.set_ylabel('Standard Deviation')
    ax8.grid(True, alpha=0.3)
    
    # 9. Heatmap: embedding sample
    ax9 = plt.subplot(3, 4, 9)
    if embeddings.shape[0] > 50:
        sample_vocab = np.random.choice(embeddings.shape[0], 50, replace=False)
        sample_embeddings = embeddings[sample_vocab, :]
    else:
        sample_embeddings = embeddings
    
    if sample_embeddings.shape[1] > 50:
        sample_dims = np.random.choice(sample_embeddings.shape[1], 50, replace=False)
        sample_embeddings = sample_embeddings[:, sample_dims]
    
    im = ax9.imshow(sample_embeddings, aspect='auto', cmap='viridis', interpolation='nearest')
    ax9.set_title('Embedding Heatmap (Sampled)', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Dimension')
    ax9.set_ylabel('Vector')
    plt.colorbar(im, ax=ax9)
    
    # 10. Kernel Density Estimation
    ax10 = plt.subplot(3, 4, 10)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(sampled_values)
        x_kde = np.linspace(sampled_values.min(), sampled_values.max(), 200)
        ax10.plot(x_kde, kde(x_kde), linewidth=2, color='darkblue')
        ax10.fill_between(x_kde, kde(x_kde), alpha=0.3)
        ax10.set_title('Kernel Density Estimation', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Value')
        ax10.set_ylabel('Density')
        ax10.grid(True, alpha=0.3)
    except:
        ax10.text(0.5, 0.5, 'KDE Not Available', ha='center', va='center')
        ax10.set_title('Kernel Density Estimation', fontsize=12, fontweight='bold')
    
    # 11. Cosine similarity distribution (sampled)
    ax11 = plt.subplot(3, 4, 11)
    n_samples = min(100, embeddings.shape[0])
    sample_indices = np.random.choice(embeddings.shape[0], n_samples, replace=False)
    sample_emb = embeddings[sample_indices]
    
    # Calculate cosine similarity
    norms = np.linalg.norm(sample_emb, axis=1, keepdims=True)
    normalized_emb = sample_emb / (norms + 1e-8)
    cosine_sim = np.dot(normalized_emb, normalized_emb.T)
    
    # Only take upper triangle (excluding diagonal)
    triu_indices = np.triu_indices_from(cosine_sim, k=1)
    cosine_similarities = cosine_sim[triu_indices]
    
    ax11.hist(cosine_similarities, bins=50, alpha=0.7, edgecolor='black', color='teal')
    ax11.set_title('Cosine Similarity Distribution', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Cosine Similarity')
    ax11.set_ylabel('Frequency')
    ax11.grid(True, alpha=0.3)
    
    # 12. Statistical summary text
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary_text = f"""
Statistical Summary
{'='*30}
Vocab Size: {embeddings.shape[0]:,}
Embedding Dim: {embeddings.shape[1]:,}

Global Statistics:
  Mean: {np.mean(all_values):.4f}
  Std: {np.std(all_values):.4f}
  Min: {np.min(all_values):.4f}
  Max: {np.max(all_values):.4f}
  
Skewness: {stats.skew(all_values):.4f}
Kurtosis: {stats.kurtosis(all_values):.4f}

Vector Norms:
  Mean: {np.mean(per_vector_norms):.4f}
  Std: {np.std(per_vector_norms):.4f}
"""
    
    ax12.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
              fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_analysis_report(analysis_results: Dict[str, Any]):
    """打印格式化的分析报告"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "EMBEDDING DISTRIBUTION ANALYSIS REPORT")
    print("=" * 80)
    
    # Global statistics
    print("\n📊 Global Statistics")
    print("-" * 80)
    gs = analysis_results['global_stats']
    print(f"  Vocab Size:              {gs['vocab_size']:,}")
    print(f"  Embedding Dim:           {gs['embedding_dim']:,}")
    print(f"  Total Elements:          {gs['total_elements']:,}")
    print(f"\n  Mean:                    {gs['mean']:.6f}")
    print(f"  Std:                     {gs['std']:.6f}")
    print(f"  Variance:                {gs['var']:.6f}")
    print(f"  Min:                     {gs['min']:.6f}")
    print(f"  Max:                     {gs['max']:.6f}")
    print(f"  Median:                  {gs['median']:.6f}")
    print(f"  25% Quantile (Q1):       {gs['q25']:.6f}")
    print(f"  75% Quantile (Q3):       {gs['q75']:.6f}")
    print(f"\n  Skewness:                {gs['skewness']:.6f}")
    print(f"  Kurtosis:                {gs['kurtosis']:.6f}")
    
    # Per-vector statistics
    print("\n📈 Per-Vector Statistics")
    print("-" * 80)
    pvs = analysis_results['per_vector_stats']
    print("  Mean Distribution:")
    print(f"    Mean: {pvs['means']['mean']:.6f}, Std: {pvs['means']['std']:.6f}")
    print(f"    Range: [{pvs['means']['min']:.6f}, {pvs['means']['max']:.6f}]")
    print("\n  Std Distribution:")
    print(f"    Mean: {pvs['stds']['mean']:.6f}, Std: {pvs['stds']['std']:.6f}")
    print(f"    Range: [{pvs['stds']['min']:.6f}, {pvs['stds']['max']:.6f}]")
    print("\n  L2 Norm Distribution:")
    print(f"    Mean: {pvs['norms']['mean']:.6f}, Std: {pvs['norms']['std']:.6f}")
    print(f"    Range: [{pvs['norms']['min']:.6f}, {pvs['norms']['max']:.6f}]")
    
    # Per-dimension statistics
    print("\n📉 Per-Dimension Statistics")
    print("-" * 80)
    pds = analysis_results['per_dimension_stats']
    print("  Mean Distribution:")
    print(f"    Mean: {pds['means']['mean']:.6f}, Std: {pds['means']['std']:.6f}")
    print(f"    Range: [{pds['means']['min']:.6f}, {pds['means']['max']:.6f}]")
    print("\n  Std Distribution:")
    print(f"    Mean: {pds['stds']['mean']:.6f}, Std: {pds['stds']['std']:.6f}")
    print(f"    Range: [{pds['stds']['min']:.6f}, {pds['stds']['max']:.6f}]")
    
    # Distribution tests
    print("\n🔬 Distribution Test Results")
    print("-" * 80)
    dt = analysis_results['distribution_tests']
    
    print("  Shapiro-Wilk Normality Test:")
    sw = dt['shapiro_wilk']
    print(f"    Statistic: {sw['statistic']:.6f}")
    print(f"    P-value: {sw['p_value']:.6f}")
    print(f"    Conclusion: {'Accept normality' if sw.get('is_normal') else 'Reject normality'} (α=0.05)")
    
    print("\n  Kolmogorov-Smirnov Test:")
    ks = dt['kolmogorov_smirnov']
    print(f"    Statistic: {ks['statistic']:.6f}")
    print(f"    P-value: {ks['p_value']:.6f}")
    print(f"    Conclusion: {'Accept normality' if ks['is_normal'] else 'Reject normality'} (α=0.05)")
    
    print("\n  Anderson-Darling Test:")
    ad = dt['anderson_darling']
    print(f"    Statistic: {ad['statistic']:.6f}")
    print(f"    Critical Values: {ad['critical_values']}")
    print(f"    Significance Levels: {ad['significance_levels']}")
    
    print("\n  📝 Distribution Type Assessment:")
    dist_type = dt['distribution_type']
    print(f"    Likely Normal: {'Yes' if dist_type['likely_normal'] else 'No'}")
    print(f"    Likely Uniform: {'Yes' if dist_type['likely_uniform'] else 'No'}")
    print(f"    Description: {dist_type['description']}")
    
    print("\n" + "=" * 80 + "\n")


# Usage examples
if __name__ == "__main__":
    # Generate test data
    print("Generating test data...")
    
    # Test 1: Normal distribution
    vocab_size, embedding_dim = 1000, 512
    embeddings_normal = np.random.randn(vocab_size, embedding_dim) * 0.02
    
    print("\nTest 1: Normal Distribution Embedding")
    results_normal = analyze_embedding_distribution(
        embeddings_normal,
        sample_size=1000,
        plot=True,
        save_path="embedding_analysis_normal.png"
    )
    print_analysis_report(results_normal)
    
    # Test 2: Uniform distribution
    embeddings_uniform = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
    
    print("\nTest 2: Uniform Distribution Embedding")
    results_uniform = analyze_embedding_distribution(
        embeddings_uniform,
        sample_size=1000,
        plot=True,
        save_path="embedding_analysis_uniform.png"
    )
    print_analysis_report(results_uniform)

