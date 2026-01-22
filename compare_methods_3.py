"""
可视化这两个指标
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_feature_quality_metrics(baseline_metrics, ours_metrics, output='metric_explanation.png'):
    """
    可视化 feature_variance 和 inter_env_distance
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Feature Variance 对比
    ax1 = axes[0, 0]
    methods = ['Baseline', 'Ours']
    variances = [baseline_metrics['feature_variance'], 
                 ours_metrics['feature_variance']]
    
    bars = ax1.bar(methods, variances, color=['coral', 'skyblue'], alpha=0.8)
    ax1.set_ylabel('Feature Variance', fontsize=12)
    ax1.set_title('Feature Variance Comparison\n(Higher = More Expressive)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars, variances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加改进百分比
    improvement = (ours_metrics['feature_variance'] - baseline_metrics['feature_variance']) / \
                  baseline_metrics['feature_variance'] * 100
    ax1.text(0.5, max(variances) * 0.5, 
            f'Improvement: {improvement:+.2f}%',
            ha='center', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            transform=ax1.transData)
    
    # 2. Inter-Environment Distance 对比
    ax2 = axes[0, 1]
    distances = [baseline_metrics['inter_env_distance'], 
                 ours_metrics['inter_env_distance']]
    
    bars = ax2.bar(methods, distances, color=['coral', 'skyblue'], alpha=0.8)
    ax2.set_ylabel('Inter-Environment Distance', fontsize=12)
    ax2.set_title('Inter-Environment Distance Comparison\n(Higher = Better Separation)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, distances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    improvement = (ours_metrics['inter_env_distance'] - baseline_metrics['inter_env_distance']) / \
                  baseline_metrics['inter_env_distance'] * 100
    ax2.text(0.5, max(distances) * 0.5, 
            f'Improvement: {improvement:+.2f}%',
            ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            transform=ax2.transData)
    
    # 3. 二维质量空间
    ax3 = axes[0, 2]
    
    # 归一化到 [0, 1]
    var_min = min(variances)
    var_max = max(variances)
    dist_min = min(distances)
    dist_max = max(distances)
    
    baseline_var_norm = (variances[0] - var_min) / (var_max - var_min) if var_max > var_min else 0.5
    ours_var_norm = (variances[1] - var_min) / (var_max - var_min) if var_max > var_min else 0.5
    baseline_dist_norm = (distances[0] - dist_min) / (dist_max - dist_min) if dist_max > dist_min else 0.5
    ours_dist_norm = (distances[1] - dist_min) / (dist_max - dist_min) if dist_max > dist_min else 0.5
    
    # 绘制质量空间
    ax3.scatter(baseline_var_norm, baseline_dist_norm, s=300, c='coral', 
               marker='o', edgecolors='black', linewidths=2, label='Baseline', alpha=0.8)
    ax3.scatter(ours_var_norm, ours_dist_norm, s=300, c='skyblue', 
               marker='s', edgecolors='black', linewidths=2, label='Ours', alpha=0.8)
    
    # 添加箭头
    ax3.annotate('', xy=(ours_var_norm, ours_dist_norm), 
                xytext=(baseline_var_norm, baseline_dist_norm),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # 标注区域
    ax3.add_patch(Rectangle((0.5, 0.5), 0.5, 0.5, 
                            facecolor='lightgreen', alpha=0.3, label='Ideal Region'))
    ax3.text(0.75, 0.75, 'Ideal\nRegion', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Feature Variance (normalized)', fontsize=11)
    ax3.set_ylabel('Inter-Env Distance (normalized)', fontsize=11)
    ax3.set_title('Feature Quality Space', fontsize=12, fontweight='bold')
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature Variance 的含义示意图
    ax4 = axes[1, 0]
    ax4.axis('off')
    
    # 低方差示例
    low_var_features = np.random.normal(0, 0.1, (100, 2))
    ax4_sub1 = fig.add_axes([0.08, 0.15, 0.12, 0.25])
    ax4_sub1.scatter(low_var_features[:, 0], low_var_features[:, 1], 
                    c='coral', alpha=0.6, s=20)
    ax4_sub1.set_title('Low Variance\n(Less Expressive)', fontsize=10)
    ax4_sub1.set_xlim(-1, 1)
    ax4_sub1.set_ylim(-1, 1)
    
    # 高方差示例
    high_var_features = np.random.normal(0, 0.5, (100, 2))
    ax4_sub2 = fig.add_axes([0.22, 0.15, 0.12, 0.25])
    ax4_sub2.scatter(high_var_features[:, 0], high_var_features[:, 1], 
                    c='skyblue', alpha=0.6, s=20)
    ax4_sub2.set_title('High Variance\n(More Expressive)', fontsize=10)
    ax4_sub2.set_xlim(-1, 1)
    ax4_sub2.set_ylim(-1, 1)
    
    # 5. Inter-Env Distance 的含义示意图
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    # 低距离示例（环境重叠）
    ax5_sub1 = fig.add_axes([0.41, 0.15, 0.12, 0.25])
    env1 = np.random.normal([0, 0], 0.2, (50, 2))
    env2 = np.random.normal([0.3, 0.3], 0.2, (50, 2))
    env3 = np.random.normal([0.6, 0], 0.2, (50, 2))
    ax5_sub1.scatter(env1[:, 0], env1[:, 1], c='red', alpha=0.6, s=20, label='Env 1')
    ax5_sub1.scatter(env2[:, 0], env2[:, 1], c='green', alpha=0.6, s=20, label='Env 2')
    ax5_sub1.scatter(env3[:, 0], env3[:, 1], c='blue', alpha=0.6, s=20, label='Env 3')
    ax5_sub1.set_title('Low Inter-Env Distance\n(Overlapping)', fontsize=10)
    ax5_sub1.set_xlim(-1, 2)
    ax5_sub1.set_ylim(-1, 2)
    
    # 高距离示例（环境分离）
    ax5_sub2 = fig.add_axes([0.55, 0.15, 0.12, 0.25])
    env1 = np.random.normal([-1, -1], 0.2, (50, 2))
    env2 = np.random.normal([0, 1], 0.2, (50, 2))
    env3 = np.random.normal([1.5, -0.5], 0.2, (50, 2))
    ax5_sub2.scatter(env1[:, 0], env1[:, 1], c='red', alpha=0.6, s=20, label='Env 1')
    ax5_sub2.scatter(env2[:, 0], env2[:, 1], c='green', alpha=0.6, s=20, label='Env 2')
    ax5_sub2.scatter(env3[:, 0], env3[:, 1], c='blue', alpha=0.6, s=20, label='Env 3')
    ax5_sub2.set_title('High Inter-Env Distance\n(Well Separated)', fontsize=10)
    ax5_sub2.set_xlim(-2, 3)
    ax5_sub2.set_ylim(-2, 3)
    
    # 6. 综合评分
    ax6 = axes[1, 2]
    
    # 计算综合得分（归一化后的加权和）
    baseline_score = baseline_var_norm * 0.5 + baseline_dist_norm * 0.5
    ours_score = ours_var_norm * 0.5 + ours_dist_norm * 0.5
    
    scores = [baseline_score, ours_score]
    bars = ax6.barh(methods, scores, color=['coral', 'skyblue'], alpha=0.8)
    ax6.set_xlabel('Overall Quality Score', fontsize=12)
    ax6.set_title('Overall Feature Quality\n(Normalized Composite Score)', 
                  fontsize=12, fontweight='bold')
    ax6.set_xlim(0, 1)
    ax6.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, scores):
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('Feature Quality Metrics: Detailed Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Metric explanation saved to {output}")

# 使用示例
if __name__ == '__main__':
    # 示例数据
    baseline_metrics = {
        'feature_variance': 0.0008,
        'inter_env_distance': 0.2363
    }
    
    ours_metrics = {
        'feature_variance': 0.0017,  # +33% 提升
        'inter_env_distance': 0.3416   # +26% 提升
    }
    
    visualize_feature_quality_metrics(baseline_metrics, ours_metrics)
