"""
visualize_tsne.py - 改进版：添加密度等高线
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import click


def plot_with_density(ax, points, color, title, cmap='Blues'):
    """绘制散点图 + 密度等高线"""
    
    # 计算密度
    xy = points.T
    try:
        z = gaussian_kde(xy)(xy)
        
        # 按密度排序（密度高的点画在上面）
        idx = z.argsort()
        x, y, z = points[idx, 0], points[idx, 1], z[idx]
        
        # 绘制散点（用密度着色）
        scatter = ax.scatter(x, y, c=z, s=10, alpha=0.6, cmap=cmap, 
                            edgecolors='none', rasterized=True)
        
        # 添加等高线
        from scipy.interpolate import griddata
        xi = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
        yi = np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='cubic')
        
        ax.contour(xi, yi, zi, levels=5, colors='black', 
                   linewidths=0.5, alpha=0.3)
        
        return scatter
        
    except np.linalg.LinAlgError:
        # 如果 KDE 失败（数据太少），回退到普通散点图
        scatter = ax.scatter(points[:, 0], points[:, 1], 
                            c=color, s=10, alpha=0.6)
        return scatter


@click.command()
@click.option('--baseline', required=True)
@click.option('--ours', required=True)
@click.option('-o', '--output', default='tsne_comparison.png')
@click.option('--max_samples', default=2000)
@click.option('--perplexity', default=100)
@click.option('--add_density', is_flag=True, default=True, 
              help='Add density contours')
def main(baseline, ours, output, max_samples, perplexity, add_density):
    
    # ========== 数据加载 ==========
    print("Loading features...")
    data_baseline = np.load(baseline)
    data_ours = np.load(ours)
    
    feat_baseline = data_baseline['features']
    feat_ours = data_ours['features']
    
    # 随机采样
    if len(feat_baseline) > max_samples:
        idx = np.random.choice(len(feat_baseline), max_samples, replace=False)
        feat_baseline = feat_baseline[idx]
    if len(feat_ours) > max_samples:
        idx = np.random.choice(len(feat_ours), max_samples, replace=False)
        feat_ours = feat_ours[idx]
    
    print(f"Baseline: {feat_baseline.shape}")
    print(f"Ours:     {feat_ours.shape}")
    
    # ========== 合并 + 标准化 ==========
    all_features = np.concatenate([feat_baseline, feat_ours], axis=0)
    labels = np.array([0] * len(feat_baseline) + [1] * len(feat_ours))
    
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    
    # ========== t-SNE ==========
    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000,
        verbose=1
    )
    embedded = tsne.fit_transform(all_features)
    
    emb_baseline = embedded[labels == 0]
    emb_ours = embedded[labels == 1]
    
    # ========== 绘图 ==========
    print("Plotting...")
    
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # --- 子图1: Baseline ---
    ax1 = axes[0]
    if add_density:
        scatter1 = plot_with_density(ax1, emb_baseline, None, 
                                     '(a) Baseline (DP)', cmap='Blues')
        plt.colorbar(scatter1, ax=ax1, label='Density', fraction=0.046, pad=0.04)
    else:
        ax1.scatter(emb_baseline[:, 0], emb_baseline[:, 1],
                   c='#3498db', alpha=0.5, s=8)
    
    ax1.set_title('(a) Baseline (DP)', fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # --- 子图2: Ours ---
    ax2 = axes[1]
    if add_density:
        scatter2 = plot_with_density(ax2, emb_ours, None,
                                     '(b) MI-DP (Ours)', cmap='Reds')
        plt.colorbar(scatter2, ax=ax2, label='Density', fraction=0.046, pad=0.04)
    else:
        ax2.scatter(emb_ours[:, 0], emb_ours[:, 1],
                   c='#e74c3c', alpha=0.5, s=8)
    
    ax2.set_title('(b) MI-DP (Ours)', fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # --- 子图3: Overlay (不加密度，保持清晰) ---
    ax3 = axes[2]
    ax3.scatter(emb_baseline[:, 0], emb_baseline[:, 1],
               c='#3498db', alpha=0.4, s=8, label='Baseline (DP)')
    ax3.scatter(emb_ours[:, 0], emb_ours[:, 1],
               c='#e74c3c', alpha=0.4, s=8, label='MI-DP (Ours)')
    
    ax3.set_title('(c) Comparison', fontweight='bold')
    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.legend(loc='upper right', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output}")
    
    # ========== 定量分析 ==========
    print("" + "=" * 60)
    print("Quantitative Analysis")
    print("=" * 60)
    
    # 1. 方差（紧凑度）
    var_baseline = np.var(emb_baseline, axis=0).sum()
    var_ours = np.var(emb_ours, axis=0).sum()
    reduction = (var_baseline - var_ours) / var_baseline * 100
    
    print(f"Intra-cluster Variance:")
    print(f"  Baseline: {var_baseline:.4f}")
    print(f"  MI-DP:    {var_ours:.4f}")
    print(f"  Reduction: {reduction:.1f}%")
    
    # 2. 密度峰值（如果启用了密度计算）
    if add_density:
        try:
            kde_baseline = gaussian_kde(emb_baseline.T)
            kde_ours = gaussian_kde(emb_ours.T)
            
            # 计算最大密度
            max_density_baseline = kde_baseline(emb_baseline.T).max()
            max_density_ours = kde_ours(emb_ours.T).max()
            
            print(f"Peak Density:")
            print(f"  Baseline: {max_density_baseline:.4f}")
            print(f"  MI-DP:    {max_density_ours:.4f}")
            print(f"  Increase: {(max_density_ours/max_density_baseline - 1)*100:.1f}%")
        except:
            pass
    
    # 3. Silhouette Score（聚类质量）
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        
        # 自动聚类
        n_clusters = 3
        kmeans_baseline = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_ours = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        labels_baseline = kmeans_baseline.fit_predict(feat_baseline)
        labels_ours = kmeans_ours.fit_predict(feat_ours)
        
        sil_baseline = silhouette_score(feat_baseline, labels_baseline)
        sil_ours = silhouette_score(feat_ours, labels_ours)
        
        print(f"Silhouette Score (higher is better):")
        print(f"  Baseline: {sil_baseline:.4f}")
        print(f"  MI-DP:    {sil_ours:.4f}")
        print(f"  Improvement: {(sil_ours - sil_baseline):.4f}")
    except Exception as e:
        print(f"\nCould not compute Silhouette Score: {e}")


if __name__ == '__main__':
    main()
