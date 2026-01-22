"""
对特征进行聚类分析
Usage:
python cluster_analysis.py --feature_dir feats_dir --n_clusters 5
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
import click

@click.command()
@click.option('--feature_dir', required=True)
@click.option('--n_clusters', default=5, type=int)
def main(feature_dir, n_clusters):
    # 加载特征
    feature_path = Path(feature_dir) / 'eval_features.pkl'
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    # 收集所有特征
    all_features = []
    for ep in data['episodes']:
        if isinstance(ep['features'], np.ndarray) and ep['features'].size > 0:
            all_features.append(ep['features'])
    
    all_features = np.vstack(all_features)
    print(f"Total samples: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")
    
    # K-means 聚类
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_features)
    
    # 加载 t-SNE 结果
    tsne_path = Path('tsne_visualization.pkl')
    with open(tsne_path, 'rb') as f:
        tsne_data = pickle.load(f)
    
    features_2d = tsne_data['features_2d']
    
    # 可视化聚类结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'K-means Clustering (k={n_clusters})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # 标注聚类中心
    centers_2d = []
    for i in range(n_clusters):
        mask = cluster_labels == i
        center = features_2d[mask].mean(axis=0)
        centers_2d.append(center)
        plt.plot(center[0], center[1], 'k*', markersize=20, 
                markeredgewidth=2, markeredgecolor='white')
        plt.text(center[0], center[1], f'C{i}', 
                fontsize=12, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f'cluster_analysis_k{n_clusters}.png', dpi=300)
    print(f"✓ Saved to cluster_analysis_k{n_clusters}.png")
    
    # 分析每个聚类
    print(f"=== Cluster Statistics ===")
    for i in range(n_clusters):
        mask = cluster_labels == i
        print(f"\nCluster {i}:")
        print(f"  Size: {mask.sum()} samples ({mask.sum()/len(mask)*100:.1f}%)")

if __name__ == '__main__':
    main()
