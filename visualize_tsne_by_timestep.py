"""
visualize_tsne_by_timestep.py - 按时间步着色
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def main():
    # 加载数据
    baseline = np.load('obs_features_dp.npz')
    ours = np.load('obs_features_midp_2.npz')
    
    feat_baseline = baseline['features']
    feat_ours = ours['features']
    
    timesteps_baseline = baseline['timesteps']  # 你需要在保存时记录这个
    timesteps_ours = ours['timesteps']
    
    # 标准化
    scaler = StandardScaler()
    feat_baseline = scaler.fit_transform(feat_baseline)
    feat_ours = scaler.fit_transform(feat_ours)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=100, random_state=42)
    emb_baseline = tsne.fit_transform(feat_baseline)
    emb_ours = tsne.fit_transform(feat_ours)
    
    # 绘图：用时间步着色
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline
    scatter1 = axes[0].scatter(emb_baseline[:, 0], emb_baseline[:, 1],
                               c=timesteps_baseline, cmap='viridis',
                               s=10, alpha=0.6)
    axes[0].set_title('(a) Baseline (DP)')
    plt.colorbar(scatter1, ax=axes[0], label='Timestep')
    
    # Ours
    scatter2 = axes[1].scatter(emb_ours[:, 0], emb_ours[:, 1],
                               c=timesteps_ours, cmap='viridis',
                               s=10, alpha=0.6)
    axes[1].set_title('(b) MI-DP (Ours)')
    plt.colorbar(scatter2, ax=axes[1], label='Timestep')
    
    plt.tight_layout()
    plt.savefig('tsne_by_timestep.png', dpi=300)
    print("Saved!")


if __name__ == '__main__':
    main()
