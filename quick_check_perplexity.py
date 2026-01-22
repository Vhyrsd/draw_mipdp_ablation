"""
quick_check_perplexity.py - 快速对比两个 perplexity
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 加载数据
baseline = np.load('obs_features_dp.npz')
ours = np.load('obs_features_midp.npz')

feat_baseline = baseline['features'][:1000]
feat_ours = ours['features'][:1000]

all_features = np.concatenate([feat_baseline, feat_ours], axis=0)
labels = np.array([0] * 1000 + [1] * 1000)

scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)

# 对比两个 perplexity
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, perp in enumerate([50, 100]):
    print(f"Running perplexity={perp}...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    embedded = tsne.fit_transform(all_features)
    
    ax = axes[idx]
    ax.scatter(embedded[labels==0, 0], embedded[labels==0, 1],
              c='blue', alpha=0.4, s=5, label='Baseline')
    ax.scatter(embedded[labels==1, 0], embedded[labels==1, 1],
              c='red', alpha=0.4, s=5, label='MI-DP')
    ax.set_title(f'Perplexity = {perp}')
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('perplexity_50_vs_100.png', dpi=200)
print("Saved!")
