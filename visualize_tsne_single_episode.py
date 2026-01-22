"""
visualize_tsne_single_episode.py - 可视化单个 episode 的特征轨迹
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class FeatureExtractorWithEpisodeTracking:
    """改进版：记录每个 episode 的特征"""
    
    def __init__(self):
        self.episodes = []  # List of episodes
        self.current_episode = []
        self.current_timestep = 0
        
    def hook_fn(self, module, input, output):
        feat = output.detach().cpu().numpy()
        if feat.ndim == 3:
            feat = feat[:, -1, :]
        
        self.current_episode.append({
            'feature': feat[0],  # 假设 batch_size=1
            'timestep': self.current_timestep
        })
        self.current_timestep += 1
    
    def end_episode(self):
        """Episode 结束时调用"""
        if len(self.current_episode) > 0:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            self.current_timestep = 0
    
    def save(self, output_path):
        """保存为 pickle"""
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(self.episodes, f)
        print(f"Saved {len(self.episodes)} episodes")


def visualize_episode_trajectory():
    """可视化单个 episode 的特征轨迹"""
    import pickle
    
    # 加载数据
    with open('output/baseline_features/episodes.pkl', 'rb') as f:
        episodes_baseline = pickle.load(f)
    with open('output/midp_features/episodes.pkl', 'rb') as f:
        episodes_ours = pickle.load(f)
    
    # 选择一个成功的 episode（你需要根据 success 标记选择）
    ep_baseline = episodes_baseline[0]  # 第一个 episode
    ep_ours = episodes_ours[0]
    
    # 提取特征
    feat_baseline = np.array([step['feature'] for step in ep_baseline])
    feat_ours = np.array([step['feature'] for step in ep_ours])
    timesteps = np.array([step['timestep'] for step in ep_baseline])
    
    # 合并做 t-SNE（保证在同一空间）
    all_features = np.concatenate([feat_baseline, feat_ours], axis=0)
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_features)//4), random_state=42)
    embedded = tsne.fit_transform(all_features)
    
    emb_baseline = embedded[:len(feat_baseline)]
    emb_ours = embedded[len(feat_baseline):]
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline
    axes[0].plot(emb_baseline[:, 0], emb_baseline[:, 1], 
                 'o-', color='blue', alpha=0.6, markersize=4)
    axes[0].scatter(emb_baseline[0, 0], emb_baseline[0, 1], 
                   c='green', s=100, marker='*', label='Start', zorder=5)
    axes[0].scatter(emb_baseline[-1, 0], emb_baseline[-1, 1], 
                   c='red', s=100, marker='X', label='End', zorder=5)
    axes[0].set_title('(a) Baseline (DP) - Feature Trajectory')
    axes[0].legend()
    
    # Ours
    axes[1].plot(emb_ours[:, 0], emb_ours[:, 1], 
                 'o-', color='red', alpha=0.6, markersize=4)
    axes[1].scatter(emb_ours[0, 0], emb_ours[0, 1], 
                   c='green', s=100, marker='*', label='Start', zorder=5)
    axes[1].scatter(emb_ours[-1, 0], emb_ours[-1, 1], 
                   c='red', s=100, marker='X', label='End', zorder=5)
    axes[1].set_title('(b) MI-DP (Ours) - Feature Trajectory')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('tsne_episode_trajectory.png', dpi=300)
    print("Saved!")


if __name__ == '__main__':
    visualize_episode_trajectory()
