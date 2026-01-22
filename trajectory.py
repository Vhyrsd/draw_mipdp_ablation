"""
可视化单个episode的轨迹
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import click
from pathlib import Path

@click.command()
@click.option('--feature_dir', required=True)
@click.option('--output', default='trajectory_vis.png')
@click.option('--perplexity', default=30, type=int)
@click.option('--n_episodes', default=5, type=int, help='Number of episodes to show')
def main(feature_dir, output, perplexity, n_episodes):
    feature_path = Path(feature_dir) / 'eval_features.pkl'
    
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    # 收集所有特征用于 t-SNE
    all_features = []
    episode_indices = []
    
    for ep_idx, episode in enumerate(data['episodes'][:n_episodes]):
        features = episode['features']
        if isinstance(features, np.ndarray) and features.size > 0:
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            all_features.append(features)
            episode_indices.extend([ep_idx] * len(features))
    
    all_features = np.vstack(all_features)
    episode_indices = np.array(episode_indices)
    
    print(f"Running t-SNE on {len(all_features)} samples from {n_episodes} episodes...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(all_features)
    
    # 绘制轨迹
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_episodes))
    
    for ep_idx in range(n_episodes):
        mask = episode_indices == ep_idx
        traj = features_2d[mask]
        
        # 绘制轨迹线
        ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[ep_idx], 
               alpha=0.3, linewidth=2)
        
        # 绘制点，颜色渐变表示时间
        scatter = ax.scatter(traj[:, 0], traj[:, 1], 
                           c=np.arange(len(traj)), 
                           cmap='viridis', 
                           s=50, alpha=0.7, 
                           edgecolors=colors[ep_idx], linewidths=2)
        
        # 标记起点和终点
        ax.scatter(traj[0, 0], traj[0, 1], marker='o', s=200, 
                  color=colors[ep_idx], edgecolors='black', linewidths=2,
                  label=f'Episode {ep_idx} start', zorder=10)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker='s', s=200, 
                  color=colors[ep_idx], edgecolors='black', linewidths=2,
                  zorder=10)
    
    ax.set_title(f'Episode Trajectories in Feature Space (perplexity={perplexity})', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(loc='best', fontsize=8)
    
    plt.colorbar(scatter, ax=ax, label='Timestep')
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Trajectory visualization saved to {output}")

if __name__ == '__main__':
    main()
