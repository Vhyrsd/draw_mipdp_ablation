"""
可视化单个episode的轨迹
Usage:
python visualize_trajectory.py --feature_dir feats_dir --episode_id 0
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE
import click
from pathlib import Path

@click.command()
@click.option('--feature_dir', required=True)
@click.option('--episode_id', default=0, type=int)
@click.option('--output', default='trajectory.gif')
def main(feature_dir, episode_id, output):
    # 加载数据
    feature_path = Path(feature_dir) / 'eval_features.pkl'
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    # 加载 t-SNE 结果
    tsne_path = Path('tsne_visualization_ours.pkl')
    with open(tsne_path, 'rb') as f:
        tsne_data = pickle.load(f)
    
    features_2d = tsne_data['features_2d']
    env_ids = tsne_data['env_ids']
    timesteps = tsne_data['timesteps']
    
    # 提取指定episode的轨迹
    episode = data['episodes'][episode_id]
    mask = env_ids == episode['env_id']
    
    traj_2d = features_2d[mask]
    traj_timesteps = timesteps[mask]
    
    # 创建动画
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制所有轨迹作为背景
    ax.scatter(features_2d[:, 0], features_2d[:, 1], 
              c='lightgray', alpha=0.3, s=5)
    
    # 当前轨迹
    line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.6)
    point, = ax.plot([], [], 'ro', markersize=10)
    
    ax.set_xlim(features_2d[:, 0].min() - 5, features_2d[:, 0].max() + 5)
    ax.set_ylim(features_2d[:, 1].min() - 5, features_2d[:, 1].max() + 5)
    ax.set_title(f'Episode {episode_id} Trajectory (env_id={episode["env_id"]})')
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
    
    def animate(i):
        line.set_data(traj_2d[:i+1, 0], traj_2d[:i+1, 1])
        point.set_data([traj_2d[i, 0]], [traj_2d[i, 1]])
        ax.set_title(f'Episode {episode_id} - Step {traj_timesteps[i]}')
        return line, point
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(traj_2d), interval=100, blit=True)
    
    anim.save(output, writer='pillow', fps=10)
    print(f"✓ Animation saved to {output}")

if __name__ == '__main__':
    main()
