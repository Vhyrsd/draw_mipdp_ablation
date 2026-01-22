"""
比较成功和失败的episode
Usage:
python analyze_success_failure.py --feature_dir feats_dir
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    feature_path = Path('feats_dir_dp') / 'eval_features.pkl'
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    tsne_path = Path('tsne_visualization.pkl')
    with open(tsne_path, 'rb') as f:
        tsne_data = pickle.load(f)
    
    features_2d = tsne_data['features_2d']
    env_ids = tsne_data['env_ids']
    
    # 根据总奖励分类成功/失败
    episode_rewards = []
    for ep in data['episodes']:
        total_reward = ep['rewards'].sum()
        episode_rewards.append((ep['env_id'], total_reward))
    
    episode_rewards.sort(key=lambda x: x[1])
    
    # 选择最好和最差的episodes
    n_show = 5
    worst_ids = [x[0] for x in episode_rewards[:n_show]]
    best_ids = [x[0] for x in episode_rewards[-n_show:]]
    
    print(f"Worst episodes: {worst_ids}")
    print(f"Best episodes: {best_ids}")
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 背景
    ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
               c='lightgray', alpha=0.2, s=5)
    ax2.scatter(features_2d[:, 0], features_2d[:, 1], 
               c='lightgray', alpha=0.2, s=5)
    
    # 最差的episodes
    for env_id in worst_ids:
        mask = env_ids == env_id
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   alpha=0.6, s=20, label=f'Env {env_id}')
    
    # 最好的episodes
    for env_id in best_ids:
        mask = env_ids == env_id
        ax2.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   alpha=0.6, s=20, label=f'Env {env_id}')
    
    ax1.set_title('Worst Episodes')
    ax1.legend()
    ax2.set_title('Best Episodes')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('success_failure_comparison.png', dpi=300)
    print("✓ Saved to success_failure_comparison.png")

if __name__ == '__main__':
    main()
