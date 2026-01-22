"""
分析轨迹的时间动态
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_trajectory_dynamics(feature_dir):
    feature_path = Path(feature_dir) / 'eval_features.pkl'
    
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 选择几个代表性的episode
    sample_episodes = data['episodes'][:6]
    
    for idx, (ax, episode) in enumerate(zip(axes.flat, sample_episodes)):
        features = episode['features']
        if features.size == 0:
            continue
        
        # 计算特征的变化率
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # 计算相邻时间步的特征距离
        feature_changes = np.linalg.norm(np.diff(features, axis=0), axis=1)
        
        # 绘制
        timesteps = episode['timesteps'][1:]  # 去掉第一个
        rewards = episode['rewards'][1:]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(timesteps, feature_changes, 'b-', label='Feature Change', linewidth=2)
        line2 = ax2.plot(timesteps, rewards, 'r--', label='Reward', linewidth=2)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Feature Change', color='b')
        ax2.set_ylabel('Reward', color='r')
        ax.set_title(f'Env {episode["env_id"]} ({episode["env_prefix"]})')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trajectory_dynamics.png', dpi=300)
    print("✓ Saved trajectory_dynamics.png")

if __name__ == '__main__':
    analyze_trajectory_dynamics('feats_dir_dp')
