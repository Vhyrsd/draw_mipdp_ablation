"""
增强版可视化脚本
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import click
from pathlib import Path

@click.command()
@click.option('--feature_dir', required=True)
@click.option('--output_dir', default='tsne_analysis')
def main(feature_dir, output_dir):
    feature_path = Path(feature_dir) / 'eval_features.pkl'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    # 提取特征
    all_features = []
    all_env_ids = []
    all_timesteps = []
    all_prefixes = []
    all_rewards = []
    
    for episode in data['episodes']:
        features = episode['features']
        if isinstance(features, np.ndarray) and features.size > 0:
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            all_features.append(features)
            all_env_ids.extend([episode['env_id']] * len(features))
            all_timesteps.extend(episode['timesteps'])
            all_prefixes.extend([episode['env_prefix']] * len(features))
            all_rewards.extend(episode['rewards'])
    
    all_features = np.vstack(all_features)
    all_env_ids = np.array(all_env_ids)
    all_timesteps = np.array(all_timesteps)
    all_prefixes = np.array(all_prefixes)
    all_rewards = np.array(all_rewards)
    
    print(f"Total samples: {len(all_features)}")
    
    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, verbose=1)
    features_2d = tsne.fit_transform(all_features)
    
    # 1. 按成功/失败着色
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 计算每个环境的总奖励
    env_total_rewards = {}
    for ep in data['episodes']:
        env_total_rewards[ep['env_id']] = ep['rewards'].sum()
    
    # 判断成功/失败（假设奖励>某个阈值为成功）
    reward_threshold = np.median(list(env_total_rewards.values()))
    is_success = np.array([env_total_rewards[eid] > reward_threshold for eid in all_env_ids])
    
    scatter = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                 c=is_success, cmap='RdYlGn', alpha=0.6, s=10)
    axes[0, 0].set_title('Success vs Failure (by total reward)')
    plt.colorbar(scatter, ax=axes[0, 0], label='Success')
    
    # 2. 按累积奖励着色
    cumulative_rewards = []
    for ep in data['episodes']:
        cum_rew = np.cumsum(ep['rewards'])
        cumulative_rewards.extend(cum_rew)
    cumulative_rewards = np.array(cumulative_rewards)
    
    scatter = axes[0, 1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                 c=cumulative_rewards, cmap='viridis', alpha=0.6, s=10)
    axes[0, 1].set_title('Cumulative Reward')
    plt.colorbar(scatter, ax=axes[0, 1], label='Cumulative Reward')
    
    # 3. 显示几条代表性轨迹
    axes[1, 0].set_title('Sample Trajectories')
    
    # 选择几个环境绘制完整轨迹
    sample_envs = np.random.choice(np.unique(all_env_ids), size=min(10, len(np.unique(all_env_ids))), replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sample_envs)))
    
    for i, env_id in enumerate(sample_envs):
        mask = all_env_ids == env_id
        traj = features_2d[mask]
        axes[1, 0].plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i], 
                       alpha=0.7, markersize=3, linewidth=1, label=f'Env {env_id}')
        # 标记起点和终点
        axes[1, 0].scatter(traj[0, 0], traj[0, 1], color=colors[i], 
                          s=100, marker='*', edgecolors='black', linewidths=1)
        axes[1, 0].scatter(traj[-1, 0], traj[-1, 1], color=colors[i], 
                          s=100, marker='s', edgecolors='black', linewidths=1)
    
    axes[1, 0].legend(fontsize=8, loc='best')
    
    # 4. 训练集 vs 测试集的密度对比
    from scipy.stats import gaussian_kde
    
    is_train = np.array([p == 'train/' for p in all_prefixes])
    train_points = features_2d[is_train]
    test_points = features_2d[~is_train]
    
    # 绘制密度等高线
    if len(train_points) > 0:
        try:
            kde_train = gaussian_kde(train_points.T)
            x_min, x_max = features_2d[:, 0].min(), features_2d[:, 0].max()
            y_min, y_max = features_2d[:, 1].min(), features_2d[:, 1].max()
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density_train = np.reshape(kde_train(positions).T, xx.shape)
            axes[1, 1].contour(xx, yy, density_train, colors='red', alpha=0.5, linewidths=2)
        except:
            pass
    
    if len(test_points) > 0:
        try:
            kde_test = gaussian_kde(test_points.T)
            density_test = np.reshape(kde_test(positions).T, xx.shape)
            axes[1, 1].contour(xx, yy, density_test, colors='blue', alpha=0.5, linewidths=2)
        except:
            pass
    
    axes[1, 1].scatter(train_points[:, 0], train_points[:, 1], 
                      c='red', alpha=0.3, s=5, label='Train')
    axes[1, 1].scatter(test_points[:, 0], test_points[:, 1], 
                      c='blue', alpha=0.3, s=5, label='Test')
    axes[1, 1].set_title('Train vs Test Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tsne_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_dir / 'tsne_detailed_analysis.png'}")
    
    # 5. 统计分析
    print("=== Statistical Analysis ===")
    print(f"Train samples: {is_train.sum()}")
    print(f"Test samples: {(~is_train).sum()}")
    print(f"Unique environments: {len(np.unique(all_env_ids))}")
    print(f"Average trajectory length: {len(all_features) / len(data['episodes']):.1f}")
    
    # 计算训练集和测试集的特征距离
    from scipy.spatial.distance import cdist
    if len(train_points) > 0 and len(test_points) > 0:
        # 随机采样以加速计算
        n_samples = min(1000, len(train_points), len(test_points))
        train_sample = train_points[np.random.choice(len(train_points), n_samples, replace=False)]
        test_sample = test_points[np.random.choice(len(test_points), n_samples, replace=False)]
        
        distances = cdist(test_sample, train_sample, metric='euclidean')
        min_distances = distances.min(axis=1)
        
        print(f"\nTest-to-Train distances:")
        print(f"  Mean: {min_distances.mean():.2f}")
        print(f"  Median: {np.median(min_distances):.2f}")
        print(f"  Std: {min_distances.std():.2f}")
    
    plt.show()

if __name__ == '__main__':
    main()
