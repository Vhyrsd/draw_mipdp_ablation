"""
用于可视化保存的特征
Usage:
python visualize_features.py --feature_dir data/pusht_eval_output/features --output tsne_vis.png
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import click
from pathlib import Path


def load_features(feature_path):
    """加载保存的特征数据"""
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\n=== Loading Features ===")
    print(f"Total episodes: {len(data['episodes'])}")

    all_features = []
    all_env_ids = []
    all_timesteps = []
    all_prefixes = []

    # 先检查数据结构
    if len(data['episodes']) == 0:
        raise ValueError("No episodes found in the data!")

    # 检查第一个episode的结构
    first_ep = data['episodes'][0]
    print(f"First episode structure:")
    print(f"  env_id: {first_ep['env_id']}")
    print(f"  env_prefix: {first_ep['env_prefix']}")
    print(f"  timesteps: {len(first_ep['timesteps'])}")
    print(f"  features type: {type(first_ep['features'])}")

    if isinstance(first_ep['features'], np.ndarray):
        print(f"  features shape: {first_ep['features'].shape}")
    elif isinstance(first_ep['features'], dict):
        print(f"  features keys: {list(first_ep['features'].keys())}")
        for k, v in first_ep['features'].items():
            print(
                f"    {k}: {v.shape if isinstance(v, np.ndarray) else type(v)}")

    # 处理每个episode
    for ep_idx, episode in enumerate(data['episodes']):
        env_id = episode['env_id']
        prefix = episode['env_prefix']
        features = episode['features']
        timesteps = episode['timesteps']

        # 检查特征是否为空
        if isinstance(features, np.ndarray):
            if features.size == 0:
                print(
                    f"Warning: Episode {ep_idx} (env {env_id}) has empty features, skipping")
                continue

            # 如果是多维数组，展平
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            elif len(features.shape) == 1:
                features = features.reshape(1, -1)

            all_features.append(features)
            all_env_ids.extend([env_id] * len(features))
            all_timesteps.extend(timesteps)
            all_prefixes.extend([prefix] * len(features))

        elif isinstance(features, dict):
            # 如果特征是字典，只取第一个特征
            if len(features) == 0:
                print(
                    f"Warning: Episode {ep_idx} (env {env_id}) has empty feature dict, skipping")
                continue

            key = list(features.keys())[0]
            feat_array = features[key]

            if feat_array.size == 0:
                print(
                    f"Warning: Episode {ep_idx} (env {env_id}) has empty features[{key}], skipping")
                continue

            # 展平特征
            if len(feat_array.shape) > 2:
                feat_array = feat_array.reshape(feat_array.shape[0], -1)
            elif len(feat_array.shape) == 1:
                feat_array = feat_array.reshape(1, -1)

            all_features.append(feat_array)
            all_env_ids.extend([env_id] * len(feat_array))
            all_timesteps.extend(timesteps)
            all_prefixes.extend([prefix] * len(feat_array))
        else:
            print(
                f"Warning: Episode {ep_idx} (env {env_id}) has unknown feature type: {type(features)}")
            continue

    # 检查是否有有效特征
    if len(all_features) == 0:
        raise ValueError(
            "No valid features found! Check if feature extraction was successful.")

    print(f"=== Feature Statistics ===")
    print(f"Valid episodes: {len(all_features)}")
    print(f"Feature shapes: {[f.shape for f in all_features[:3]]}...")  # 显示前3个

    # 合并所有特征
    all_features = np.vstack(all_features)
    all_env_ids = np.array(all_env_ids)
    all_timesteps = np.array(all_timesteps)
    all_prefixes = np.array(all_prefixes)

    print(f"Total samples: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Unique environments: {len(np.unique(all_env_ids))}")
    print(f"Prefixes: {np.unique(all_prefixes)}")

    return all_features, all_env_ids, all_timesteps, all_prefixes


@click.command()
@click.option('--feature_dir', required=True, help='Directory containing eval_features.pkl')
@click.option('--output', default='tsne_visualization.png', help='Output image path')
@click.option('--perplexity', default=100, type=int, help='t-SNE perplexity')
@click.option('--n_samples', default=None, type=int, help='Subsample for faster computation')
def main(feature_dir, output, perplexity, n_samples):
    plt.rcParams.update({'font.family': 'Calibri'})
    feature_path = Path(feature_dir) / 'eval_features.pkl'

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    print(f"Loading features from {feature_path}")

    try:
        features, env_ids, timesteps, prefixes = load_features(feature_path)
    except Exception as e:
        print(f"\n❌ Error loading features: {e}")
        print("\nPlease check:")
        print("1. Was feature extraction successful during eval?")
        print("2. Does the policy have obs_encoder or vision_encoder?")
        print("3. Check the eval output for any warnings about feature extraction")
        raise

    # 子采样（如果数据太大）
    if n_samples is not None and len(features) > n_samples:
        print(f"Subsampling from {len(features)} to {n_samples} samples...")
        indices = np.random.choice(len(features), n_samples, replace=False)
        features = features[indices]
        env_ids = env_ids[indices]
        timesteps = timesteps[indices]
        prefixes = prefixes[indices]

    # 执行 t-SNE
    print(f"Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=perplexity, verbose=1)
    features_2d = tsne.fit_transform(features)

    print("\nCreating visualizations...")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 按环境ID着色
    scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1],
                               c=env_ids, cmap='tab20', alpha=0.6, s=10)
    axes[0].set_title('t-SNE colored by Environment ID')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter1, ax=axes[0], label='Environment ID')

    # 按时间步着色
    scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1],
                               c=timesteps, cmap='viridis', alpha=0.6, s=10)
    axes[1].set_title('t-SNE colored by Timestep')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter2, ax=axes[1], label='Timestep')

    # # 按train/test着色
    # is_train = np.array([p == 'train/' for p in prefixes])
    # scatter3 = axes[2].scatter(features_2d[:, 0], features_2d[:, 1],
    #                            c=is_train, cmap='RdYlBu', alpha=0.6, s=10)
    # axes[2].set_title('t-SNE colored by Train/Test')
    # axes[2].set_xlabel('t-SNE Dimension 1')
    # axes[2].set_ylabel('t-SNE Dimension 2')
    # cbar = plt.colorbar(scatter3, ax=axes[2])
    # cbar.set_ticks([0, 1])
    # cbar.set_ticklabels(['Test', 'Train'])

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output}")

    # 保存降维后的数据
    output_data = {
        'features_2d': features_2d,
        'env_ids': env_ids,
        'timesteps': timesteps,
        'prefixes': prefixes
    }
    output_pkl = Path(output).with_suffix('.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"✓ t-SNE results saved to {output_pkl}")


if __name__ == '__main__':
    main()
