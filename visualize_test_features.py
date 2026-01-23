"""
可视化测试集特征（修正colorbar显示0-49）
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import click
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm


def generate_distinct_colors(n_colors):
    """
    生成n个尽可能区分的颜色
    """
    if n_colors <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_colors]
    elif n_colors <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_colors]
    else:
        # 组合多个colormap
        colors_list = []

        # 1. tab20: 20种颜色
        colors_list.append(plt.cm.tab20(np.linspace(0, 1, 20)))

        # 2. tab20b: 20种颜色
        colors_list.append(plt.cm.tab20b(np.linspace(0, 1, 20)))

        # 3. tab20c: 20种颜色
        colors_list.append(plt.cm.tab20c(np.linspace(0, 1, 20)))

        # 4. Set3: 12种颜色
        colors_list.append(plt.cm.Set3(np.linspace(0, 1, 12)))

        # 5. Paired: 12种颜色
        colors_list.append(plt.cm.Paired(np.linspace(0, 1, 12)))

        # 6. Set1: 9种颜色
        colors_list.append(plt.cm.Set1(np.linspace(0, 1, 9)))

        # 7. Set2: 8种颜色
        colors_list.append(plt.cm.Set2(np.linspace(0, 1, 8)))

        # 8. Dark2: 8种颜色
        colors_list.append(plt.cm.Dark2(np.linspace(0, 1, 8)))

        # 9. Pastel1: 9种颜色
        colors_list.append(plt.cm.Pastel1(np.linspace(0, 1, 9)))

        # 10. Pastel2: 8种颜色
        colors_list.append(plt.cm.Pastel2(np.linspace(0, 1, 8)))

        # 合并所有颜色（总共约114种）
        colors = np.vstack(colors_list)[:n_colors]

    return colors


def load_features(feature_path):
    """加载保存的特征数据，只保留测试集"""
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\n=== Loading Features (Test Only) ===")
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

    # 处理每个episode，只保留测试集
    test_count = 0
    train_count = 0

    for ep_idx, episode in enumerate(data['episodes']):
        env_id = episode['env_id']
        prefix = episode['env_prefix']
        features = episode['features']
        timesteps = episode['timesteps']

        # 只保留测试集
        if prefix != 'test/':
            train_count += 1
            continue

        test_count += 1

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
            "No valid test features found! Check if feature extraction was successful.")

    print(f"=== Feature Statistics ===")
    print(f"Train episodes (skipped): {train_count}")
    print(f"Test episodes (kept): {test_count}")
    print(f"Valid test episodes with features: {len(all_features)}")
    print(f"Feature shapes: {[f.shape for f in all_features[:3]]}...")

    # 合并所有特征
    all_features = np.vstack(all_features)
    all_env_ids = np.array(all_env_ids)
    all_timesteps = np.array(all_timesteps)
    all_prefixes = np.array(all_prefixes)

    print(f"Total test samples: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Unique test environments: {len(np.unique(all_env_ids))}")
    print(f"Test environment IDs: {sorted(np.unique(all_env_ids))}")

    return all_features, all_env_ids, all_timesteps, all_prefixes


@click.command()
@click.option('--feature_dir', required=True, help='Directory containing eval_features.pkl')
@click.option('--output', default='tsne_test_only.png', help='Output image path')
@click.option('--perplexity', default=50, type=int, help='t-SNE perplexity')
@click.option('--n_samples', default=None, type=int, help='Subsample for faster computation')
@click.option('--colorbar_fontsize', default=8, type=int, help='Colorbar tick label font size')
@click.option('--show_env_id', is_flag=True, help='Show actual environment IDs instead of 0-49 indices')
def main(feature_dir, output, perplexity, n_samples, colorbar_fontsize, show_env_id):
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

    # 可视化 - 只有两个子图
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # 1. 按环境ID着色（离散颜色）
    unique_envs = np.unique(env_ids)
    n_envs = len(unique_envs)

    print(f"Generating {n_envs} distinct colors...")

    # 创建环境ID到连续索引的映射（从0开始）
    env_id_to_idx = {env_id: idx for idx,
                     env_id in enumerate(sorted(unique_envs))}
    env_indices = np.array([env_id_to_idx[env_id] for env_id in env_ids])

    # 生成足够多的离散颜色
    colors = generate_distinct_colors(n_envs)

    # 创建离散colormap
    cmap_discrete = ListedColormap(colors)

    # 使用离散的边界
    bounds = np.arange(n_envs + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap_discrete.N)

    scatter1 = axes[0].scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=env_indices,  # 使用从0开始的索引
        cmap=cmap_discrete,
        norm=norm,
        alpha=0.6,
        s=20
    )
    axes[0].set_title(f't-SNE colored by Environment Index\n(Test Set Only, {n_envs} environments)',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)

    # 创建colorbar - 关键修改：显示0-49的索引
    if n_envs <= 20:
        # 显示所有索引（0到n_envs-1）
        cbar1 = plt.colorbar(scatter1, ax=axes[0], ticks=np.arange(n_envs))
        if show_env_id:
            # 如果用户想看实际环境ID
            cbar1.set_label('Environment ID', fontsize=11)
            cbar1.ax.set_yticklabels([str(env_id) for env_id in sorted(unique_envs)],
                                     fontsize=colorbar_fontsize)
        else:
            # 默认显示0-49的索引
            cbar1.set_label('Environment Index (0-based)', fontsize=11)
            cbar1.ax.set_yticklabels([str(i) for i in range(n_envs)],
                                     fontsize=colorbar_fontsize)
    else:
        # 只显示部分刻度（每隔几个显示一个）
        tick_interval = max(1, n_envs // 20)  # 最多显示20个刻度
        tick_positions = np.arange(0, n_envs, tick_interval)

        cbar1 = plt.colorbar(scatter1, ax=axes[0], ticks=tick_positions)

        if show_env_id:
            # 显示实际环境ID
            cbar1.set_label('Environment ID', fontsize=11)
            cbar1.ax.set_yticklabels([str(sorted(unique_envs)[i]) for i in tick_positions],
                                     fontsize=colorbar_fontsize)
        else:
            # 显示0-49的索引
            cbar1.set_label('Environment Index (0-based)', fontsize=11)
            cbar1.ax.set_yticklabels([str(i) for i in tick_positions],
                                     fontsize=colorbar_fontsize)

        # 添加说明文字
        if show_env_id:
            info_text = f'Total: {n_envs} environments\nShowing every {tick_interval}th ID'
        else:
            info_text = f'Total: {n_envs} environments (Index 0-{n_envs-1})\nShowing every {tick_interval}th index'

        axes[0].text(0.02, 0.98, info_text,
                     transform=axes[0].transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                     fontsize=9)

    # 2. 按时间步着色（连续渐变）
    scatter2 = axes[1].scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=timesteps,
        cmap='viridis',  # 使用连续的渐变colormap
        alpha=0.6,
        s=20
    )
    axes[1].set_title(f't-SNE colored by Timestep\n(Test Set Only)',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)

    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Timestep', fontsize=11)

    # 添加网格
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output}")

    # 保存降维后的数据
    output_data = {
        'features_2d': features_2d,
        'env_ids': env_ids,
        'env_indices': env_indices,
        'timesteps': timesteps,
        'prefixes': prefixes,
        'n_test_envs': n_envs,
        'test_env_ids': sorted(unique_envs),
        'env_id_to_idx': env_id_to_idx
    }

    output_pkl = Path(output).with_suffix('.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"✓ t-SNE results saved to {output_pkl}")

    # 打印统计信息
    print(f"=== Visualization Summary ===")
    print(f"Total test samples visualized: {len(features_2d)}")
    print(f"Number of test environments: {n_envs}")
    print(f"Test environment IDs: {sorted(unique_envs)}")
    print(f"Environment Index range: [0, {n_envs-1}]")
    print(f"Timestep range: [{timesteps.min()}, {timesteps.max()}]")
    print(f"Average samples per environment: {len(features_2d) / n_envs:.1f}")

    # 保存颜色映射信息
    color_map_file = Path(output).parent / 'env_color_mapping.txt'
    with open(color_map_file, 'w') as f:
        f.write("Environment Index to ID Mapping")
        f.write("=" * 60 + "\n")
        f.write(f"{'Index':<10} {'Env ID':<10} {'RGB Color':<30}\n")
        f.write("-" * 60 + "\n")
        for env_id in sorted(unique_envs):
            idx = env_id_to_idx[env_id]
            color = colors[idx]
            f.write(
                f"{idx:<10} {env_id:<10} RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})\n")
    print(f"✓ Color mapping saved to {color_map_file}")


if __name__ == '__main__':
    main()
