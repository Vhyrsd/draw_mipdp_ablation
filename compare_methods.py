"""
对比两个方法的特征质量
Usage:
python compare_methods.py \
    --baseline_dir feats_dir_baseline \
    --ours_dir feats_dir_ours \
    --output comparison_report.png
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import click
from pathlib import Path


def load_and_process_features(feature_dir):
    """加载并处理特征"""
    feature_path = Path(feature_dir) / 'eval_features.pkl'

    with open(feature_path, 'rb') as f:
        data = pickle.load(f)

    all_features = []
    all_env_ids = []
    all_timesteps = []
    all_prefixes = []

    for episode in data['episodes']:
        features = episode['features']
        if isinstance(features, np.ndarray) and features.size > 0:
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            all_features.append(features)
            all_env_ids.extend([episode['env_id']] * len(features))
            all_timesteps.extend(episode['timesteps'])
            all_prefixes.extend([episode['env_prefix']] * len(features))

    all_features = np.vstack(all_features)
    all_env_ids = np.array(all_env_ids)
    all_timesteps = np.array(all_timesteps)
    all_prefixes = np.array(all_prefixes)

    return all_features, all_env_ids, all_timesteps, all_prefixes, data


def compute_metrics(features, env_ids, prefixes):
    """计算特征质量指标"""
    metrics = {}

    # 1. Silhouette Score (越高越好，范围 [-1, 1])
    # 衡量同一类内部的紧密度和不同类之间的分离度
    try:
        metrics['silhouette_env'] = silhouette_score(
            features, env_ids, sample_size=min(10000, len(features)))
    except:
        metrics['silhouette_env'] = np.nan

    # 2. Davies-Bouldin Index (越低越好)
    # 衡量簇内相似度与簇间差异度的比值
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(features, env_ids)
    except:
        metrics['davies_bouldin'] = np.nan

    # 3. Calinski-Harabasz Score (越高越好)
    # 衡量簇间方差与簇内方差的比值
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(
            features, env_ids)
    except:
        metrics['calinski_harabasz'] = np.nan

    # 4. 轨迹平滑度 (越低越好)
    # 计算相邻时间步特征变化的平均值
    unique_envs = np.unique(env_ids)
    smoothness_scores = []
    for env_id in unique_envs:
        mask = env_ids == env_id
        env_features = features[mask]
        if len(env_features) > 1:
            changes = np.linalg.norm(np.diff(env_features, axis=0), axis=1)
            smoothness_scores.append(changes.mean())
    metrics['trajectory_smoothness'] = np.mean(smoothness_scores)
    metrics['trajectory_smoothness_std'] = np.std(smoothness_scores)

    # 5. Train-Test 分离度
    is_train = np.array([p == 'train/' for p in prefixes])
    train_features = features[is_train]
    test_features = features[~is_train]

    if len(train_features) > 0 and len(test_features) > 0:
        # 计算测试集到训练集的最小距离
        n_samples = min(1000, len(train_features), len(test_features))
        train_sample = train_features[np.random.choice(
            len(train_features), n_samples, replace=False)]
        test_sample = test_features[np.random.choice(
            len(test_features), n_samples, replace=False)]

        distances = cdist(test_sample, train_sample, metric='euclidean')
        min_distances = distances.min(axis=1)

        metrics['test_to_train_dist_mean'] = min_distances.mean()
        metrics['test_to_train_dist_std'] = min_distances.std()

        # 计算重叠度（距离小于阈值的比例）
        threshold = np.percentile(min_distances, 25)
        metrics['train_test_overlap'] = (min_distances < threshold).mean()

    # 6. 特征方差（表示特征的表达能力）
    metrics['feature_variance'] = np.var(features, axis=0).mean()

    # 7. 环境间分离度
    # 计算不同环境中心之间的平均距离
    env_centers = []
    for env_id in unique_envs:
        mask = env_ids == env_id
        env_centers.append(features[mask].mean(axis=0))
    env_centers = np.array(env_centers)

    if len(env_centers) > 1:
        inter_env_distances = cdist(
            env_centers, env_centers, metric='euclidean')
        # 只取上三角（不包括对角线）
        mask = np.triu(np.ones_like(inter_env_distances, dtype=bool), k=1)
        metrics['inter_env_distance'] = inter_env_distances[mask].mean()

    return metrics


def compute_trajectory_metrics(data):
    """计算轨迹级别的指标"""
    metrics = {}

    trajectory_lengths = []
    trajectory_variances = []

    for episode in data['episodes']:
        features = episode['features']
        if isinstance(features, np.ndarray) and features.size > 0:
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)

            trajectory_lengths.append(len(features))

            # 计算轨迹的方差（表示轨迹的探索范围）
            if len(features) > 1:
                trajectory_variances.append(np.var(features, axis=0).mean())

    metrics['avg_trajectory_length'] = np.mean(trajectory_lengths)
    metrics['avg_trajectory_variance'] = np.mean(trajectory_variances)

    return metrics


@click.command()
@click.option('--baseline_dir', required=True, help='Baseline feature directory')
@click.option('--ours_dir', required=True, help='Ours feature directory')
@click.option('--output', default='comparison_report.png', help='Output image path')
def main(baseline_dir, ours_dir, output):
    print("Loading baseline features...")
    baseline_features, baseline_env_ids, baseline_timesteps, baseline_prefixes, baseline_data = \
        load_and_process_features(baseline_dir)

    print("Loading ours features...")
    ours_features, ours_env_ids, ours_timesteps, ours_prefixes, ours_data = \
        load_and_process_features(ours_dir)

    print("\nComputing metrics...")
    baseline_metrics = compute_metrics(
        baseline_features, baseline_env_ids, baseline_prefixes)
    ours_metrics = compute_metrics(ours_features, ours_env_ids, ours_prefixes)

    baseline_traj_metrics = compute_trajectory_metrics(baseline_data)
    ours_traj_metrics = compute_trajectory_metrics(ours_data)

    # 合并指标
    baseline_metrics.update(baseline_traj_metrics)
    ours_metrics.update(ours_traj_metrics)

    # 打印对比
    print("" + "="*60)
    print("QUANTITATIVE COMPARISON")
    print("="*60)

    print(f"\n{'Metric':<35} {'Baseline':<15} {'Ours':<15} {'Improvement':<15}")
    print("-"*80)

    for key in baseline_metrics.keys():
        baseline_val = baseline_metrics[key]
        ours_val = ours_metrics[key]

        if not np.isnan(baseline_val) and not np.isnan(ours_val):
            # 对于某些指标，越低越好
            if key in ['davies_bouldin', 'trajectory_smoothness', 'test_to_train_dist_mean']:
                improvement = (baseline_val - ours_val) / baseline_val * 100
                better = "✓" if improvement > 0 else "✗"
            else:
                improvement = (ours_val - baseline_val) / \
                    abs(baseline_val) * 100
                better = "✓" if improvement > 0 else "✗"

            print(
                f"{key:<35} {baseline_val:<15.4f} {ours_val:<15.4f} {better} {improvement:>+6.2f}%")

    # 可视化对比
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. 主要指标对比（柱状图）
    ax1 = fig.add_subplot(gs[0, :2])

    key_metrics = ['silhouette_env', 'calinski_harabasz', 'inter_env_distance']
    key_labels = ['Silhouette Score\n(higher better)',
                  'Calinski-Harabasz\n(higher better)',
                  'Inter-Env Distance\n(higher better)']

    x = np.arange(len(key_metrics))
    width = 0.35

    baseline_vals = [baseline_metrics[k] for k in key_metrics]
    ours_vals = [ours_metrics[k] for k in key_metrics]

    # 归一化以便比较
    baseline_vals_norm = []
    ours_vals_norm = []
    for b, o in zip(baseline_vals, ours_vals):
        max_val = max(abs(b), abs(o))
        if max_val > 0:
            baseline_vals_norm.append(b / max_val)
            ours_vals_norm.append(o / max_val)
        else:
            baseline_vals_norm.append(0)
            ours_vals_norm.append(0)

    bars1 = ax1.bar(x - width/2, baseline_vals_norm,
                    width, label='Baseline', alpha=0.8)
    bars2 = ax1.bar(x + width/2, ours_vals_norm, width,
                    label='Ours (MI-enhanced)', alpha=0.8)

    ax1.set_ylabel('Normalized Score')
    ax1.set_title('Key Metrics Comparison (Normalized)',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(key_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. 轨迹平滑度对比
    ax2 = fig.add_subplot(gs[0, 2:])

    smoothness_metrics = ['trajectory_smoothness', 'davies_bouldin']
    smoothness_labels = [
        'Trajectory Smoothness\n(lower better)', 'Davies-Bouldin Index\n(lower better)']

    x2 = np.arange(len(smoothness_metrics))
    baseline_smooth = [baseline_metrics[k] for k in smoothness_metrics]
    ours_smooth = [ours_metrics[k] for k in smoothness_metrics]

    # 归一化
    baseline_smooth_norm = []
    ours_smooth_norm = []
    for b, o in zip(baseline_smooth, ours_smooth):
        max_val = max(abs(b), abs(o))
        if max_val > 0:
            baseline_smooth_norm.append(b / max_val)
            ours_smooth_norm.append(o / max_val)
        else:
            baseline_smooth_norm.append(0)
            ours_smooth_norm.append(0)

    bars1 = ax2.bar(x2 - width/2, baseline_smooth_norm, width,
                    label='Baseline', alpha=0.8, color='coral')
    bars2 = ax2.bar(x2 + width/2, ours_smooth_norm, width,
                    label='Ours (MI-enhanced)', alpha=0.8, color='skyblue')

    ax2.set_ylabel('Normalized Score')
    ax2.set_title('Smoothness Metrics (Lower is Better)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(smoothness_labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3-4. t-SNE 可视化对比
    print("Running t-SNE for visualization...")

    # Baseline t-SNE
    ax3 = fig.add_subplot(gs[1, :2])
    tsne_baseline = TSNE(n_components=2, random_state=42, perplexity=100)
    baseline_2d = tsne_baseline.fit_transform(
        baseline_features[:5000])  # 子采样加速
    scatter1 = ax3.scatter(baseline_2d[:, 0], baseline_2d[:, 1],
                           c=baseline_env_ids[:5000], cmap='tab20', alpha=0.6, s=10)
    ax3.set_title('Baseline: t-SNE by Environment ID',
                  fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=ax3, label='Environment ID')

    # Ours t-SNE
    ax4 = fig.add_subplot(gs[1, 2:])
    tsne_ours = TSNE(n_components=2, random_state=42, perplexity=100)
    ours_2d = tsne_ours.fit_transform(ours_features[:5000])
    scatter2 = ax4.scatter(ours_2d[:, 0], ours_2d[:, 1],
                           c=ours_env_ids[:5000], cmap='tab20', alpha=0.6, s=10)
    ax4.set_title('Ours (MI-enhanced): t-SNE by Environment ID',
                  fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=ax4, label='Environment ID')

    # 5. 轨迹示例对比
    ax5 = fig.add_subplot(gs[2, :2])
    ax6 = fig.add_subplot(gs[2, 2:])

    # 选择相同的环境进行对比
    sample_env = 0

    # Baseline轨迹
    mask_baseline = baseline_env_ids[:5000] == sample_env
    if mask_baseline.sum() > 0:
        traj_baseline = baseline_2d[mask_baseline]
        ax5.plot(traj_baseline[:, 0], traj_baseline[:, 1], 'o-',
                 alpha=0.7, markersize=5, linewidth=2, color='coral')
        ax5.scatter(traj_baseline[0, 0], traj_baseline[0, 1],
                    s=200, marker='*', color='green', edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax5.scatter(traj_baseline[-1, 0], traj_baseline[-1, 1],
                    s=200, marker='s', color='red', edgecolors='black', linewidths=2, label='End', zorder=5)
    ax5.set_title(
        f'Baseline: Trajectory Example (Env {sample_env})', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Ours轨迹
    mask_ours = ours_env_ids[:5000] == sample_env
    if mask_ours.sum() > 0:
        traj_ours = ours_2d[mask_ours]
        ax6.plot(traj_ours[:, 0], traj_ours[:, 1], 'o-',
                 alpha=0.7, markersize=5, linewidth=2, color='skyblue')
        ax6.scatter(traj_ours[0, 0], traj_ours[0, 1],
                    s=200, marker='*', color='green', edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax6.scatter(traj_ours[-1, 0], traj_ours[-1, 1],
                    s=200, marker='s', color='red', edgecolors='black', linewidths=2, label='End', zorder=5)
    ax6.set_title(
        f'Ours (MI-enhanced): Trajectory Example (Env {sample_env})', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Baseline vs Ours (MI-enhanced): Feature Quality Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison report saved to {output}")

    # 保存数值结果
    output_txt = Path(output).with_suffix('.txt')
    with open(output_txt, 'w') as f:
        f.write("="*60 + "\n")
        f.write("QUANTITATIVE COMPARISON REPORT\n")
        f.write("="*60 + "\n")

        f.write(
            f"{'Metric':<35} {'Baseline':<15} {'Ours':<15} {'Improvement':<15}\n")
        f.write("-"*80 + "\n")

        for key in baseline_metrics.keys():
            baseline_val = baseline_metrics[key]
            ours_val = ours_metrics[key]

            if not np.isnan(baseline_val) and not np.isnan(ours_val):
                if key in ['davies_bouldin', 'trajectory_smoothness', 'test_to_train_dist_mean']:
                    improvement = (baseline_val - ours_val) / \
                        baseline_val * 100
                    better = "G" if improvement > 0 else "B"
                else:
                    improvement = (ours_val - baseline_val) / \
                        abs(baseline_val) * 100
                    better = "G" if improvement > 0 else "B"
                f.write(
                    f"{key:<35} {baseline_val:<15.4f} {ours_val:<15.4f} {better} {improvement:>+6.2f}%\n")

    print(f"✓ Numerical results saved to {output_txt}")


if __name__ == '__main__':
    main()
