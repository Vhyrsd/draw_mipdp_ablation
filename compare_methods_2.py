"""
计算更适合轨迹数据的指标
"""
import click
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

from compare_methods import load_and_process_features

def compute_trajectory_specific_metrics(features, env_ids, timesteps, data):
    """
    计算专门针对轨迹数据的指标
    """
    metrics = {}
    
    # 1. 轨迹内部一致性（Intra-trajectory Consistency）
    # 同一轨迹的相邻时间步应该相似
    intra_traj_distances = []
    for episode in data['episodes']:
        ep_features = episode['features']
        if len(ep_features) > 1:
            # 计算相邻时间步的距离
            consecutive_dist = np.linalg.norm(
                np.diff(ep_features.reshape(len(ep_features), -1), axis=0), 
                axis=1
            )
            intra_traj_distances.extend(consecutive_dist)
    
    metrics['intra_trajectory_consistency'] = np.mean(intra_traj_distances)
    metrics['intra_trajectory_consistency_std'] = np.std(intra_traj_distances)
    
    # 2. 轨迹间可区分性（Inter-trajectory Discriminability）
    # 不同环境的轨迹应该可区分
    unique_envs = np.unique(env_ids)
    inter_traj_distances = []
    
    for i, env1 in enumerate(unique_envs[:10]):  # 采样以加速
        for env2 in unique_envs[i+1:10]:
            mask1 = env_ids == env1
            mask2 = env_ids == env2
            
            feat1 = features[mask1]
            feat2 = features[mask2]
            
            if len(feat1) > 0 and len(feat2) > 0:
                # 计算两个轨迹之间的最小距离
                dist = cdist(feat1[:10], feat2[:10], metric='euclidean').min()
                inter_traj_distances.append(dist)
    
    metrics['inter_trajectory_distance'] = np.mean(inter_traj_distances)
    
    # 3. 时间一致性（Temporal Consistency）
    # 相同时间步的不同环境应该有相似的特征模式
    unique_timesteps = np.unique(timesteps)
    temporal_variances = []
    
    for t in unique_timesteps[:20]:  # 采样
        mask = timesteps == t
        if mask.sum() > 1:
            t_features = features[mask]
            temporal_variances.append(np.var(t_features, axis=0).mean())
    
    metrics['temporal_consistency'] = np.mean(temporal_variances)
    
    # 4. 轨迹完整性（Trajectory Completeness）
    # 轨迹应该覆盖足够的特征空间
    trajectory_spans = []
    for episode in data['episodes']:
        ep_features = episode['features']
        if len(ep_features) > 1:
            # 计算轨迹的跨度（最大距离）
            ep_features_flat = ep_features.reshape(len(ep_features), -1)
            distances = pairwise_distances(ep_features_flat)
            trajectory_spans.append(distances.max())
    
    metrics['trajectory_span'] = np.mean(trajectory_spans)
    
    # 5. 特征利用率（Feature Utilization）
    # 特征维度应该被充分利用
    feature_std = np.std(features, axis=0)
    metrics['feature_utilization'] = (feature_std > 0.01).mean()  # 有效维度比例
    
    # 6. 轨迹平滑度的变异系数（Coefficient of Variation）
    # 衡量轨迹平滑度的稳定性
    smoothness_scores = []
    for episode in data['episodes']:
        ep_features = episode['features']
        if len(ep_features) > 1:
            changes = np.linalg.norm(
                np.diff(ep_features.reshape(len(ep_features), -1), axis=0), 
                axis=1
            )
            smoothness_scores.append(changes.mean())
    
    metrics['smoothness_cv'] = np.std(smoothness_scores) / np.mean(smoothness_scores)
    
    return metrics

def compute_generalization_metrics(features, env_ids, prefixes):
    """
    计算泛化相关的指标
    """
    metrics = {}
    
    is_train = np.array([p == 'train/' for p in prefixes])
    train_features = features[is_train]
    test_features = features[~is_train]
    
    if len(train_features) > 0 and len(test_features) > 0:
        # 1. 特征分布相似度（Feature Distribution Similarity）
        # 使用 Wasserstein 距离或 KL 散度
        from scipy.stats import wasserstein_distance
        
        # 对每个维度计算 Wasserstein 距离
        dim_distances = []
        for dim in range(min(10, features.shape[1])):  # 采样维度
            dist = wasserstein_distance(
                train_features[:, dim], 
                test_features[:, dim]
            )
            dim_distances.append(dist)
        
        metrics['train_test_distribution_distance'] = np.mean(dim_distances)
        
        # 2. 覆盖率（Coverage）
        # 测试集特征是否在训练集的覆盖范围内
        train_min = train_features.min(axis=0)
        train_max = train_features.max(axis=0)
        
        in_range = np.logical_and(
            test_features >= train_min,
            test_features <= train_max
        )
        metrics['test_coverage'] = in_range.mean()
        
        # 3. 最近邻距离比（Nearest Neighbor Distance Ratio）
        # 测试样本到最近训练样本的距离
        n_samples = min(500, len(train_features), len(test_features))
        train_sample = train_features[np.random.choice(len(train_features), n_samples, replace=False)]
        test_sample = test_features[np.random.choice(len(test_features), n_samples, replace=False)]
        
        distances = cdist(test_sample, train_sample, metric='euclidean')
        nn_distances = distances.min(axis=1)
        
        # 与训练集内部的最近邻距离比较
        train_distances = cdist(train_sample, train_sample, metric='euclidean')
        np.fill_diagonal(train_distances, np.inf)
        train_nn_distances = train_distances.min(axis=1)
        
        metrics['nn_distance_ratio'] = nn_distances.mean() / train_nn_distances.mean()
        
    return metrics

# 在 compare_methods.py 中添加这些指标
def enhanced_comparison(baseline_dir, ours_dir):
    """
    增强版对比，使用更相关的指标
    """
    # 加载数据
    baseline_features, baseline_env_ids, baseline_timesteps, baseline_prefixes, baseline_data = \
        load_and_process_features(baseline_dir)
    ours_features, ours_env_ids, ours_timesteps, ours_prefixes, ours_data = \
        load_and_process_features(ours_dir)
    
    # 计算轨迹特定指标
    print("\n=== Trajectory-Specific Metrics ===")
    baseline_traj_metrics = compute_trajectory_specific_metrics(
        baseline_features, baseline_env_ids, baseline_timesteps, baseline_data
    )
    ours_traj_metrics = compute_trajectory_specific_metrics(
        ours_features, ours_env_ids, ours_timesteps, ours_data
    )
    
    print(f"\n{'Metric':<40} {'Baseline':<15} {'Ours':<15} {'Better':<10}")
    print("-" * 80)
    
    # 轨迹内部一致性（越低越好 - 更平滑）
    print(f"{'Intra-Trajectory Consistency':<40} "
          f"{baseline_traj_metrics['intra_trajectory_consistency']:<15.4f} "
          f"{ours_traj_metrics['intra_trajectory_consistency']:<15.4f} "
          f"{'✓ Ours' if ours_traj_metrics['intra_trajectory_consistency'] < baseline_traj_metrics['intra_trajectory_consistency'] else '✓ Baseline'}")
    
    # 轨迹间可区分性（越高越好）
    print(f"{'Inter-Trajectory Distance':<40} "
          f"{baseline_traj_metrics['inter_trajectory_distance']:<15.4f} "
          f"{ours_traj_metrics['inter_trajectory_distance']:<15.4f} "
          f"{'✓ Ours' if ours_traj_metrics['inter_trajectory_distance'] > baseline_traj_metrics['inter_trajectory_distance'] else '✓ Baseline'}")
    
    # 特征利用率（越高越好）
    print(f"{'Feature Utilization':<40} "
          f"{baseline_traj_metrics['feature_utilization']:<15.4f} "
          f"{ours_traj_metrics['feature_utilization']:<15.4f} "
          f"{'✓ Ours' if ours_traj_metrics['feature_utilization'] > baseline_traj_metrics['feature_utilization'] else '✓ Baseline'}")
    
    # 轨迹跨度（越大越好 - 探索更充分）
    print(f"{'Trajectory Span':<40} "
          f"{baseline_traj_metrics['trajectory_span']:<15.4f} "
          f"{ours_traj_metrics['trajectory_span']:<15.4f} "
          f"{'✓ Ours' if ours_traj_metrics['trajectory_span'] > baseline_traj_metrics['trajectory_span'] else '✓ Baseline'}")
    
    # 计算泛化指标
    print("\n=== Generalization Metrics ===")
    baseline_gen_metrics = compute_generalization_metrics(
        baseline_features, baseline_env_ids, baseline_prefixes
    )
    ours_gen_metrics = compute_generalization_metrics(
        ours_features, ours_env_ids, ours_prefixes
    )
    
    # 训练-测试分布距离（越小越好 - 泛化更好）
    print(f"{'Train-Test Distribution Distance':<40} "
          f"{baseline_gen_metrics['train_test_distribution_distance']:<15.4f} "
          f"{ours_gen_metrics['train_test_distribution_distance']:<15.4f} "
          f"{'✓ Ours' if ours_gen_metrics['train_test_distribution_distance'] < baseline_gen_metrics['train_test_distribution_distance'] else '✓ Baseline'}")
    
    # 测试覆盖率（越高越好）
    print(f"{'Test Coverage':<40} "
          f"{baseline_gen_metrics['test_coverage']:<15.4f} "
          f"{ours_gen_metrics['test_coverage']:<15.4f} "
          f"{'✓ Ours' if ours_gen_metrics['test_coverage'] > baseline_gen_metrics['test_coverage'] else '✓ Baseline'}")
    
    # 最近邻距离比（接近1最好 - 测试集不会太远离训练集）
    print(f"{'NN Distance Ratio':<40} "
          f"{baseline_gen_metrics['nn_distance_ratio']:<15.4f} "
          f"{ours_gen_metrics['nn_distance_ratio']:<15.4f} "
          f"{'✓ Ours' if abs(ours_gen_metrics['nn_distance_ratio'] - 1) < abs(baseline_gen_metrics['nn_distance_ratio'] - 1) else '✓ Baseline'}")


@click.command()
@click.option('--baseline_dir', required=True, help='Baseline feature directory')
@click.option('--ours_dir', required=True, help='Ours feature directory')
# @click.option('--output', default='comparison_report.png', help='Output image path')
def main(baseline_dir, ours_dir):
    enhanced_comparison(baseline_dir=baseline_dir, ours_dir=ours_dir)
    
if __name__ == '__main__':
    main()