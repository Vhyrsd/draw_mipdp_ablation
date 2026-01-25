import pickle
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, using numpy implementation")


def effective_rank(features):
    """
    计算特征的有效秩 (Effective Rank)
    
    Args:
        features: numpy array, shape (N, D)
                 N: 样本数, D: 特征维度
    
    Returns:
        effective_rank: float, 特征空间的有效维度
    """
    # 确保是numpy array
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    
    # 转换为float类型
    features = features.astype(np.float32)
    
    # 中心化特征
    features = features - features.mean(axis=0, keepdims=True)
    
    # SVD分解 (使用numpy)
    try:
        U, S, Vt = np.linalg.svd(features, full_matrices=False)
    except np.linalg.LinAlgError:
        print("Warning: SVD failed, returning 0")
        return 0.0
    
    # 归一化奇异值
    S_sum = S.sum()
    if S_sum < 1e-10:
        return 0.0
    
    S_norm = S / S_sum
    
    # 计算熵 (过滤掉接近0的值避免log(0))
    S_norm = S_norm[S_norm > 1e-10]
    entropy = -(S_norm * np.log(S_norm)).sum()
    
    # 有效秩 = exp(熵)
    eff_rank = np.exp(entropy)
    
    return float(eff_rank)


def compute_effective_rank_from_pkl(pkl_path):
    """
    从features.pkl文件计算有效秩
    
    Args:
        pkl_path: features.pkl文件路径
    
    Returns:
        dict: 包含多种统计的有效秩结果
    """
    # 加载数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Total episodes: {len(data['episodes'])}")
    
    # 收集所有特征
    all_features = []
    env_features = {}  # 按环境分组
    
    for episode in data['episodes']:
        env_id = episode['env_id']
        features = episode['features']  # shape: (timesteps, feature_dim)
        
        # 确保是numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # 收集所有特征
        all_features.append(features)
        
        # 按环境分组
        if env_id not in env_features:
            env_features[env_id] = []
        env_features[env_id].append(features)
    
    # 1. 全局有效秩 (所有环境所有timestep)
    all_features_concat = np.concatenate(all_features, axis=0)
    print(f"Concatenated features shape: {all_features_concat.shape}")
    
    global_eff_rank = effective_rank(all_features_concat)
    
    print(f"\n{'='*60}")
    print(f"Global Effective Rank (all data): {global_eff_rank:.4f}")
    print(f"Feature dimension: {all_features_concat.shape[1]}")
    print(f"Total samples: {all_features_concat.shape[0]}")
    print(f"Utilization ratio: {global_eff_rank/all_features_concat.shape[1]*100:.2f}%")
    
    # 2. 每个环境的有效秩
    print(f"\n{'='*60}")
    print("Per-Environment Effective Rank:")
    env_eff_ranks = {}
    
    for env_id, features_list in sorted(env_features.items()):
        env_features_concat = np.concatenate(features_list, axis=0)
        env_eff_rank = effective_rank(env_features_concat)
        env_eff_ranks[env_id] = env_eff_rank
        print(f"  Env {env_id}: {env_eff_rank:.4f} (samples: {env_features_concat.shape[0]})")
    
    avg_env_eff_rank = np.mean(list(env_eff_ranks.values()))
    print(f"  Average: {avg_env_eff_rank:.4f}")
    
    # 3. 每条轨迹的有效秩
    print(f"\n{'='*60}")
    print("Per-Trajectory Effective Rank Statistics:")
    traj_eff_ranks = []
    
    for episode in data['episodes']:
        features = episode['features']
        if not isinstance(features, np.ndarray):
            features = np.array(features)
            
        if len(features) > 1:  # 至少需要2个样本
            traj_eff_rank = effective_rank(features)
            traj_eff_ranks.append(traj_eff_rank)
    
    if len(traj_eff_ranks) > 0:
        print(f"  Mean: {np.mean(traj_eff_ranks):.4f}")
        print(f"  Std:  {np.std(traj_eff_ranks):.4f}")
        print(f"  Min:  {np.min(traj_eff_ranks):.4f}")
        print(f"  Max:  {np.max(traj_eff_ranks):.4f}")
    else:
        print("  No valid trajectories found")
    
    # 返回结果
    results = {
        'global_effective_rank': global_eff_rank,
        'avg_env_effective_rank': avg_env_eff_rank,
        'per_env_effective_rank': env_eff_ranks,
        'trajectory_effective_rank_mean': np.mean(traj_eff_ranks) if len(traj_eff_ranks) > 0 else 0,
        'trajectory_effective_rank_std': np.std(traj_eff_ranks) if len(traj_eff_ranks) > 0 else 0,
        'feature_dim': all_features_concat.shape[1],
        'utilization_ratio': global_eff_rank/all_features_concat.shape[1]
    }
    
    return results


def compare_effective_ranks(baseline_pkl, ours_pkl):
    """
    比较baseline和ours的有效秩
    
    Args:
        baseline_pkl: baseline的features.pkl路径
        ours_pkl: ours的features.pkl路径
    """
    print("="*60)
    print("BASELINE Results:")
    print("="*60)
    baseline_results = compute_effective_rank_from_pkl(baseline_pkl)
    
    print("\n" + "="*60)
    print("OURS Results:")
    print("="*60)
    ours_results = compute_effective_rank_from_pkl(ours_pkl)
    
    # 对比结果
    print("\n" + "="*60)
    print("COMPARISON:")
    print("="*60)
    
    metrics = [
        ('Global Effective Rank', 'global_effective_rank'),
        ('Avg Env Effective Rank', 'avg_env_effective_rank'),
        ('Trajectory EffRank Mean', 'trajectory_effective_rank_mean'),
        ('Feature Utilization', 'utilization_ratio')
    ]
    
    for name, key in metrics:
        baseline_val = baseline_results[key]
        ours_val = ours_results[key]
        improvement = (ours_val - baseline_val) / (baseline_val + 1e-10) * 100
        symbol = "✓" if improvement > 0 else "✗"
        
        if 'ratio' in key:
            print(f"{name:30s} {baseline_val:.4f} → {ours_val:.4f}  {symbol} {improvement:+.2f}%")
        else:
            print(f"{name:30s} {baseline_val:.4f} → {ours_val:.4f}  {symbol} {improvement:+.2f}%")
    
    return baseline_results, ours_results


# 使用示例
if __name__ == "__main__":
    # # 单个文件分析
    # results = compute_effective_rank_from_pkl('features_baseline.pkl')
    
    # 或者对比两个模型
    baseline_results, ours_results = compare_effective_ranks(
        'feats_dir_dp/eval_features.pkl',
        'feats_dir_ours/eval_features.pkl'
    )