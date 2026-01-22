"""
检查保存的特征文件
Usage:
python check_features.py --feature_dir data/pusht_eval_output/features
"""
import pickle
import numpy as np
import click
from pathlib import Path

@click.command()
@click.option('--feature_dir', required=True)
def main(feature_dir):
    feature_path = Path(feature_dir) / 'eval_features.pkl'
    
    print(f"Checking: {feature_path}")
    
    if not feature_path.exists():
        print(f"❌ File not found!")
        return
    
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ File loaded successfully")
    print(f"Keys: {list(data.keys())}")
    print(f"Number of episodes: {len(data['episodes'])}")
    
    if len(data['episodes']) == 0:
        print("❌ No episodes found!")
        return
    
    print("\n=== Episode Details ===")
    for i, ep in enumerate(data['episodes'][:3]):  # 只显示前3个
        print(f"Episode {i}:")
        print(f"  env_id: {ep['env_id']}")
        print(f"  env_prefix: {ep['env_prefix']}")
        print(f"  env_seed: {ep['env_seed']}")
        print(f"  timesteps: {len(ep['timesteps'])}")
        print(f"  observations: {len(ep['observations'])}")
        print(f"  actions shape: {ep['actions'].shape}")
        print(f"  rewards shape: {ep['rewards'].shape}")
        
        features = ep['features']
        if isinstance(features, np.ndarray):
            print(f"  features shape: {features.shape}")
            print(f"  features dtype: {features.dtype}")
            if features.size > 0:
                print(f"  features range: [{features.min():.3f}, {features.max():.3f}]")
            else:
                print(f"  ⚠ features is EMPTY!")
        elif isinstance(features, dict):
            print(f"  features type: dict with keys {list(features.keys())}")
            for k, v in features.items():
                if isinstance(v, np.ndarray):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                    if v.size > 0:
                        print(f"      range=[{v.min():.3f}, {v.max():.3f}]")
                    else:
                        print(f"      ⚠ EMPTY!")
        else:
            print(f"  ⚠ features type: {type(features)}")
    
    # 统计有效特征的数量
    valid_count = 0
    empty_count = 0
    for ep in data['episodes']:
        features = ep['features']
        if isinstance(features, np.ndarray):
            if features.size > 0:
                valid_count += 1
            else:
                empty_count += 1
        elif isinstance(features, dict):
            if any(v.size > 0 for v in features.values() if isinstance(v, np.ndarray)):
                valid_count += 1
            else:
                empty_count += 1
    
    print(f"=== Summary ===")
    print(f"Total episodes: {len(data['episodes'])}")
    print(f"Valid features: {valid_count}")
    print(f"Empty features: {empty_count}")
    
    if valid_count == 0:
        print("\n❌ No valid features found!")
        print("Possible reasons:")
        print("1. Feature extraction failed during eval")
        print("2. Policy doesn't have obs_encoder or vision_encoder")
        print("3. Feature extraction code has bugs")
    else:
        print(f"\n✓ Found {valid_count} episodes with valid features")

if __name__ == '__main__':
    main()
