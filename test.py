import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# baseline = np.load('obs_features_dp.npz')
# ours = np.load('obs_features_midp_5.npz')

# feat_baseline = baseline['features']
# feat_ours = ours['features']

# timesteps_baseline = baseline['timesteps']  # 你需要在保存时记录这个
# timesteps_ours = ours['timesteps']

# episodes_ours = ours['episode_ids']

# print(timesteps_ours[:200])
# print(len(timesteps_ours))

# print(episodes_ours[:])
# print(len(episodes_ours))

data = np.load('20260120_170455.npz')
print(data['data'].shape)
