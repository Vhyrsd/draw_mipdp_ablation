import pickle
# 打开并读取.pkl文件
with open('feats_dir_dp/eval_features.pkl', 'rb') as file:
   data = pickle.load(file)
print(data['episodes'][55].keys())
# print(data['episodes'][10]['features'])