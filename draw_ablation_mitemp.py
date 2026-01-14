import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline

# ==========================================
# 第一步：生成一个测试用的 CSV 文件 (如果你已有文件，可跳过此步)
# ==========================================
csv_filename = 'mitemp_wandb_export_2026-01-14T14_21_07.630+08_00.csv'


# ==========================================
# 第二步：读取 CSV 并画图 (核心代码)
# ==========================================

# 1. 设置中文字体 (非常重要，否则中文会显示成方块)
# Windows 系统通常使用 'SimHei' (黑体)
# Mac 系统通常使用 'Arial Unicode MS' 或 'PingFang HK'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

try:
    # 2. 读取 CSV 文件
    # encoding='utf-8' 是标准格式，如果报错可能是 'gbk'
    df = pd.read_csv(csv_filename)

    # 3. 数据预处理 (可选但推荐)
    # 如果x轴是日期，最好转换为日期格式，这样matplotlib会自动优化坐标轴显示
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'])
        df.sort_values('日期', inplace=True) # 确保按日期排序

    # 4. 创建画布
    plt.figure(figsize=(10, 6))  # 设置图片大小 (宽, 高)

    # smooth
    x = df['Step']
    y1 = df['miloss0.01_mitemp0.5_15/50_2022.12.29-22.31.27_train_diffusion_unet_hybrid_lift_image - test/mean_score']
    y2 = df['miloss0.01_mitemp0.2_15/50_2022.12.29-22.31.27_train_diffusion_unet_hybrid_lift_image - test/mean_score']
    y3 = df['miloss0.01_mitemp0.05_15/50_2022.12.29-22.31.27_train_diffusion_unet_hybrid_lift_image - test/mean_score']
    y4 = df['miloss0.01_mitemp0.1_15/50_2022.12.29-22.31.27_train_diffusion_unet_hybrid_lift_image - test/mean_score']
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl1 = make_interp_spline(x, y1, k=3)
    spl2 = make_interp_spline(x, y2, k=3)
    spl3 = make_interp_spline(x, y3, k=3)
    spl4 = make_interp_spline(x, y4, k=3)
    y1_smooth = spl1(x_smooth)
    y2_smooth = spl2(x_smooth)
    y3_smooth = spl3(x_smooth)
    y4_smooth = spl4(x_smooth)
    y1_smooth = np.clip(y1_smooth, 0, 1)
    y2_smooth = np.clip(y2_smooth, 0, 1)
    y3_smooth = np.clip(y3_smooth, 0, 1)
    y4_smooth = np.clip(y4_smooth, 0, 1)
    plt.plot(x_smooth, y1_smooth, label='temperature=0.5')
    plt.plot(x_smooth, y2_smooth, label='temperature=0.2')
    plt.plot(x_smooth, y3_smooth, label='temperature=0.05')
    plt.plot(x_smooth, y4_smooth, label='temperature=0.1')
    plt.scatter(x, y1, alpha=0.3, s=10)
    plt.scatter(x, y2, alpha=0.3, s=10)
    plt.scatter(x, y3, alpha=0.3, s=10)
    plt.scatter(x, y4, alpha=0.3, s=10)

    # 5. 绘制折线图
    # x轴数据, y轴数据, label=图例名称, marker=数据点样式, ...
    # plt.plot(df['Step'], df['miloss0.001_mitemp0.1_15/50_2022.12.29-22.31.27_train_diffusion_unet_hybrid_lift_image - test/mean_score'], label='weight=0.001', marker='', linestyle='-', color='b')
    # plt.plot(df['Step'], df['miloss0.01_mitemp0.1_15/50_2022.12.29-22.31.27_train_diffusion_unet_hybrid_lift_image - test/mean_score'], label='weight=0.01', marker='', linestyle='-', color='r')
    # plt.plot(df['Step'], df['miloss0.1_mitemp0.1_15/50_2022.12.29-22.31.27_train_diffusion_unet_hybrid_lift_image - test/mean_score'], label='weight=0.1', marker='', linestyle='-', color='g')

    # rolling mean
    # window_size = 3
    # plt.plot(df['Step'], df['miloss0.01_mitemp0.5_15/50_2022.12.29-22.31.27_train_diffusion_unet_hybrid_lift_image - test/mean_score'].rolling(window=window_size, min_periods=1).mean(), label='0.001_rollingmean', marker='', linestyle='--', color='b')
    # plt.plot(df['Step'], df['miloss0.01_mitemp0.2_15/50_2022.12.29-22.31.27_train_diffusion_unet_hybrid_lift_image - test/mean_score'].rolling(window=window_size, min_periods=1).mean(), label='0.01_rollingmean', marker='', linestyle='-', color='r')
    # plt.plot(df['Step'], df['miloss0.01_mitemp0.05_15/50_2022.12.29-22.31.27_train_diffusion_unet_hybrid_lift_image - test/mean_score'].rolling(window=window_size, min_periods=1).mean(), label='0.1_rollingmean', marker='', linestyle='-.', color='g')
    # plt.plot(df['Step'], df['miloss0.01_mitemp0.1_15/50_2022.12.29-22.31.27_train_diffusion_unet_hybrid_lift_image - test/mean_score'].rolling(window=window_size, min_periods=1).mean(), label='0.1_rollingmean', marker='', linestyle=':', color='c')

    # 6. 添加图表细节
    plt.title('figure', fontsize=16)  # 标题
    plt.xlabel('step', fontsize=12)            # X轴标签
    plt.ylabel('success rate', fontsize=12)            # Y轴标签
    plt.grid(True, linestyle='--', alpha=0.5) # 添加网格线，alpha是透明度
    plt.legend()                              # 显示图例

    # 自动旋转日期标签，防止重叠
    plt.xticks(rotation=45)
    
    # 调整布局，防止标签被切掉
    plt.tight_layout()

    # 7. 显示或保存图片
    # plt.savefig('result.png') # 如果想保存成图片，取消注释这行
    plt.show()

except FileNotFoundError:
    print(f"错误：找不到文件 {csv_filename}，请检查路径。")
except Exception as e:
    print(f"发生了错误: {e}")

# 清理生成的测试文件 (可选)
# os.remove(csv_filename) 
