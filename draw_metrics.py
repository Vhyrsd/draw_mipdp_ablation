import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 数据
metrics = ['Feature Variance', 'Inter Env Distance',
           'Global Effective Rank', 'Average Env Effective Rank']
baseline = [0.0008, 0.2363, 31.9, 8.7]
ours = [0.0017, 0.3416, 40.1, 9.6]

# 现代配色方案
color_baseline = '#4A90E2'  # 专业蓝
color_ours = '#F5A623'      # 活力橙
color_bg = '#F8F9FA'        # 浅灰背景
color_grid = '#E1E8ED'      # 网格颜色

# ============= 图表1: 四宫格对比 =============
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')

# 添加总标题
fig.suptitle('Performance Metrics Comparison',
             fontsize=22, fontweight='bold', y=0.98,
             color='#2C3E50', family='sans-serif')

# 创建子图
for idx, (metric, base_val, our_val) in enumerate(zip(metrics, baseline, ours)):
    ax = plt.subplot(2, 2, idx + 1)
    ax.set_facecolor(color_bg)

    # 数据准备
    x = np.arange(2)
    values = [base_val, our_val]
    colors_bar = [color_baseline, color_ours]

    # 绘制柱状图 - 添加渐变效果
    bars = ax.bar(x, values, width=0.6, color=colors_bar,
                  alpha=0.85, edgecolor='white', linewidth=3,
                  zorder=3)

    # 添加阴影效果
    for i, bar in enumerate(bars):
        shadow = Rectangle((bar.get_x(), 0), bar.get_width(), bar.get_height(),
                           facecolor='gray', alpha=0.1, zorder=2,
                           transform=ax.transData)
        shadow.set_x(bar.get_x() + 0.02)
        shadow.set_y(-0.02 * max(values))
        ax.add_patch(shadow)

    # 添加数值标签 - 更精美的样式
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        label_text = f'{val:.4f}' if val < 1 else f'{val:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label_text,
                ha='center', va='bottom', fontsize=13,
                fontweight='bold', color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white',
                          edgecolor=colors_bar[i],
                          linewidth=2, alpha=0.9))

    # 设置标题
    ax.set_title(metric, fontsize=14, fontweight='bold',
                 pad=15, color='#34495E', family='sans-serif')

    # 设置坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline', 'Ours'], fontsize=12, fontweight='600')
    ax.set_ylabel('Value', fontsize=11, fontweight='600', color='#5D6D7E')

    # 美化网格
    ax.grid(axis='y', alpha=0.3, linestyle='--',
            linewidth=1, color=color_grid, zorder=1)
    ax.set_axisbelow(True)

    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # 计算并显示提升百分比 - 更醒目的样式
    improvement = ((our_val - base_val) / base_val) * 100
    color_improvement = '#27AE60' if improvement > 0 else '#E74C3C'

    bbox_props = dict(boxstyle='round,pad=0.6',
                      facecolor=color_improvement,
                      edgecolor='white',
                      linewidth=2.5, alpha=0.9)

    ax.text(0.98, 0.97, f'{improvement:+.1f}%',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold', color='white',
            bbox=bbox_props, zorder=10)

    # 添加小图标标识
    ax.text(0.02, 0.97, '📊', transform=ax.transAxes,
            ha='left', va='top', fontsize=16, zorder=10)

plt.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.savefig('metrics_comparison_beautiful.png', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# ============= 图表2: 综合对比图 =============
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor(color_bg)

# 归一化数据
baseline_norm = np.array(baseline) / np.array(baseline)
ours_norm = np.array(ours) / np.array(baseline)

x = np.arange(len(metrics))
width = 0.35

# 绘制柱状图
bars1 = ax.bar(x - width/2, baseline_norm, width,
               label='Baseline', color=color_baseline,
               alpha=0.85, edgecolor='white', linewidth=2.5, zorder=3)

bars2 = ax.bar(x + width/2, ours_norm, width,
               label='Ours', color=color_ours,
               alpha=0.85, edgecolor='white', linewidth=2.5, zorder=3)

# 添加阴影效果
for bars in [bars1, bars2]:
    for bar in bars:
        shadow = Rectangle((bar.get_x(), 0), bar.get_width(), bar.get_height(),
                           facecolor='gray', alpha=0.08, zorder=2)
        shadow.set_x(bar.get_x() + 0.03)
        shadow.set_y(-0.01)
        ax.add_patch(shadow)

# 添加原始数值标签
for bars, values, color in [(bars1, baseline, color_baseline),
                            (bars2, ours, color_ours)]:
    for bar, val in zip(bars, values):
        height = bar.get_height()
        label_text = f'{val:.4f}' if val < 1 else f'{val:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                label_text,
                ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white',
                          edgecolor=color,
                          linewidth=1.5, alpha=0.95))

# 设置标题和标签
ax.set_xlabel('Metrics', fontsize=13, fontweight='bold',
              color='#34495E', labelpad=10)
ax.set_ylabel('Normalized Value (Baseline = 1.0)',
              fontsize=13, fontweight='bold', color='#34495E', labelpad=10)
ax.set_title('Comprehensive Metrics Comparison (Normalized)',
             fontsize=18, fontweight='bold', pad=20,
             color='#2C3E50', family='sans-serif')

# 设置x轴标签
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=0, ha='center',
                   fontsize=11, fontweight='600')

# 美化图例
legend = ax.legend(fontsize=12, loc='upper left',
                   frameon=True, fancybox=True, shadow=True,
                   framealpha=0.95, edgecolor='#BDC3C7',
                   borderpad=1, labelspacing=0.8)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_linewidth(2)

# 添加基准线
ax.axhline(y=1, color='#95A5A6', linestyle='--',
           linewidth=2, alpha=0.6, zorder=1, label='_nolegend_')

# 添加基准线标注
ax.text(len(metrics)-0.1, 1.02, 'Baseline Reference',
        fontsize=9, style='italic', color='#7F8C8D',
        ha='right', va='bottom')

# 美化网格
ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=1,
        color=color_grid, zorder=1)
ax.set_axisbelow(True)

# 移除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BDC3C7')
ax.spines['bottom'].set_color('#BDC3C7')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# 设置y轴范围，留出空间
y_max = max(ours_norm) * 1.15
ax.set_ylim(0, y_max)

# 添加平均提升信息框
avg_improvement = np.mean([(o-b)/b*100 for o, b in zip(ours, baseline)])
info_text = f'Average Improvement: {avg_improvement:+.1f}%'
bbox_props = dict(boxstyle='round,pad=0.8',
                  facecolor='#27AE60' if avg_improvement > 0 else '#E74C3C',
                  edgecolor='white', linewidth=3, alpha=0.9)
ax.text(0.98, 0.97, info_text, transform=ax.transAxes,
        fontsize=13, fontweight='bold', color='white',
        ha='right', va='top', bbox=bbox_props, zorder=10)

# 添加装饰性图标
ax.text(0.02, 0.97, '📈', transform=ax.transAxes,
        fontsize=20, ha='left', va='top', zorder=10)

plt.tight_layout()
plt.savefig('metrics_comparison_normalized_beautiful.png', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# ============= 打印美化的统计信息 =============
print("\n" + "="*70)
print("  📊 PERFORMANCE METRICS COMPARISON SUMMARY")
print("="*70)

for i, (metric, base_val, our_val) in enumerate(zip(metrics, baseline, ours), 1):
    improvement = ((our_val - base_val) / base_val) * 100
    arrow = "↑" if improvement > 0 else "↓"

    print(f"\n{i}. {metric}")
    print(f"   {'─'*60}")
    print(f"   Baseline:    {base_val:.4f}" if base_val <
          1 else f"   Baseline:    {base_val:.1f}")
    print(f"   Ours:        {our_val:.4f}" if our_val <
          1 else f"   Ours:        {our_val:.1f}")
    print(f"   Improvement: {arrow} {abs(improvement):.2f}%")

print("\n" + "="*70)
print(f"  Average Improvement: {avg_improvement:+.2f}%")
print("="*70 + "\n")
