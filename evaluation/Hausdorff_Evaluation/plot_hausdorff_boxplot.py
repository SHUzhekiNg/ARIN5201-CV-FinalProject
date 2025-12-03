#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Hunyuan3D-2 与 TRELLIS 的 Hausdorff 距离箱线图对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv('/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/hausdorff_batch_results.csv')

# 准备数据
hunyuan_hd = df['hunyuan3d2_hd'].dropna()
trellis_hd = df['trellis_hd'].dropna()

# 色盘
colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2']

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

# 绘制箱线图
data = [hunyuan_hd, trellis_hd]
labels = ['Hunyuan3D-2', 'TRELLIS']

bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                medianprops=dict(color='#333333', linewidth=2),
                whiskerprops=dict(color='#666666', linewidth=1.5),
                capprops=dict(color='#666666', linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='#FA7F6F', markersize=6, alpha=0.7))

# 设置箱体颜色
box_colors = [colors[0], colors[3]]  # #8ECFC9 (青绿) 和 #82B0D2 (蓝)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor('#333333')
    patch.set_linewidth(1.5)

# 添加散点 (jittered)
np.random.seed(42)
for i, d in enumerate(data, 1):
    jitter = np.random.normal(0, 0.04, size=len(d))
    ax.scatter(np.full(len(d), i) + jitter, d, alpha=0.5, s=25, 
               color=colors[1] if i == 1 else colors[4], edgecolor='white', linewidth=0.5, zorder=3)

# 添加均值标记
means = [d.mean() for d in data]
ax.scatter([1, 2], means, marker='D', color='#FA7F6F', s=80, zorder=5, edgecolor='white', linewidth=1.5, label='Mean')

# 标注统计信息
for i, (d, mean) in enumerate(zip(data, means), 1):
    ax.annotate(f'Mean: {mean:.3f}\nMedian: {d.median():.3f}', 
                xy=(i, d.max() + 0.05), ha='center', fontsize=9, color='#333333')

# 设置标签和标题
ax.set_ylabel('Hausdorff Distance (↓ lower is better)', fontsize=12, fontweight='bold')
ax.set_title('3D Reconstruction Quality: Hunyuan3D-2 vs TRELLIS\n(Hausdorff Distance on Toys2h Dataset, n=50)', 
             fontsize=13, fontweight='bold', pad=15)

# 美化
ax.set_ylim(0, max(hunyuan_hd.max(), trellis_hd.max()) + 0.2)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加图例
ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()

# 保存
output_path = '/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/hausdorff_boxplot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
print(f"Saved: {output_path.replace('.png', '.pdf')}")

# 打印统计摘要
print("\n===== Statistical Summary =====")
print(f"Hunyuan3D-2: Mean={hunyuan_hd.mean():.4f}, Std={hunyuan_hd.std():.4f}, Median={hunyuan_hd.median():.4f}")
print(f"TRELLIS:     Mean={trellis_hd.mean():.4f}, Std={trellis_hd.std():.4f}, Median={trellis_hd.median():.4f}")
print(f"\nTRELLIS improvement: {(1 - trellis_hd.mean()/hunyuan_hd.mean())*100:.1f}% lower HD (better)")

plt.show()
