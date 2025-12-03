#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Hunyuan3D-2 与 TRELLIS 的 Hausdorff 距离箱线图对比
包含 Post-Processing 前后的对比（共4个箱线图）
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取原始数据（未经后处理）
df = pd.read_csv('/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/Hausdorff_Evaluation/hausdorff_batch_results.csv')
hunyuan_hd = df['hunyuan3d2_hd'].dropna().values
trellis_hd = df['trellis_hd'].dropna().values

# 读取 Post-Processed 数据
with open('/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/Post_Evaluation/Hausdorff_Evaluation/hausdorff_Hunyuan.json', 'r') as f:
    post_hunyuan_data = json.load(f)

with open('/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/Post_Evaluation/Hausdorff_Evaluation/hausdorff_Trellis.json', 'r') as f:
    post_trellis_data = json.load(f)

# 提取 Post-Processed Hausdorff 距离
post_hunyuan_hd = np.array([v['symmetric_hausdorff'] for v in post_hunyuan_data['results'].values() if v.get('success', False)])
post_trellis_hd = np.array([v['symmetric_hausdorff'] for v in post_trellis_data['results'].values() if v.get('success', False)])

# 色盘
colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2']

# 创建图形
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

# 准备数据：Hunyuan3D-2, Post-Hunyuan3D-2, TRELLIS, Post-TRELLIS
data = [hunyuan_hd, post_hunyuan_hd, trellis_hd, post_trellis_hd]
labels = ['Hunyuan3D-2', 'Post-Hunyuan3D-2', 'TRELLIS', 'Post-TRELLIS']

bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                medianprops=dict(color='#333333', linewidth=2),
                whiskerprops=dict(color='#666666', linewidth=1.5),
                capprops=dict(color='#666666', linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='#FA7F6F', markersize=6, alpha=0.7))

# 设置箱体颜色：原始用浅色，Post用深色（同系列）
box_colors = [colors[0], '#5DB5AB', colors[3], '#5A8FB8']  # 青绿系列，蓝色系列
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor('#333333')
    patch.set_linewidth(1.5)

# 添加散点 (jittered)
np.random.seed(42)
scatter_colors = [colors[1], colors[2], colors[4], colors[5]]  # 橙、红、紫、米
for i, d in enumerate(data, 1):
    jitter = np.random.normal(0, 0.06, size=len(d))
    ax.scatter(np.full(len(d), i) + jitter, d, alpha=0.5, s=20, 
               color=scatter_colors[i-1], edgecolor='white', linewidth=0.3, zorder=3)

# 添加均值标记
means = [d.mean() for d in data]
ax.scatter([1, 2, 3, 4], means, marker='D', color='#FA7F6F', s=80, zorder=5, 
           edgecolor='white', linewidth=1.5, label='Mean')

# 标注统计信息
for i, (d, mean) in enumerate(zip(data, means), 1):
    median = np.median(d)
    ax.annotate(f'Mean: {mean:.3f}\nMedian: {median:.3f}', 
                xy=(i, d.max() + 0.05), ha='center', fontsize=8, color='#333333')

# 添加分隔线区分两个模型
ax.axvline(x=2.5, color='#999999', linestyle='--', linewidth=1.5, alpha=0.7)

# 添加模型标签
ax.text(1.5, -0.18, 'Hunyuan3D-2', ha='center', fontsize=11, fontweight='bold', 
        transform=ax.get_xaxis_transform(), color='#2E7D6B')
ax.text(3.5, -0.18, 'TRELLIS', ha='center', fontsize=11, fontweight='bold',
        transform=ax.get_xaxis_transform(), color='#3A6B8C')

# 设置标签和标题
ax.set_ylabel('Hausdorff Distance (↓ lower is better)', fontsize=12, fontweight='bold')
ax.set_title('Geometric Quality: Effect of Post-Processing\n(Hausdorff Distance on Toys2h Dataset, n=50)', 
             fontsize=14, fontweight='bold', pad=15)

# 美化
y_max = max([d.max() for d in data])
ax.set_ylim(0, y_max + 0.25)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加图例
ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()

# 保存
output_path = '/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/Post_Evaluation/Hausdorff_Evaluation/output/hausdorff_post_comparison_boxplot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
print(f"Saved: {output_path.replace('.png', '.pdf')}")

# 打印统计摘要
print("\n" + "="*70)
print("Statistical Summary: Hausdorff Distance (Geometric Quality)")
print("="*70)
print(f"\n{'Model':<20} {'Mean':<10} {'Std':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
print("-"*70)
print(f"{'Hunyuan3D-2':<20} {hunyuan_hd.mean():<10.4f} {hunyuan_hd.std():<10.4f} {np.median(hunyuan_hd):<10.4f} {hunyuan_hd.min():<10.4f} {hunyuan_hd.max():<10.4f}")
print(f"{'Post-Hunyuan3D-2':<20} {post_hunyuan_hd.mean():<10.4f} {post_hunyuan_hd.std():<10.4f} {np.median(post_hunyuan_hd):<10.4f} {post_hunyuan_hd.min():<10.4f} {post_hunyuan_hd.max():<10.4f}")
print(f"{'TRELLIS':<20} {trellis_hd.mean():<10.4f} {trellis_hd.std():<10.4f} {np.median(trellis_hd):<10.4f} {trellis_hd.min():<10.4f} {trellis_hd.max():<10.4f}")
print(f"{'Post-TRELLIS':<20} {post_trellis_hd.mean():<10.4f} {post_trellis_hd.std():<10.4f} {np.median(post_trellis_hd):<10.4f} {post_trellis_hd.min():<10.4f} {post_trellis_hd.max():<10.4f}")
print("-"*70)

# 计算改进百分比
hunyuan_improvement = (1 - post_hunyuan_hd.mean() / hunyuan_hd.mean()) * 100
trellis_improvement = (1 - post_trellis_hd.mean() / trellis_hd.mean()) * 100

print(f"\nPost-Processing Improvement:")
print(f"  Hunyuan3D-2: {hunyuan_improvement:+.1f}% HD reduction")
print(f"  TRELLIS:     {trellis_improvement:+.1f}% HD reduction")
print("="*70)

plt.show()
