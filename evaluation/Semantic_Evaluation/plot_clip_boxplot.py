#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Hunyuan3D-2 与 TRELLIS 的 CLIP Score 箱线图对比
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
with open('/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/clip_batch_results.json', 'r') as f:
    data = json.load(f)

# 提取 CLIP 分数
hunyuan_scores = []
trellis_scores = []

for sample in data['samples']:
    if sample['hunyuan3d2'] and 'clip_score_mean' in sample['hunyuan3d2']:
        hunyuan_scores.append(sample['hunyuan3d2']['clip_score_mean'])
    if sample['trellis'] and 'clip_score_mean' in sample['trellis']:
        trellis_scores.append(sample['trellis']['clip_score_mean'])

hunyuan_scores = np.array(hunyuan_scores)
trellis_scores = np.array(trellis_scores)

# 色盘
colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2']

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

# 绘制箱线图
data_plot = [hunyuan_scores, trellis_scores]
labels = ['Hunyuan3D-2', 'TRELLIS']

bp = ax.boxplot(data_plot, labels=labels, patch_artist=True, widths=0.5,
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
for i, d in enumerate(data_plot, 1):
    jitter = np.random.normal(0, 0.04, size=len(d))
    ax.scatter(np.full(len(d), i) + jitter, d, alpha=0.5, s=25, 
               color=colors[1] if i == 1 else colors[4], edgecolor='white', linewidth=0.5, zorder=3)

# 添加均值标记
means = [d.mean() for d in data_plot]
ax.scatter([1, 2], means, marker='D', color='#FA7F6F', s=80, zorder=5, edgecolor='white', linewidth=1.5, label='Mean')

# 标注统计信息
for i, (d, mean) in enumerate(zip(data_plot, means), 1):
    ax.annotate(f'Mean: {mean:.4f}\nMedian: {np.median(d):.4f}', 
                xy=(i, d.max() + 0.02), ha='center', fontsize=9, color='#333333')

# 设置标签和标题
ax.set_ylabel('CLIP Score (↑ higher is better)', fontsize=12, fontweight='bold')
ax.set_title('Semantic Alignment: Hunyuan3D-2 vs TRELLIS\n(CLIP Score on Toys2h Dataset, n=50)', 
             fontsize=13, fontweight='bold', pad=15)

# 美化
ax.set_ylim(min(hunyuan_scores.min(), trellis_scores.min()) - 0.05, 
            max(hunyuan_scores.max(), trellis_scores.max()) + 0.08)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加图例
ax.legend(loc='lower right', framealpha=0.9)

plt.tight_layout()

# 保存
output_path = '/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/clip_boxplot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
print(f"Saved: {output_path.replace('.png', '.pdf')}")

# 打印统计摘要
print("\n===== Statistical Summary =====")
print(f"Hunyuan3D-2: Mean={hunyuan_scores.mean():.4f}, Std={hunyuan_scores.std():.4f}, Median={np.median(hunyuan_scores):.4f}")
print(f"TRELLIS:     Mean={trellis_scores.mean():.4f}, Std={trellis_scores.std():.4f}, Median={np.median(trellis_scores):.4f}")

diff = trellis_scores.mean() - hunyuan_scores.mean()
print(f"\nDifference (TRELLIS - Hunyuan3D-2): {diff:+.4f}")
winner = "TRELLIS" if diff > 0 else "Hunyuan3D-2"
print(f"Better semantic alignment: {winner}")

plt.show()
