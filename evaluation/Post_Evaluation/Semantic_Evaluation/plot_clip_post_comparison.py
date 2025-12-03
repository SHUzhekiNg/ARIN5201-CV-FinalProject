#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Hunyuan3D-2 与 TRELLIS 的 CLIP Score 箱线图对比
包含 Post-Processing 前后的对比（共4个箱线图）
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# 读取原始数据（未经后处理）
with open('/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/Semantic_Evaluation/clip_batch_results.json', 'r') as f:
    orig_data = json.load(f)

# 读取 Post-Processed 数据
with open('/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/Post_Evaluation/Semantic_Evaluation/clip_batch_results.json', 'r') as f:
    post_data = json.load(f)

# 提取原始 CLIP 分数
hunyuan_scores = []
trellis_scores = []
for sample in orig_data['samples']:
    if sample['hunyuan3d2'] and 'clip_score_mean' in sample['hunyuan3d2']:
        hunyuan_scores.append(sample['hunyuan3d2']['clip_score_mean'])
    if sample['trellis'] and 'clip_score_mean' in sample['trellis']:
        trellis_scores.append(sample['trellis']['clip_score_mean'])

hunyuan_scores = np.array(hunyuan_scores)
trellis_scores = np.array(trellis_scores)

# 提取 Post-Processed CLIP 分数
post_hunyuan_scores = []
post_trellis_scores = []
for sample in post_data['samples']:
    if sample['hunyuan3d2'] and 'clip_score_mean' in sample['hunyuan3d2']:
        post_hunyuan_scores.append(sample['hunyuan3d2']['clip_score_mean'])
    if sample['trellis'] and 'clip_score_mean' in sample['trellis']:
        post_trellis_scores.append(sample['trellis']['clip_score_mean'])

post_hunyuan_scores = np.array(post_hunyuan_scores)
post_trellis_scores = np.array(post_trellis_scores)

# 色盘
colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2']

# 创建图形
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

# 准备数据：Hunyuan3D-2, Post-Hunyuan3D-2, TRELLIS, Post-TRELLIS
data = [hunyuan_scores, post_hunyuan_scores, trellis_scores, post_trellis_scores]
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
    ax.annotate(f'Mean: {mean:.4f}\nMedian: {median:.4f}', 
                xy=(i, d.max() + 0.01), ha='center', fontsize=8, color='#333333')

# 添加分隔线区分两个模型
ax.axvline(x=2.5, color='#999999', linestyle='--', linewidth=1.5, alpha=0.7)

# 添加模型标签
ax.text(1.5, -0.08, 'Hunyuan3D-2', ha='center', fontsize=11, fontweight='bold', 
        transform=ax.get_xaxis_transform(), color='#2E7D6B')
ax.text(3.5, -0.08, 'TRELLIS', ha='center', fontsize=11, fontweight='bold',
        transform=ax.get_xaxis_transform(), color='#3A6B8C')

# 设置标签和标题
ax.set_ylabel('CLIP Score (↑ higher is better)', fontsize=12, fontweight='bold')
ax.set_title('Semantic Alignment: Effect of Post-Processing\n(CLIP Score on Toys2h Dataset, n=50)', 
             fontsize=14, fontweight='bold', pad=15)

# 美化
all_data = np.concatenate(data)
ax.set_ylim(all_data.min() - 0.03, all_data.max() + 0.06)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加图例
ax.legend(loc='lower right', framealpha=0.9)

plt.tight_layout()

# 保存
output_path = '/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/Post_Evaluation/Semantic_Evaluation/output/clip_post_comparison_boxplot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
print(f"Saved: {output_path.replace('.png', '.pdf')}")

# 打印统计摘要
print("\n" + "="*70)
print("Statistical Summary: CLIP Score (Semantic Alignment)")
print("="*70)
print(f"\n{'Model':<20} {'Mean':<10} {'Std':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
print("-"*70)
print(f"{'Hunyuan3D-2':<20} {hunyuan_scores.mean():<10.4f} {hunyuan_scores.std():<10.4f} {np.median(hunyuan_scores):<10.4f} {hunyuan_scores.min():<10.4f} {hunyuan_scores.max():<10.4f}")
print(f"{'Post-Hunyuan3D-2':<20} {post_hunyuan_scores.mean():<10.4f} {post_hunyuan_scores.std():<10.4f} {np.median(post_hunyuan_scores):<10.4f} {post_hunyuan_scores.min():<10.4f} {post_hunyuan_scores.max():<10.4f}")
print(f"{'TRELLIS':<20} {trellis_scores.mean():<10.4f} {trellis_scores.std():<10.4f} {np.median(trellis_scores):<10.4f} {trellis_scores.min():<10.4f} {trellis_scores.max():<10.4f}")
print(f"{'Post-TRELLIS':<20} {post_trellis_scores.mean():<10.4f} {post_trellis_scores.std():<10.4f} {np.median(post_trellis_scores):<10.4f} {post_trellis_scores.min():<10.4f} {post_trellis_scores.max():<10.4f}")
print("-"*70)

# 计算改进百分比
hunyuan_improvement = (post_hunyuan_scores.mean() / hunyuan_scores.mean() - 1) * 100
trellis_improvement = (post_trellis_scores.mean() / trellis_scores.mean() - 1) * 100

print(f"\nPost-Processing Effect:")
print(f"  Hunyuan3D-2: {hunyuan_improvement:+.2f}% CLIP score change")
print(f"  TRELLIS:     {trellis_improvement:+.2f}% CLIP score change")
print("="*70)

plt.show()
