#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Post-Processing 网格简化效果柱状图
"""

import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['Hunyuan3D-2', 'TRELLIS']
vertices_drop = [46.76, 42.28]
edges_drop = [31.48, 28.81]
faces_drop = [20.81, 20.01]

# 色盘
colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2']

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, vertices_drop, width, label='Vertices Drop', color=colors[0], edgecolor='#333333', linewidth=1.2)
bars2 = ax.bar(x, edges_drop, width, label='Edges Drop', color=colors[1], edgecolor='#333333', linewidth=1.2)
bars3 = ax.bar(x + width, faces_drop, width, label='Faces Drop', color=colors[3], edgecolor='#333333', linewidth=1.2)

# 添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# 设置标签和标题
ax.set_ylabel('Drop Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Post-Processing: Mesh Simplification Effect\n(Percentage of Elements Removed)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9)

# 美化
ax.set_ylim(0, 55)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# 保存
output_path = '/disk2/licheng/code/ARIN5201-CV-FinalProject/PostProcessing/mesh_simplification_bar.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
print(f"Saved: {output_path.replace('.png', '.pdf')}")

plt.show()
