## GLB Post-Processing Module

基于 Blender 的 GLB 文件自动化后处理工具，用于清理和优化 3D 网格模型

### Requirements

- Blender 3.0+  我用的4.0
- Python 3.x（Blender 内置）

### Usage

```bash
# 单文件
blender --background --python PostProcessing/PostProcessing.py -- --input model.glb --output processed.glb

# batch
blender --background --python PostProcessing/PostProcessing.py -- --batch --input ./models/ --output ./output/
```

### 小巧思

最有用的是合并顶点(distance) 和最后的 Decimate 减面,主要优化就在这

如果减面效果不好或者让模型丢mesh了，尝试：

1. 把 decimate-ratio 调高一点，这个是减面的比例，越低减的越多，我开的0.8，对hunyuan可能0.5都可以，对TRELLIS而言可能要关掉 smooth 和 decimate，它的面有点太少了
2. 把平滑ban了
3. 把 merge-distance 调低一点，不过我感觉这个应该影响很小，草他都已经这么低了还能破坏mesh那我也没招了

实测hunyuan的可以优化的点和面会多一点，TRELLIS 感觉生成的面就不多，再减面可能会破坏模型本身的结构

```bash
# 这是一组很保守的参数
blender --background --python PostProcessing/PostProcessing.py -- --batch --input TRELLIS --output post_lower_merge --merge-distance 0.00001  --no-smooth --no-decimate
```

### ToDo

对 post 的 glb 跑一下 evaluate，看看会不会影响模型的效果

### 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` |  | 输入文件路径或文件夹路径 |
| `--output` | `-o` | `post_原文件名.glb` | 输出文件路径或文件夹路径 |
| `--batch` | `-b` | `False` |  |
| `--merge-distance` | | `0.0001` | 合并顶点的距离阈值 |
| `--decimate-ratio` | | `0.8` | Decimate 简化比率 (0-1) |
| `--smooth-iterations` | | `5` | 平滑迭代次数 |
| `--no-smooth` | | `False` | 禁用表面平滑 |
| `--no-decimate` | | `False` | 禁用 Decimate 简化 |
| `--no-merge` | | `False` | 禁用合并(不建议) |

____

#### 以下为 Gemini生成，G老师说得对，不知道看啥就看看这吧()

### 输出文件命名

- 默认输出文件名格式：`post_原文件名.glb`

### 处理流程

```
输入 GLB
    │
    ▼
┌─────────────────┐
│  Mesh Cleanup   │
│  ─────────────  │
│  • 合并顶点      │
│  • 删除孤立元素   │
│  • 删除退化面    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Geometry Repair │
│  ─────────────  │
│  • 修复法线方向   │
│  • 修复非流形边   │
│  • 填充孔洞      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Topology Optim  │
│  ─────────────  │
│  • Decimate     │
│  • Smooth       │
└────────┬────────┘
         │
         ▼
输出 post_xxx.glb
```

### 功能特性

#### 1. Mesh Cleanup（网格清理）

- **Remove duplicate vertices** - 移除重复顶点
- **Delete isolated vertices & edges** - 删除孤立的顶点和边
- **Delete degenerate faces** - 删除退化面（零面积面）

#### 2. Geometry Repair（几何修复）

- **Fix normal direction** - 修复法线方向
- **Repair non-manifold edges** - 修复非流形边
- **Fill holes** - 填充孔洞

#### 3. Topology Optimization（拓扑优化）

- **Decimate** - 网格简化
- **Surface smooth** - 表面平滑