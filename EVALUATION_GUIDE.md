# 3D Model Evaluation Guide

本指南介绍如何使用 `evaluate.py` 评测三维生成模型的质量。

---

## 🎯 评测指标概览

### **1. CLIP Score（条件对齐）**
- 评估生成的3D模型与输入条件（文本/图像）的语义一致性
- **支持模式**：文生3D 和 图生3D
- **分数范围**：0.0 - 1.0（越高越好）
- **典型阈值**：
  - 优秀：≥ 0.80
  - 良好：0.70 - 0.80
  - 一般：0.60 - 0.70
  - 较差：< 0.60

### **2. Mesh Validity（几何有效性）**
- **Watertight**：网格是否封闭（无洞）
- **Manifold**：网格是否流形（每条边恰好被2个面共享）
- **Self-intersection**：是否有面片相互穿插
- **Validity Score**：综合有效性评分（0-1）

### **3. Aesthetic Score（视觉美学）**
- 使用预训练模型评估渲染图像的美学质量
- **需要额外模型**：`cafeai/cafe_aesthetic`
- **分数范围**：通常 0.0 - 10.0

### **4. Multi-view Consistency（多视角一致性）**
- 使用 LPIPS 评估不同视角之间的一致性
- **分数含义**：LPIPS距离，越低越好（更一致）
- **需要额外模型**：`lpips`

### **5. Geometric Statistics（几何统计）**
- 顶点数、面片数、表面积、体积等基础信息

---

## 📦 依赖安装

```bash
# 基础依赖（必须）
pip install torch torchvision
pip install trimesh pyrender pillow numpy pandas tqdm
pip install git+https://github.com/openai/CLIP.git

# 可选依赖（增强功能）
pip install lpips  # 用于多视角一致性评估
pip install transformers  # 用于美学评分

# 系统依赖（用于离屏渲染）
# Ubuntu/Debian:
sudo apt-get install libosmesa6-dev freeglut3-dev
# 或使用 EGL（推荐用于服务器）
sudo apt-get install libegl1-mesa-dev
```

**注意**：如果在远程服务器上运行，建议配置 EGL 用于无头渲染：
```bash
export PYOPENGL_PLATFORM=egl
```

---

## 🚀 使用方法

### **方式 1：评估 inference3d.py 的输出**

最简单的方式，直接从推理结果的 meta JSON 文件评估：

```bash
python evaluate.py \
  --meta-json outputs/hy3d_out/meta_xxx.json \
  --save-renders \
  --render-dir eval_renders
```

**输出**：
- 自动在 meta JSON 同目录生成 `meta_xxx.eval.json`
- 可选：保存多视角渲染图到 `eval_renders/`

---

### **方式 2：评估单个 GLB 文件（图生3D）**

```bash
python evaluate.py \
  --model outputs/chair_trellis.glb \
  --condition inputs/chair.png \
  --type image \
  --output-json results/chair_eval.json \
  --save-renders
```

---

### **方式 3：评估单个 GLB 文件（文生3D）**

```bash
python evaluate.py \
  --model outputs/car.glb \
  --condition "a red sports car with aerodynamic design" \
  --type text \
  --output-json results/car_eval.json
```

---

### **方式 4：批量评估（无条件）**

评估整个目录的所有模型，用于纯质量分析：

```bash
python evaluate.py \
  --batch-dir outputs/hy3d_out \
  --pattern "*.glb" \
  --output-csv results/batch_evaluation.csv
```

**输出 CSV 包含**：
- 模型名称、路径
- 几何统计（顶点数、面片数等）
- 网格有效性（水密性、流形性等）
- CLIP Score（如果提供了条件）
- 其他启用的指标

---

## ⚙️ 高级配置

### **使用更高质量的 CLIP 模型**

```bash
python evaluate.py \
  --meta-json outputs/meta_xxx.json \
  --clip-model ViT-L/14 \
  --resolution 1024 \
  --num-views 16 \
  --elevations -30 0 30 60
```

- `--clip-model ViT-L/14`：更精确但慢 1.5-2 倍
- `--resolution 1024`：更高分辨率渲染
- `--num-views 16`：每个仰角 16 个方位角
- `--elevations`：多个仰角（俯视、平视、仰视）

### **启用所有可选评测**

```bash
python evaluate.py \
  --meta-json outputs/meta_xxx.json \
  --use-aesthetic \
  --use-consistency \
  --save-renders
```

---

## 📊 输出格式

### **JSON 输出示例**

```json
{
  "model_path": "outputs/chair_trellis.glb",
  "model_name": "chair_trellis.glb",
  "condition": "image",
  "backend": "trellis",
  
  "evaluation_time": {
    "total": 15.23,
    "rendering": 3.45
  },
  
  "geometric_stats": {
    "num_vertices": 10234,
    "num_faces": 20156,
    "surface_area": 1.234,
    "volume": 0.567
  },
  
  "mesh_validity": {
    "is_watertight": true,
    "is_manifold": true,
    "has_self_intersection": false,
    "validity_score": 1.0,
    "num_degenerate_faces": 0
  },
  
  "clip_score": {
    "clip_score_mean": 0.8234,
    "clip_score_std": 0.0456,
    "clip_score_min": 0.7521,
    "clip_score_max": 0.8892,
    "clip_score_median": 0.8301,
    "per_view_scores": [0.82, 0.81, 0.79, ...],
    "condition_type": "image",
    "clip_model": "ViT-B/32"
  },
  
  "aesthetic_score": {
    "aesthetic_score_mean": 6.8,
    "aesthetic_score_std": 0.5,
    "per_view_scores": [6.5, 6.9, 7.1, ...]
  },
  
  "consistency": {
    "consistency_score_mean": 0.12,
    "consistency_score_std": 0.03,
    "note": "Lower LPIPS = better consistency"
  }
}
```

### **CSV 输出示例**

| model_name | num_vertices | num_faces | is_watertight | validity_score | clip_score_mean | clip_score_std |
|------------|--------------|-----------|---------------|----------------|-----------------|----------------|
| chair_trellis.glb | 10234 | 20156 | True | 1.0 | 0.8234 | 0.0456 |
| chair_hunyuan3d2.glb | 15678 | 31234 | True | 0.95 | 0.8567 | 0.0321 |

---

## 🔬 典型评测工作流

### **场景 1：对比 TRELLIS vs Hunyuan3D-2**

```bash
# Step 1: 生成模型（使用 inference3d.py）
python inference3d.py --model trellis --image inputs/chair.png --output outputs/
python inference3d.py --model hunyuan3d-2 --image inputs/chair.png --output outputs/ \
  --kwargs-json '{"model_path": "/path/to/models/Hunyuan3D-2"}'

# Step 2: 评估两个模型
python evaluate.py --meta-json outputs/meta_trellis_xxx.json --save-renders
python evaluate.py --meta-json outputs/meta_hunyuan_xxx.json --save-renders

# Step 3: 比较结果
python compare_results.py \
  outputs/meta_trellis_xxx.eval.json \
  outputs/meta_hunyuan_xxx.eval.json
```

### **场景 2：批量评估数据集**

```bash
# 假设你有 100 个输入图像，生成了 100×2=200 个模型（两个后端）

# 批量生成（使用脚本循环调用 inference3d.py）
for img in inputs/*.png; do
  python inference3d.py --model trellis --image "$img" --output outputs/trellis/
  python inference3d.py --model hunyuan3d-2 --image "$img" --output outputs/hunyuan/
done

# 批量评估
python evaluate.py \
  --batch-dir outputs/trellis \
  --pattern "*.glb" \
  --output-csv results/trellis_evaluation.csv

python evaluate.py \
  --batch-dir outputs/hunyuan \
  --pattern "*.glb" \
  --output-csv results/hunyuan_evaluation.csv

# 分析结果
python analyze_csv.py results/trellis_evaluation.csv results/hunyuan_evaluation.csv
```

---

## 📈 论文中如何呈现结果

### **表格 1：定量比较**

| Model | CLIP Score ↑ | Validity ↑ | Watertight (%) | Vertices | Faces | Time (s) |
|-------|--------------|------------|----------------|----------|-------|----------|
| TRELLIS | 0.823 ± 0.046 | 0.98 | 95.2% | 10.2k ± 2.1k | 20.3k ± 4.2k | 15 ± 2 |
| Hunyuan3D-2 | **0.857 ± 0.032** | **1.00** | **98.7%** | **15.6k ± 3.2k** | **31.2k ± 6.4k** | 20 ± 3 |

### **表格 2：细分指标**

| Metric | TRELLIS | Hunyuan3D-2 | Better |
|--------|---------|-------------|--------|
| CLIP Score (mean) | 0.823 | **0.857** | Hunyuan |
| CLIP Score (std) | 0.046 | **0.032** | Hunyuan |
| Aesthetic Score | 6.5 | **6.8** | Hunyuan |
| Consistency (LPIPS ↓) | 0.15 | **0.12** | Hunyuan |
| Watertight Rate | 95.2% | **98.7%** | Hunyuan |
| Generation Time | **15s** | 20s | TRELLIS |

### **可视化建议**
1. **雷达图**：多维度指标对比
2. **箱线图**：CLIP Score 分布
3. **散点图**：CLIP Score vs 几何复杂度
4. **渲染网格**：多视角效果展示

---

## 🐛 常见问题

### **1. 渲染失败：No EGL display**
```bash
# 解决方案 1：设置环境变量
export PYOPENGL_PLATFORM=egl

# 解决方案 2：安装 OSMesa
sudo apt-get install libosmesa6-dev
export PYOPENGL_PLATFORM=osmesa
```

### **2. CLIP 模型下载慢**
```bash
# 预先下载到本地
import clip
clip.load("ViT-B/32", download_root="/path/to/cache")

# 或设置代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### **3. CUDA 内存不足**
```bash
# 使用 CPU
python evaluate.py --device cpu ...

# 或减少分辨率和视角数
python evaluate.py --resolution 256 --num-views 4 ...
```

### **4. 批量评估时部分模型失败**
- 评测器会跳过失败的模型并在 CSV 中标记 error 列
- 检查日志查看具体错误原因

---

## 📚 参考

- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **LPIPS**: [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924)
- **Hunyuan3D-2**: 论文中使用的指标（CMMD, FID-CLIP, FID, CLIP-score）
- **TRELLIS**: 使用 TRELLIS-500K 数据集评测

---

## 🤝 贡献

如需添加新的评测指标或改进现有实现，请提交 PR 或 Issue。
