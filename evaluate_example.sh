#!/bin/bash
# Example usage of evaluate.py

# ============================================
# 示例 1: 评估单个从 inference3d.py 生成的模型
# ============================================
echo "Example 1: Evaluate from inference meta JSON"
python evaluate.py \
  --meta-json outputs/hy3d_out/meta_ff11db2d9eca45bf9f55ae75adb37492.json \
  --save-renders \
  --render-dir eval_renders \
  --use-consistency

# ============================================
# 示例 2: 评估单个 GLB 文件（带输入图像）
# ============================================
echo -e "\nExample 2: Evaluate single GLB with image condition"
python evaluate.py \
  --model outputs/hy3d_out/demo_hunyuan3d2.glb \
  --condition inputs/chair.png \
  --type image \
  --save-renders \
  --output-json results/chair_eval.json

# ============================================
# 示例 3: 评估单个 GLB 文件（带文本提示）
# ============================================
echo -e "\nExample 3: Evaluate single GLB with text prompt"
python evaluate.py \
  --model outputs/car.glb \
  --condition "a red sports car with aerodynamic design" \
  --type text \
  --output-json results/car_eval.json

# ============================================
# 示例 4: 批量评估整个目录（无条件，纯质量评估）
# ============================================
echo -e "\nExample 4: Batch evaluate all GLB files"
python evaluate.py \
  --batch-dir outputs/hy3d_out \
  --pattern "*.glb" \
  --output-csv results/batch_evaluation.csv \
  --save-renders \
  --render-dir eval_renders

# ============================================
# 示例 5: 使用更高质量的 CLIP 模型
# ============================================
echo -e "\nExample 5: High-quality evaluation with ViT-L/14"
python evaluate.py \
  --meta-json outputs/hy3d_out/meta_xxx.json \
  --clip-model ViT-L/14 \
  --resolution 1024 \
  --num-views 16 \
  --elevations -30 0 30 \
  --use-aesthetic \
  --use-consistency

# ============================================
# 示例 6: 对比两个模型的评测结果
# ============================================
echo -e "\nExample 6: Compare TRELLIS vs Hunyuan3D-2"
# 评估 TRELLIS
python evaluate.py \
  --model outputs/chair_trellis.glb \
  --condition inputs/chair.png \
  --output-json results/chair_trellis_eval.json

# 评估 Hunyuan3D-2
python evaluate.py \
  --model outputs/chair_hunyuan3d2.glb \
  --condition inputs/chair.png \
  --output-json results/chair_hunyuan3d2_eval.json

# 比较结果（需要额外脚本或手动分析）
echo "Results saved. Compare JSON files in results/ directory"
