# 3D Assets Generator

## Requirements
- Input: A text prompt OR a single 2D image

- Output: A 3D model file (.obj, .glb, .fbx) with textures

- Core Methodology: Implement at least one modern 3D generation technique (e.g., Diffusion Models, NeRFs, 3D Gaussians)

- Evaluation: Final report with quantitative and qualitative analysis

## Expected Outcomes
A successful project will deliver a complete pipeline and report demonstrating:

- **Geometric Quality**: Accuracy and structure of 3D shape using metrics like **Hausdorff Distance**

- **Visual Fidelity**: Realism and quality of applied textures and materials

- **Semantic Consistency**: Alignment between generated 3D model and input content

## Implementation Steps
1. Select a base model and 3D representation approach

2. Preprocess input data and set up training pipeline

3. Implement chosen generation technique with optimization

4. Generate 3D assets and evaluate using specified metrics


## Tips
- Use clear, well-lit images with simple backgrounds for image inputs

- Employ specific prompts describing materials, style, and proportions for text inputs

- Plan for post-processing in software like Blender for optimal quality



# TODOs:

- [x] Basic Pipeline @ Licheng
    - [x] Dataset Prepare
    - [x] Hunyuan3D 2.0
    - [x] Trellis3D

- [x] Website @ CC

- [x] Metrics(**Hausdorff Distance**) @ CC

- [x] Blender API PostProcessing @ Kexiang



# Installation

1. Follow [Hunyuan3D_2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2?tab=readme-ov-file#install-requirements)'s instructions to make your own `hunyuan3d2` env

2. Follow [TRELLIS](https://github.com/microsoft/TRELLIS?tab=readme-ov-file#-installation)'s  instructions to make your own `trellis` env

3. Install flask and rtree in `trellis` env

    ```python
    conda activate trellis
    pip install flask rtree
    ```

4. change the `config.yaml` to your path


# Start Frontend

```python
conda activate trellis
python app.py
```

# Dataset for eval

You may refer to this https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k

and do preprocessing following https://github.com/microsoft/TRELLIS/blob/main/DATASET.md#dataset-toolkits


# Evaluation
