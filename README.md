# 3D Assets Generator


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

We provide a comprehensive evaluation suite to assess the quality of generated 3D models. The evaluation is divided into two main categories: **Reference-based** (Geometric Quality) and **Reference-free** (Semantic & Visual Quality).

## Metrics

### 1. Geometric Quality (Reference-based)
- **Hausdorff Distance**: Measures the maximum distance between a point on the generated mesh and the nearest point on the ground truth mesh. Lower is better.
  - **Script**: `hausdorff_eval.py`
  - **Usage**: Compares a generated `.glb` against a ground truth `.ply`.

### 2. Semantic & Visual Quality (Reference-free)
- **CLIP Score**: Measures the semantic alignment between the input (text/image) and the rendered views of the generated 3D model. Higher is better.
- **Aesthetic Score**: Evaluates the visual appeal of the rendered views using a pre-trained aesthetic predictor.
- **Multi-view Consistency**: Measures the structural coherence of the 3D model across different viewpoints.
- **Mesh Validity**: Checks for geometric integrity (watertightness, manifoldness, self-intersections).
  - **Script**: `evaluate.py`

## Usage

### Unified Evaluator (`evaluate.py`)
This script runs a battery of reference-free metrics including CLIP Score, Aesthetic Score, and Mesh Validity.

```bash
# Evaluate a single model
python evaluate.py --model_path path/to/model.glb --prompt "a red car" --output_dir results/

# Evaluate a directory of models
python evaluate.py --model_dir path/to/models/ --prompt_file prompts.json --output_dir results/
```

### Hausdorff Distance Evaluator (`hausdorff_eval.py`)
This script computes the symmetric Hausdorff distance between a prediction and a ground truth mesh.

```bash
python hausdorff_eval.py --pred path/to/pred.glb --gt path/to/gt.ply --output results.json
```

### Ablation Studies
For TRELLIS specific ablation studies, we provide specialized scripts:

- **Geometric Evaluation**: `TRELLIS/ablation_trellis_evaluate.py`
- **CLIP Score Evaluation**: `TRELLIS/ablation_trellis_clip_evaluate.py`

```bash
# Run geometric ablation evaluation
python TRELLIS/ablation_trellis_evaluate.py --results_dir outputs/ablation --gt_dir datasets/Toys4k/

# Run CLIP score ablation evaluation
python TRELLIS/ablation_trellis_clip_evaluate.py --results_dir outputs/ablation --metadata datasets/Toys4k/metadata.csv
```

# PostProcessing

We provide a Blender-based post-processing module to clean up and optimize the generated 3D meshes. This module handles mesh cleanup, geometry repair, and topology optimization.

## Features
- **Mesh Cleanup**: Removes duplicate vertices, loose geometry, and degenerate faces.
- **Geometry Repair**: Fixes normal directions, non-manifold edges, and fills holes.
- **Topology Optimization**: Simplifies mesh (Decimate) and applies surface smoothing.

## Usage

**Note**: This script requires Blender 3.0+ to be installed and accessible via the `blender` command.

### Single File
```bash
blender --background --python PostProcessing/PostProcessing.py -- --input model.glb --output processed.glb
```

### Batch Processing
```bash
blender --background --python PostProcessing/PostProcessing.py -- --batch --input ./models/ --output ./output/
```

### Advanced Options
You can tune the optimization parameters:

- `--merge-distance`: Threshold for merging vertices (default: 0.0001)
- `--decimate-ratio`: Ratio for mesh simplification (0-1, default: 0.8)
- `--smooth-iterations`: Number of smoothing iterations (default: 5)
- `--no-smooth`: Disable smoothing
- `--no-decimate`: Disable decimation

**Example for conservative optimization (recommended for TRELLIS):**
```bash
blender --background --python PostProcessing/PostProcessing.py -- --batch --input TRELLIS_output --output post_processed --merge-distance 0.00001 --no-smooth --no-decimate
```

