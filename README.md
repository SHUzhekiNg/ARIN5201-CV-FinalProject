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
- [ ] Basic Pipeline @ Licheng
    - [ ] Dataset Prepare
    - [ ] Hunyuan3D 2.0
    - [ ] Trellis3D

- [ ] Gradio Website

- [ ] Metrics(**Hausdorff Distance**)

- [ ] Blender API PostProcessing @ Kexiang



# Paper Works:

### Technical Report

6-page CVPR-style paper (references excluded from page count) containing:

- Team member names and responsibilities

- Project timeline with milestones

- Methodology and technical approach

- Experimental results and analysis

- Conclusion and references


### Presentation Slides

Slide deck for final presentation (15-minute presentation + 5 minutes Q&A).

Requirements

- The contributions of all team members should be highly relevant to the outcomes of the project.

- Report is supposed to follow CVPR formatting guidelines. You can find the official template at
https://github.com/cvpr-org/author-kit/releases

- Projects should address all original requirements. Implementations that handle challenges in
corresponding areas successfully are welcomed and can contribute to higher scores.

