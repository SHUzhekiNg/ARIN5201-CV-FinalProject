#!/usr/bin/env python3
"""
Batch Hunyuan3D-2 inference script.

Reads image paths from extract_test_paths.txt (each path has 065.png),
runs Hunyuan3D-2 inference directly, and saves GLB files to evaluation/Hunyuan3D2.

Output naming: uses metadata.csv to map folder sha256 to file_identifier,
extracts object name (content between '/' and '.'), and names output as xxx.glb.
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
from datetime import datetime
from PIL import Image

# Import Hunyuan3D inference functions
sys.path.insert(0, str(Path(__file__).parent))
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

def log(msg: str):
    """Simple logging utility with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)




logger = None  # placeholder


def load_metadata_mapping(metadata_file: str) -> Dict[str, str]:
    """
    Load metadata.csv and create sha256 -> object_name mapping.
    
    object_name is extracted from file_identifier:
    file_identifier format: "category/object_xxx/object_xxx.blend"
    Extract: "object_xxx"
    """
    df = pd.read_csv(metadata_file)
    mapping = {}
    for _, row in df.iterrows():
        sha256 = str(row['sha256'])
        file_id = str(row['file_identifier'])
        # Extract object name: last part after '/' and before '.'
        object_name = file_id.split('/')[-1].replace('.blend', '')
        mapping[sha256] = object_name
    return mapping


def batch_infer_hunyuan(
    paths_file: str,
    metadata_file: str,
    output_base: str,
    model_path: str = "/disk2/licheng/models/Hunyuan3D-2/",
    cuda_visible_devices: Optional[str] = None,
) -> None:
    """
    Batch inference on Hunyuan3D-2 directly.
    
    Args:
        paths_file: path to txt file with ground truth folder paths (one per line)
        metadata_file: path to metadata.csv
        output_base: base directory to save GLB files (e.g., evaluation/Hunyuan3D2)
        model_path: path to Hunyuan3D-2 model directory
        cuda_visible_devices: CUDA devices (e.g., "0,1")
    """
    # Load metadata mapping
    log("Loading metadata mapping...")
    sha256_to_name = load_metadata_mapping(metadata_file)
    log(f"Loaded {len(sha256_to_name)} object names from metadata")
    
    # Read paths file
    with open(paths_file, 'r') as f:
        gt_paths = [line.strip() for line in f if line.strip()]
    log(f"Read {len(gt_paths)} ground truth paths")
    
    # Create output directory
    output_dir = Path(output_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")
    
    # Prepare results log
    results_log = output_dir / 'batch_inference.log'
    with open(results_log, 'w') as f:
        f.write(f"Batch Hunyuan3D-2 Inference Log\n")
        f.write(f"{'='*80}\n")
    
    # Filter tasks: skip already processed and invalid entries
    tasks = []
    for idx, gt_path_str in enumerate(gt_paths, 1):
        gt_path = Path(gt_path_str)
        sha256 = gt_path.name
        object_name = sha256_to_name.get(sha256)
        
        if not object_name:
            log(f"[{idx}/{len(gt_paths)}] SKIP: sha256 {sha256} not found in metadata")
            continue
        
        input_image = gt_path / '065.png'
        if not input_image.exists():
            log(f"[{idx}/{len(gt_paths)}] SKIP: image not found: {input_image}")
            continue
        
        output_glb = output_dir / f'{object_name}.glb'
        if output_glb.exists():
            log(f"[{idx}/{len(gt_paths)}] SKIP: {object_name} (GLB already exists)")
            continue
        
        tasks.append({
            'object_name': object_name,
            'input_image': str(input_image),  # Convert to string for pickling
            'output_glb': str(output_glb),
        })
    
    log(f"Prepared {len(tasks)} tasks to process")
    
    # Set CUDA devices if specified (must be done before loading models)
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        log(f"Set CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

    # Initialize pipelines
    log("Loading Hunyuan3D-2 pipelines...")
    try:
        rembg = BackgroundRemover()
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
        log("Pipelines loaded successfully.")
    except Exception as e:
        log(f"FATAL: Failed to load pipelines: {e}")
        return

    # Process tasks sequentially
    successful = 0
    failed = 0
    errors = []
    total_tasks = len(tasks)
    
    log(f"🎯 Starting batch inference for {total_tasks} tasks...")
    log("="*80)
    
    batch_start_time = time.time()
    elapsed_times = []
    
    for task_idx, task in enumerate(tasks, 1):
        object_name = task['object_name']
        input_image = Path(task['input_image'])
        output_glb = Path(task['output_glb'])
        
        start_time = time.time()
        print(f"🚀 Starting: {object_name}", flush=True)
        
        success = False
        error_msg = None
        
        try:
            # Run Hunyuan3D-2 inference directly
            print(f"⚙️  Running inference for {object_name}...", flush=True)
            
            try:
                # Load and preprocess image
                image = Image.open(str(input_image)).convert("RGBA")
                image = rembg(image)

                # Generate shape from image
                print(f"  {object_name}: Generating shape...", flush=True)
                mesh = pipeline_shapegen(image=image)[0]
                
                # Paint/texture the shape
                print(f"  {object_name}: Texturing shape...", flush=True)
                mesh = pipeline_texgen(mesh, image=image)
                
                # Export
                mesh.export(str(output_glb))
                
                # Check if GLB was generated
                if output_glb.exists():
                    success = True
                    print(f"  {object_name}: ✅ Shape generated successfully", flush=True)
                else:
                    error_msg = f"Output GLB not found at {output_glb}"
            
            except Exception as e:
                error_msg = f"Inference failed: {str(e)}"
                print(f"  {object_name}: Inference error: {error_msg}", flush=True)
        
        except Exception as e:
            error_msg = str(e)
        
        elapsed = time.time() - start_time
        elapsed_times.append(elapsed)
        
        if success:
            successful += 1
            # Calculate ETA
            avg_time = sum(elapsed_times) / len(elapsed_times)
            remaining = total_tasks - task_idx
            eta_seconds = avg_time * remaining
            eta_mins = int(eta_seconds / 60)
            
            progress_pct = int((task_idx / total_tasks) * 100)
            progress_bar = "█" * (progress_pct // 5) + "░" * (20 - progress_pct // 5)
            
            log(f"[{progress_bar}] {task_idx}/{total_tasks} | ✅ {successful}✓ {failed}✗ | ETA: {eta_mins}m | ✅ {object_name} ({elapsed:.1f}s)")
        else:
            failed += 1
            error_short = error_msg[:60] if error_msg else 'Unknown error'
            
            progress_pct = int((task_idx / total_tasks) * 100)
            progress_bar = "█" * (progress_pct // 5) + "░" * (20 - progress_pct // 5)
            
            log(f"[{progress_bar}] {task_idx}/{total_tasks} | ✅ {successful}✓ {failed}✗ | ❌ {object_name} - {error_short}")
            errors.append(f"{object_name}: {error_msg}")
    
    # Print summary
    total_time = time.time() - batch_start_time
    avg_time_per_task = total_time / total_tasks if total_tasks > 0 else 0
    
    log("\n" + "="*80)
    log("✨ Batch Inference Complete ✨")
    log("="*80)
    log(f"Total Tasks: {len(gt_paths)}")
    log(f"Successful: {successful} ✅")
    log(f"Failed: {failed} ❌")
    log(f"Total Time: {int(total_time // 60)}m {int(total_time % 60)}s")
    log(f"Avg Time/Task: {avg_time_per_task:.1f}s")
    log("="*80 + "\n")
    
    if errors:
        log("Errors:")
        for error in errors:
            log(f"  - {error}")
    
    # Save summary to log file
    with open(results_log, 'a') as f:
        f.write(f"\nTotal: {len(gt_paths)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        if errors:
            f.write(f"\nErrors:\n")
            for error in errors:
                f.write(f"  - {error}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch Hunyuan3D-2 inference on images directly.'
    )
    parser.add_argument('--paths-file', type=str,
                        default='../extract_test_paths.txt',
                        help='Path to txt file with ground truth folder paths')
    parser.add_argument('--metadata-file', type=str,
                        default='../datasets/Toys2h/metadata.csv',
                        help='Path to metadata.csv')
    parser.add_argument('--output', type=str,
                        default='../evaluation/Hunyuan3D2',
                        help='Output directory for GLB files')
    parser.add_argument('--model-path', type=str,
                        default='/disk2/licheng/models/Hunyuan3D-2/',
                        help='Path to Hunyuan3D-2 model directory')
    parser.add_argument('--cuda-devices', type=str, default=None,
                        help='CUDA visible devices (e.g., 0,1)')
    
    args = parser.parse_args()
    
    batch_infer_hunyuan(
        paths_file=args.paths_file,
        metadata_file=args.metadata_file,
        output_base=args.output,
        model_path=args.model_path,
        cuda_visible_devices=args.cuda_devices,
    )


if __name__ == '__main__':
    main()
