#!/usr/bin/env python3
"""
Batch TRELLIS inference script.

Reads image paths from extract_test_paths.txt (each path has 065.png),
runs TRELLIS inference directly, and saves GLB files to evaluation/TRELLIS.

Output naming: uses metadata.csv to map folder sha256 to file_identifier,
extracts object name (content between '/' and '.'), and names output as xxx.glb.
"""

import os
import sys
import argparse

# Parse --cuda-devices BEFORE any CUDA-related imports
def _parse_cuda_devices():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cuda-devices', type=str, default=None)
    args, _ = parser.parse_known_args()
    return args.cuda_devices

_cuda_devices = _parse_cuda_devices()
if _cuda_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_devices
    print(f"[Init] Set CUDA_VISIBLE_DEVICES={_cuda_devices}")

# Set environment variables before importing other modules
os.environ['SPCONV_ALGO'] = 'native'  # 'native' is recommended for single runs

import shutil
import time
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from datetime import datetime
from PIL import Image

# Import TRELLIS pipeline (CUDA init happens here)
sys.path.insert(0, str(Path(__file__).parent))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def log(msg: str):
    """Simple logging utility with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


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


def batch_infer_trellis(
    paths_file: str,
    metadata_file: str,
    output_base: str,
    model_name: str = "microsoft/TRELLIS-image-large",
    cuda_visible_devices: Optional[str] = None,
) -> None:
    """
    Batch inference on TRELLIS directly.
    
    Args:
        paths_file: path to txt file with ground truth folder paths (one per line)
        metadata_file: path to metadata.csv
        output_base: base directory to save GLB files (e.g., evaluation/TRELLIS)
        model_name: TRELLIS model name or path
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
        f.write(f"Batch TRELLIS Inference Log\n")
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
            'input_image': str(input_image),
            'output_glb': str(output_glb),
        })
    
    log(f"Prepared {len(tasks)} tasks to process")
    
    # Set CUDA devices if specified (must be done before loading models)
    # Note: Already set at script startup, but log it here for consistency
    if cuda_visible_devices:
        log(f"Using CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

    # Initialize pipeline
    log("Loading TRELLIS pipeline...")
    try:
        pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        pipeline.cuda()
        log("Pipeline loaded successfully.")
    except Exception as e:
        log(f"FATAL: Failed to load pipeline: {e}")
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
            # Run TRELLIS inference directly
            print(f"⚙️  Running inference for {object_name}...", flush=True)
            
            try:
                # Load image
                image = Image.open(str(input_image))

                # Run pipeline with step=50 to align with Hunyuan3D
                print(f"  {object_name}: Generating 3D model...", flush=True)
                outputs = pipeline.run(
                    image,
                    seed=1,
                    sparse_structure_sampler_params={
                        "steps": 50,
                        "cfg_strength": 7.5,
                    },
                    slat_sampler_params={
                        "steps": 50,
                        "cfg_strength": 3,
                    },
                )
                
                # Export GLB
                print(f"  {object_name}: Exporting GLB...", flush=True)
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.95,
                    texture_size=1024,
                )
                glb.export(str(output_glb))
                
                # Check if GLB was generated
                if output_glb.exists():
                    success = True
                    print(f"  {object_name}: ✅ GLB generated successfully", flush=True)
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
        description='Batch TRELLIS inference on images directly.'
    )
    parser.add_argument('--paths-file', type=str,
                        default='../extract_test_paths.txt',
                        help='Path to txt file with ground truth folder paths')
    parser.add_argument('--metadata-file', type=str,
                        default='../datasets/Toys2h/metadata.csv',
                        help='Path to metadata.csv')
    parser.add_argument('--output', type=str,
                        default='../evaluation/TRELLIS',
                        help='Output directory for GLB files')
    parser.add_argument('--model-name', type=str,
                        default='microsoft/TRELLIS-image-large',
                        help='TRELLIS model name or path')
    parser.add_argument('--cuda-devices', type=str, default=None,
                        help='CUDA visible devices (e.g., 0,1)')
    
    args = parser.parse_args()
    
    batch_infer_trellis(
        paths_file=args.paths_file,
        metadata_file=args.metadata_file,
        output_base=args.output,
        model_name=args.model_name,
        cuda_visible_devices=args.cuda_devices,
    )


if __name__ == '__main__':
    main()
