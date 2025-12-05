#!/usr/bin/env python3
"""
TRELLIS Ablation Study Inference Script.

Runs multiple ablation experiments with different parameter configurations.
Each experiment produces outputs in a separate directory.

Ablation Configurations:
- Baseline:   SS(steps=50, cfg=7.5), SLAT(steps=50, cfg=3.0)  [已完成]
- Ablation 1: SS(steps=12, cfg=7.5), SLAT(steps=12, cfg=3.0)  [默认步数]
- Ablation 2: SS(steps=50, cfg=3.0), SLAT(steps=50, cfg=3.0)  [低SS CFG]
- Ablation 3: SS(steps=50, cfg=15.0), SLAT(steps=50, cfg=3.0) [高SS CFG]
- Ablation 4: SS(steps=50, cfg=7.5), SLAT(steps=50, cfg=1.0)  [低SLAT CFG]
- Ablation 5: SS(steps=50, cfg=7.5), SLAT(steps=50, cfg=7.5)  [高SLAT CFG]

Author: Chris Chan & Copilot
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

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

os.environ['SPCONV_ALGO'] = 'native'

import pandas as pd
from PIL import Image

# Import TRELLIS pipeline
sys.path.insert(0, str(Path(__file__).parent))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ========================
# Ablation Configurations
# ========================
# 消融实验设计：
# 1. Steps 消融 (固定 cfg=7.5): steps = 10, 20, 30, 40, 50
# 2. CFG 消融 (固定 steps=30): cfg = 3, 5, 7.5, 10, 15
# 注意：SS 和 SLAT 使用相同的 steps 和 cfg 值

ABLATION_CONFIGS = {
    # ========== Steps 消融 (固定 cfg=7.5) ==========
    "steps_10": {
        "description": "Steps=10, CFG=7.5 (Steps ablation)",
        "ss_steps": 10,
        "ss_cfg": 7.5,
        "slat_steps": 10,
        "slat_cfg": 7.5,
    },
    "steps_20": {
        "description": "Steps=20, CFG=7.5 (Steps ablation)",
        "ss_steps": 20,
        "ss_cfg": 7.5,
        "slat_steps": 20,
        "slat_cfg": 7.5,
    },
    "steps_30": {
        "description": "Steps=30, CFG=7.5 (Steps ablation)",
        "ss_steps": 30,
        "ss_cfg": 7.5,
        "slat_steps": 30,
        "slat_cfg": 7.5,
    },
    "steps_40": {
        "description": "Steps=40, CFG=7.5 (Steps ablation)",
        "ss_steps": 40,
        "ss_cfg": 7.5,
        "slat_steps": 40,
        "slat_cfg": 7.5,
    },
    "steps_50": {
        "description": "Steps=50, CFG=7.5 (Steps ablation)",
        "ss_steps": 50,
        "ss_cfg": 7.5,
        "slat_steps": 50,
        "slat_cfg": 7.5,
    },
    # ========== CFG 消融 (固定 steps=30) ==========
    "cfg_3": {
        "description": "Steps=30, CFG=3.0 (CFG ablation)",
        "ss_steps": 30,
        "ss_cfg": 3.0,
        "slat_steps": 30,
        "slat_cfg": 3.0,
    },
    "cfg_5": {
        "description": "Steps=30, CFG=5.0 (CFG ablation)",
        "ss_steps": 30,
        "ss_cfg": 5.0,
        "slat_steps": 30,
        "slat_cfg": 5.0,
    },
    # cfg_7.5 与 steps_30 相同，跳过
    "cfg_10": {
        "description": "Steps=30, CFG=10.0 (CFG ablation)",
        "ss_steps": 30,
        "ss_cfg": 10.0,
        "slat_steps": 30,
        "slat_cfg": 10.0,
    },
    "cfg_15": {
        "description": "Steps=30, CFG=15.0 (CFG ablation)",
        "ss_steps": 30,
        "ss_cfg": 15.0,
        "slat_steps": 30,
        "slat_cfg": 15.0,
    },
}


def load_metadata_mapping(metadata_file: str) -> Dict[str, str]:
    """Load metadata.csv and create sha256 -> object_name mapping."""
    df = pd.read_csv(metadata_file)
    mapping = {}
    for _, row in df.iterrows():
        sha256 = str(row['sha256'])
        file_id = str(row['file_identifier'])
        object_name = file_id.split('/')[-1].replace('.blend', '')
        mapping[sha256] = object_name
    return mapping


def run_ablation_experiment(
    pipeline: TrellisImageTo3DPipeline,
    tasks: List[Dict],
    config_name: str,
    config: Dict,
    output_base: str,
) -> Tuple[int, int, List[str]]:
    """
    Run a single ablation experiment.
    
    Returns:
        (successful, failed, errors)
    """
    output_dir = Path(output_base) / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log(f"\n{'='*80}")
    log(f"🧪 Ablation Experiment: {config_name}")
    log(f"   Description: {config['description']}")
    log(f"   SS: steps={config['ss_steps']}, cfg={config['ss_cfg']}")
    log(f"   SLAT: steps={config['slat_steps']}, cfg={config['slat_cfg']}")
    log(f"   Output: {output_dir}")
    log(f"{'='*80}")
    
    successful = 0
    failed = 0
    errors = []
    
    # Filter already processed
    tasks_to_run = []
    for task in tasks:
        output_glb = output_dir / f"{task['object_name']}.glb"
        if output_glb.exists():
            log(f"  SKIP: {task['object_name']} (already exists)")
            successful += 1  # Count as successful
            continue
        tasks_to_run.append(task)
    
    if not tasks_to_run:
        log(f"  All {len(tasks)} tasks already completed!")
        return successful, failed, errors
    
    log(f"  Processing {len(tasks_to_run)} tasks...")
    
    batch_start = time.time()
    elapsed_times = []
    
    for idx, task in enumerate(tasks_to_run, 1):
        object_name = task['object_name']
        input_image = task['input_image']
        output_glb = output_dir / f"{object_name}.glb"
        
        start_time = time.time()
        
        try:
            image = Image.open(input_image)
            
            outputs = pipeline.run(
                image,
                seed=1,
                sparse_structure_sampler_params={
                    "steps": config['ss_steps'],
                    "cfg_strength": config['ss_cfg'],
                },
                slat_sampler_params={
                    "steps": config['slat_steps'],
                    "cfg_strength": config['slat_cfg'],
                },
            )
            
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(str(output_glb))
            
            elapsed = time.time() - start_time
            elapsed_times.append(elapsed)
            successful += 1
            
            # Progress
            avg_time = sum(elapsed_times) / len(elapsed_times)
            remaining = len(tasks_to_run) - idx
            eta_mins = int((avg_time * remaining) / 60)
            
            log(f"  [{idx}/{len(tasks_to_run)}] ✅ {object_name} ({elapsed:.1f}s) | ETA: {eta_mins}m")
            
        except Exception as e:
            elapsed = time.time() - start_time
            failed += 1
            error_msg = f"{object_name}: {str(e)[:100]}"
            errors.append(error_msg)
            log(f"  [{idx}/{len(tasks_to_run)}] ❌ {object_name} - {str(e)[:50]}")
    
    total_time = time.time() - batch_start
    log(f"  Experiment complete: {successful}✅ {failed}❌ in {int(total_time//60)}m {int(total_time%60)}s")
    
    return successful, failed, errors


def main():
    parser = argparse.ArgumentParser(description='TRELLIS Ablation Study')
    parser.add_argument('--paths-file', type=str,
                        default='../extract_test_paths.txt',
                        help='Path to txt file with ground truth folder paths')
    parser.add_argument('--metadata-file', type=str,
                        default='../datasets/Toys2h/metadata.csv',
                        help='Path to metadata.csv')
    parser.add_argument('--output', type=str,
                        default='../evaluation/ablation_trellis',
                        help='Base output directory for ablation experiments')
    parser.add_argument('--model-name', type=str,
                        default='microsoft/TRELLIS-image-large',
                        help='TRELLIS model name or path')
    parser.add_argument('--cuda-devices', type=str, default=None,
                        help='CUDA visible devices')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Specific experiments to run (e.g., ablation_1_default_steps)')
    parser.add_argument('--list-experiments', action='store_true',
                        help='List available experiments and exit')
    
    args = parser.parse_args()
    
    # List experiments
    if args.list_experiments:
        print("\nAvailable Ablation Experiments:")
        print("="*60)
        for name, config in ABLATION_CONFIGS.items():
            print(f"\n  {name}:")
            print(f"    Description: {config['description']}")
            print(f"    SS: steps={config['ss_steps']}, cfg={config['ss_cfg']}")
            print(f"    SLAT: steps={config['slat_steps']}, cfg={config['slat_cfg']}")
        print("\n" + "="*60)
        return
    
    # Load metadata
    log("Loading metadata mapping...")
    sha256_to_name = load_metadata_mapping(args.metadata_file)
    log(f"Loaded {len(sha256_to_name)} object names")
    
    # Read paths
    with open(args.paths_file, 'r') as f:
        gt_paths = [line.strip() for line in f if line.strip()]
    log(f"Read {len(gt_paths)} ground truth paths")
    
    # Prepare tasks
    tasks = []
    for gt_path_str in gt_paths:
        gt_path = Path(gt_path_str)
        sha256 = gt_path.name
        object_name = sha256_to_name.get(sha256)
        
        if not object_name:
            continue
        
        input_image = gt_path / '065.png'
        if not input_image.exists():
            continue
        
        tasks.append({
            'object_name': object_name,
            'input_image': str(input_image),
        })
    
    log(f"Prepared {len(tasks)} valid tasks")
    
    # Determine which experiments to run
    if args.experiments:
        experiments = {k: v for k, v in ABLATION_CONFIGS.items() if k in args.experiments}
        if not experiments:
            log(f"ERROR: No valid experiments found in {args.experiments}")
            log(f"Available: {list(ABLATION_CONFIGS.keys())}")
            return
    else:
        experiments = ABLATION_CONFIGS
    
    # Load pipeline once
    log("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_name)
    pipeline.cuda()
    log("Pipeline loaded successfully")
    
    # Create output base directory
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    summary = {}
    total_start = time.time()
    
    for config_name, config in experiments.items():
        successful, failed, errors = run_ablation_experiment(
            pipeline=pipeline,
            tasks=tasks,
            config_name=config_name,
            config=config,
            output_base=str(output_base),
        )
        summary[config_name] = {
            'successful': successful,
            'failed': failed,
            'errors': errors,
            'config': config,
        }
    
    total_time = time.time() - total_start
    
    # Print summary
    log("\n" + "="*80)
    log("🎯 ABLATION STUDY SUMMARY")
    log("="*80)
    
    for name, result in summary.items():
        log(f"\n  {name}:")
        log(f"    {result['config']['description']}")
        log(f"    Success: {result['successful']} | Failed: {result['failed']}")
    
    log(f"\nTotal Time: {int(total_time//60)}m {int(total_time%60)}s")
    log("="*80)
    
    # Save summary to JSON
    summary_file = output_base / "ablation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()
