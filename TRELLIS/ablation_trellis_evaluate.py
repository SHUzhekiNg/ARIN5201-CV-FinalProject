#!/usr/bin/env python3
"""
TRELLIS Ablation Study Evaluation Script.

Evaluates all ablation experiments using Hausdorff Distance metric
and generates comparison plots.

Author: Chris Chan & Copilot
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import trimesh

# Try to use GPU-accelerated distance computation
USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        from pytorch3d.ops import knn_points
        USE_GPU = True
        print("[INFO] Using PyTorch3D GPU acceleration for Hausdorff distance")
except ImportError:
    pass

if not USE_GPU:
    try:
        import point_cloud_utils as pcu
        print("[INFO] Using point_cloud_utils for fast Hausdorff distance")
    except ImportError:
        print("[WARNING] Neither pytorch3d nor point_cloud_utils available, using slow trimesh")


def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def load_metadata_mapping(metadata_file: str) -> Dict[str, str]:
    """Load metadata.csv and create object_name -> sha256 mapping."""
    df = pd.read_csv(metadata_file)
    name_to_sha256 = {}
    sha256_to_name = {}
    for _, row in df.iterrows():
        sha256 = str(row['sha256'])
        file_id = str(row['file_identifier'])
        object_name = file_id.split('/')[-1].replace('.blend', '')
        name_to_sha256[object_name] = sha256
        sha256_to_name[sha256] = object_name
    return name_to_sha256, sha256_to_name


def compute_hausdorff_distance(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: int = 100000
) -> Dict[str, float]:
    """Compute symmetric Hausdorff distance between two meshes."""
    # Sample points from both meshes
    points1, _ = trimesh.sample.sample_surface(mesh1, num_samples)
    points2, _ = trimesh.sample.sample_surface(mesh2, num_samples)
    
    if USE_GPU:
        # GPU-accelerated using PyTorch3D
        import torch
        from pytorch3d.ops import knn_points
        
        device = torch.device('cuda')
        p1 = torch.tensor(points1, dtype=torch.float32, device=device).unsqueeze(0)
        p2 = torch.tensor(points2, dtype=torch.float32, device=device).unsqueeze(0)
        
        # KNN with k=1 gives closest point distances
        knn1 = knn_points(p1, p2, K=1)
        knn2 = knn_points(p2, p1, K=1)
        
        dist1_to_2 = torch.sqrt(knn1.dists.squeeze()).cpu().numpy()
        dist2_to_1 = torch.sqrt(knn2.dists.squeeze()).cpu().numpy()
        
        hd_1_to_2 = float(np.max(dist1_to_2))
        hd_2_to_1 = float(np.max(dist2_to_1))
    else:
        # CPU fallback using scipy KDTree (faster than trimesh)
        from scipy.spatial import cKDTree
        
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        dist1_to_2, _ = tree2.query(points1, k=1)
        dist2_to_1, _ = tree1.query(points2, k=1)
        
        hd_1_to_2 = float(np.max(dist1_to_2))
        hd_2_to_1 = float(np.max(dist2_to_1))
    
    symmetric_hd = max(hd_1_to_2, hd_2_to_1)
    
    return {
        "hd_pred_to_gt": hd_1_to_2,
        "hd_gt_to_pred": hd_2_to_1,
        "symmetric_hausdorff": float(symmetric_hd),
    }


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh to unit sphere centered at origin."""
    mesh = mesh.copy()
    center = mesh.vertices.mean(axis=0)
    mesh.vertices -= center
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if scale > 0:
        mesh.vertices /= scale
    return mesh


def evaluate_experiment(
    experiment_dir: Path,
    gt_base: str,
    name_to_sha256: Dict[str, str],
    num_samples: int = 100000,
) -> Dict[str, any]:
    """Evaluate a single ablation experiment."""
    results = {}
    hd_values = []
    
    glb_files = list(experiment_dir.glob("*.glb"))
    log(f"  Found {len(glb_files)} GLB files in {experiment_dir.name}")
    
    for glb_path in tqdm(glb_files, desc=f"  Evaluating {experiment_dir.name}"):
        object_name = glb_path.stem
        sha256 = name_to_sha256.get(object_name)
        
        if not sha256:
            continue
        
        # Find ground truth - mesh.ply is in renders/{sha256}/ directory
        gt_path = Path(gt_base) / sha256 / "mesh.ply"
        if not gt_path.exists():
            gt_path = Path(gt_base) / f"{sha256}.ply"
        if not gt_path.exists():
            gt_path = Path(gt_base) / sha256 / "mesh.glb"
        if not gt_path.exists():
            continue
        
        try:
            # Load meshes
            pred_mesh = trimesh.load(str(glb_path), force='mesh')
            gt_mesh = trimesh.load(str(gt_path), force='mesh')
            
            # Normalize
            pred_mesh = normalize_mesh(pred_mesh)
            gt_mesh = normalize_mesh(gt_mesh)
            
            # Compute Hausdorff
            hd_result = compute_hausdorff_distance(pred_mesh, gt_mesh, num_samples)
            
            results[object_name] = {
                "success": True,
                **hd_result,
            }
            hd_values.append(hd_result["symmetric_hausdorff"])
            
        except Exception as e:
            results[object_name] = {
                "success": False,
                "error": str(e),
            }
    
    # Compute summary statistics
    if hd_values:
        summary = {
            "count": len(hd_values),
            "mean": float(np.mean(hd_values)),
            "std": float(np.std(hd_values)),
            "median": float(np.median(hd_values)),
            "min": float(np.min(hd_values)),
            "max": float(np.max(hd_values)),
        }
    else:
        summary = {"count": 0}
    
    return {
        "results": results,
        "summary": summary,
    }


def plot_ablation_comparison(
    all_results: Dict[str, Dict],
    output_path: str,
    baseline_name: str = "baseline",
):
    """Plot comparison of all ablation experiments - separate plots for Steps and CFG."""
    import matplotlib.pyplot as plt
    
    # Color palette
    colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2']
    
    # Separate Steps and CFG experiments
    steps_results = {}
    cfg_results = {}
    
    for name, result in all_results.items():
        if name.startswith("steps_"):
            steps_results[name] = result
        elif name.startswith("cfg_"):
            cfg_results[name] = result
    
    # ========== Plot 1: Steps Ablation ==========
    if steps_results:
        # Sort by steps number
        sorted_steps = sorted(steps_results.items(), key=lambda x: int(x[0].split('_')[1]))
        
        data = []
        labels = []
        for name, result in sorted_steps:
            hd_values = [
                r["symmetric_hausdorff"] 
                for r in result["results"].values() 
                if r.get("success", False)
            ]
            if hd_values:
                data.append(hd_values)
                steps_num = name.split('_')[1]
                labels.append(f"Steps={steps_num}")
        
        if data:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                            medianprops=dict(color='#333333', linewidth=2),
                            whiskerprops=dict(color='#666666', linewidth=1.5),
                            capprops=dict(color='#666666', linewidth=1.5))
            
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.85)
                patch.set_edgecolor('#333333')
                patch.set_linewidth(1.5)
            
            means = [np.mean(d) for d in data]
            ax.scatter(range(1, len(data)+1), means, marker='D', color='#FA7F6F', 
                       s=80, zorder=5, edgecolor='white', linewidth=1.5, label='Mean')
            
            for i, (d, mean) in enumerate(zip(data, means), 1):
                ax.annotate(f'{mean:.3f}', xy=(i, max(d) + 0.02), ha='center', 
                            fontsize=9, fontweight='bold')
            
            ax.set_ylabel('Hausdorff Distance (↓ lower is better)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sampling Steps', fontsize=12, fontweight='bold')
            ax.set_title('TRELLIS Ablation: Effect of Sampling Steps\n(Fixed CFG=7.5)', 
                         fontsize=14, fontweight='bold', pad=15)
            
            ax.yaxis.grid(True, linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            steps_plot_path = output_path.replace('.png', '_steps.png')
            plt.savefig(steps_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(steps_plot_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
            plt.close()
            log(f"Saved Steps ablation plot: {steps_plot_path}")
    
    # ========== Plot 2: CFG Ablation ==========
    # Include steps_30 as cfg_7.5 reference
    if "steps_30" in steps_results:
        cfg_results["cfg_7.5"] = steps_results["steps_30"]
    
    if cfg_results:
        # Sort by CFG value
        def get_cfg_value(name):
            cfg_str = name.split('_')[1]
            return float(cfg_str)
        
        sorted_cfg = sorted(cfg_results.items(), key=lambda x: get_cfg_value(x[0]))
        
        data = []
        labels = []
        for name, result in sorted_cfg:
            hd_values = [
                r["symmetric_hausdorff"] 
                for r in result["results"].values() 
                if r.get("success", False)
            ]
            if hd_values:
                data.append(hd_values)
                cfg_val = name.split('_')[1]
                labels.append(f"CFG={cfg_val}")
        
        if data:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                            medianprops=dict(color='#333333', linewidth=2),
                            whiskerprops=dict(color='#666666', linewidth=1.5),
                            capprops=dict(color='#666666', linewidth=1.5))
            
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.85)
                patch.set_edgecolor('#333333')
                patch.set_linewidth(1.5)
            
            means = [np.mean(d) for d in data]
            ax.scatter(range(1, len(data)+1), means, marker='D', color='#FA7F6F', 
                       s=80, zorder=5, edgecolor='white', linewidth=1.5, label='Mean')
            
            for i, (d, mean) in enumerate(zip(data, means), 1):
                ax.annotate(f'{mean:.3f}', xy=(i, max(d) + 0.02), ha='center', 
                            fontsize=9, fontweight='bold')
            
            ax.set_ylabel('Hausdorff Distance (↓ lower is better)', fontsize=12, fontweight='bold')
            ax.set_xlabel('CFG Strength', fontsize=12, fontweight='bold')
            ax.set_title('TRELLIS Ablation: Effect of CFG Strength\n(Fixed Steps=30)', 
                         fontsize=14, fontweight='bold', pad=15)
            
            ax.yaxis.grid(True, linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            cfg_plot_path = output_path.replace('.png', '_cfg.png')
            plt.savefig(cfg_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(cfg_plot_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
            plt.close()
            log(f"Saved CFG ablation plot: {cfg_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='TRELLIS Ablation Evaluation')
    parser.add_argument('--ablation-base', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/ablation_trellis',
                        help='Base directory containing ablation experiment outputs')
    parser.add_argument('--baseline-dir', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/Trellis',
                        help='Directory containing baseline TRELLIS outputs')
    parser.add_argument('--gt-base', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/datasets/Toys2h/renders',
                        help='Base directory for ground truth meshes')
    parser.add_argument('--metadata-file', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/datasets/Toys2h/metadata.csv',
                        help='Path to metadata.csv')
    parser.add_argument('--output', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/ablation_trellis/ablation_evaluation.json',
                        help='Output JSON file for evaluation results')
    parser.add_argument('--num-samples', type=int, default=100000,
                        help='Number of surface samples for Hausdorff distance')
    
    args = parser.parse_args()
    
    # Load metadata
    log("Loading metadata mapping...")
    name_to_sha256, sha256_to_name = load_metadata_mapping(args.metadata_file)
    log(f"Loaded {len(name_to_sha256)} mappings")
    
    ablation_base = Path(args.ablation_base)
    all_results = {}
    
    # Evaluate baseline first
    baseline_dir = Path(args.baseline_dir)
    if baseline_dir.exists():
        log("\n📊 Evaluating Baseline (steps=50, ss_cfg=7.5, slat_cfg=3.0)...")
        all_results["baseline"] = evaluate_experiment(
            baseline_dir, args.gt_base, name_to_sha256, args.num_samples
        )
        mean_val = all_results['baseline']['summary'].get('mean')
        log(f"  Baseline HD: mean={mean_val:.4f}" if mean_val is not None else "  Baseline HD: mean=N/A")
    
    # Evaluate each ablation experiment
    for exp_dir in sorted(ablation_base.iterdir()):
        if exp_dir.is_dir() and (exp_dir.name.startswith("steps_") or exp_dir.name.startswith("cfg_")):
            log(f"\n📊 Evaluating {exp_dir.name}...")
            all_results[exp_dir.name] = evaluate_experiment(
                exp_dir, args.gt_base, name_to_sha256, args.num_samples
            )
            summary = all_results[exp_dir.name]['summary']
            mean_val = summary.get('mean')
            log(f"  {exp_dir.name} HD: mean={mean_val:.4f}" if mean_val is not None else f"  {exp_dir.name} HD: mean=N/A")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to: {output_path}")
    
    # Generate comparison plot
    plot_path = str(output_path).replace('.json', '_boxplot.png')
    plot_ablation_comparison(all_results, plot_path)
    
    # Print summary table
    log("\n" + "="*80)
    log("ABLATION STUDY RESULTS SUMMARY")
    log("="*80)
    log(f"{'Experiment':<30} {'Count':<8} {'Mean HD':<10} {'Std':<10} {'Median':<10}")
    log("-"*80)
    
    for name, result in all_results.items():
        s = result['summary']
        if s.get('count', 0) > 0:
            log(f"{name:<30} {s['count']:<8} {s['mean']:<10.4f} {s['std']:<10.4f} {s['median']:<10.4f}")
    
    log("="*80)


if __name__ == '__main__':
    main()
