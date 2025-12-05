#!/usr/bin/env python3
"""
TRELLIS Ablation Study - CLIP Score Evaluation Script.

Evaluates all ablation experiments using CLIP Score metric
(image-to-3D semantic alignment).

Author: Chris Chan & Copilot
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import torch
import clip
from PIL import Image
import trimesh
import pyrender

# Set up pyrender for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'


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


class MeshRenderer:
    """Render 3D mesh from multiple viewpoints using pyrender"""
    
    def __init__(self, resolution: int = 512):
        self.resolution = resolution
        
    def load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        """Load and normalize mesh"""
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # Normalize to unit sphere
        center = mesh.vertices.mean(axis=0)
        mesh.vertices -= center
        scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
        if scale > 0:
            mesh.vertices /= scale
            
        return mesh
    
    def render_views(
        self,
        mesh: trimesh.Trimesh,
        num_views: int = 8,
        elevations: List[float] = [0.0, 30.0],
        distance: float = 2.0,
        background_color: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0])
    ) -> List[Image.Image]:
        """Render mesh from multiple viewpoints"""
        images = []
        
        # Create scene
        scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=background_color)
        
        # Add mesh
        mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(mesh_pr)
        
        # Add directional light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=np.eye(4))
        
        # Camera setup
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
        
        # Renderer
        renderer = pyrender.OffscreenRenderer(self.resolution, self.resolution)
        
        try:
            azimuths = np.linspace(0, 360, num_views, endpoint=False)
            
            for elev in elevations:
                for azim in azimuths:
                    azim_rad = np.radians(azim)
                    elev_rad = np.radians(elev)
                    
                    x = distance * np.cos(elev_rad) * np.cos(azim_rad)
                    y = distance * np.cos(elev_rad) * np.sin(azim_rad)
                    z = distance * np.sin(elev_rad)
                    
                    eye = np.array([x, y, z])
                    target = np.array([0, 0, 0])
                    up = np.array([0, 0, 1])
                    
                    z_axis = eye - target
                    z_axis = z_axis / np.linalg.norm(z_axis)
                    x_axis = np.cross(up, z_axis)
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                    
                    camera_pose = np.eye(4)
                    camera_pose[:3, 0] = x_axis
                    camera_pose[:3, 1] = y_axis
                    camera_pose[:3, 2] = z_axis
                    camera_pose[:3, 3] = eye
                    
                    camera_node = scene.add(camera, pose=camera_pose)
                    color, _ = renderer.render(scene)
                    img = Image.fromarray(color)
                    images.append(img)
                    scene.remove_node(camera_node)
                    
        finally:
            renderer.delete()
            
        return images


class CLIPScoreEvaluator:
    """Evaluate condition alignment using CLIP"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model_name = model_name
        log(f"CLIP model loaded: {model_name} on {device}")
        
    def compute_score(
        self,
        condition_image: Image.Image,
        rendered_views: List[Image.Image],
    ) -> Dict[str, Any]:
        """Compute CLIP scores between condition image and rendered views"""
        
        with torch.no_grad():
            # Encode condition image
            condition_input = self.preprocess(condition_image).unsqueeze(0).to(self.device)
            condition_features = self.model.encode_image(condition_input)
            condition_features = condition_features / condition_features.norm(dim=-1, keepdim=True)
            
            # Encode rendered views
            scores = []
            for view in rendered_views:
                view_input = self.preprocess(view).unsqueeze(0).to(self.device)
                view_features = self.model.encode_image(view_input)
                view_features = view_features / view_features.norm(dim=-1, keepdim=True)
                
                score = (condition_features * view_features).sum().item()
                scores.append(score)
        
        scores = np.array(scores)
        
        return {
            "clip_score_mean": float(scores.mean()),
            "clip_score_std": float(scores.std()),
            "clip_score_min": float(scores.min()),
            "clip_score_max": float(scores.max()),
            "clip_score_median": float(np.median(scores)),
            "per_view_scores": [float(s) for s in scores],
        }


def evaluate_experiment(
    experiment_dir: Path,
    cond_base: str,
    name_to_sha256: Dict[str, str],
    clip_evaluator: CLIPScoreEvaluator,
    renderer: MeshRenderer,
    num_views: int = 8,
    elevations: List[float] = [0.0, 30.0],
) -> Dict[str, Any]:
    """Evaluate a single ablation experiment using CLIP score."""
    results = {}
    clip_scores = []
    
    glb_files = list(experiment_dir.glob("*.glb"))
    log(f"  Found {len(glb_files)} GLB files in {experiment_dir.name}")
    
    for glb_path in tqdm(glb_files, desc=f"  Evaluating {experiment_dir.name}"):
        object_name = glb_path.stem
        sha256 = name_to_sha256.get(object_name)
        
        if not sha256:
            continue
        
        # Find condition image (use 000.png as the input view)
        cond_path = Path(cond_base) / sha256 / "000.png"
        if not cond_path.exists():
            # Try other views
            cond_dir = Path(cond_base) / sha256
            if cond_dir.exists():
                pngs = list(cond_dir.glob("*.png"))
                if pngs:
                    cond_path = pngs[0]
                else:
                    continue
            else:
                continue
        
        try:
            # Load condition image
            cond_image = Image.open(cond_path).convert("RGB")
            
            # Load and render mesh
            mesh = renderer.load_mesh(str(glb_path))
            rendered_views = renderer.render_views(
                mesh, num_views=num_views, elevations=elevations
            )
            
            # Compute CLIP score
            clip_result = clip_evaluator.compute_score(cond_image, rendered_views)
            
            results[object_name] = {
                "success": True,
                **clip_result,
            }
            clip_scores.append(clip_result["clip_score_mean"])
            
        except Exception as e:
            results[object_name] = {
                "success": False,
                "error": str(e),
            }
    
    # Compute summary statistics
    if clip_scores:
        summary = {
            "count": len(clip_scores),
            "mean": float(np.mean(clip_scores)),
            "std": float(np.std(clip_scores)),
            "median": float(np.median(clip_scores)),
            "min": float(np.min(clip_scores)),
            "max": float(np.max(clip_scores)),
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
        sorted_steps = sorted(steps_results.items(), key=lambda x: int(x[0].split('_')[1]))
        
        data = []
        labels = []
        for name, result in sorted_steps:
            clip_values = [
                r["clip_score_mean"] 
                for r in result["results"].values() 
                if r.get("success", False)
            ]
            if clip_values:
                data.append(clip_values)
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
                ax.annotate(f'{mean:.3f}', xy=(i, min(d) - 0.015), ha='center', 
                            fontsize=9, fontweight='bold')
            
            ax.set_ylabel('CLIP Score (↑ higher is better)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sampling Steps', fontsize=12, fontweight='bold')
            ax.set_title('TRELLIS Ablation: Effect of Sampling Steps on CLIP Score\n(Fixed CFG=7.5)', 
                         fontsize=14, fontweight='bold', pad=15)
            
            ax.yaxis.grid(True, linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='lower right')
            
            plt.tight_layout()
            steps_plot_path = output_path.replace('.png', '_steps.png')
            plt.savefig(steps_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(steps_plot_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
            plt.close()
            log(f"Saved Steps ablation plot: {steps_plot_path}")
    
    # ========== Plot 2: CFG Ablation ==========
    if "steps_30" in steps_results:
        cfg_results["cfg_7.5"] = steps_results["steps_30"]
    
    if cfg_results:
        def get_cfg_value(name):
            cfg_str = name.split('_')[1]
            return float(cfg_str)
        
        sorted_cfg = sorted(cfg_results.items(), key=lambda x: get_cfg_value(x[0]))
        
        data = []
        labels = []
        for name, result in sorted_cfg:
            clip_values = [
                r["clip_score_mean"] 
                for r in result["results"].values() 
                if r.get("success", False)
            ]
            if clip_values:
                data.append(clip_values)
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
                ax.annotate(f'{mean:.3f}', xy=(i, min(d) - 0.015), ha='center', 
                            fontsize=9, fontweight='bold')
            
            ax.set_ylabel('CLIP Score (↑ higher is better)', fontsize=12, fontweight='bold')
            ax.set_xlabel('CFG Strength', fontsize=12, fontweight='bold')
            ax.set_title('TRELLIS Ablation: Effect of CFG Strength on CLIP Score\n(Fixed Steps=30)', 
                         fontsize=14, fontweight='bold', pad=15)
            
            ax.yaxis.grid(True, linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='lower right')
            
            plt.tight_layout()
            cfg_plot_path = output_path.replace('.png', '_cfg.png')
            plt.savefig(cfg_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(cfg_plot_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
            plt.close()
            log(f"Saved CFG ablation plot: {cfg_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='TRELLIS Ablation CLIP Score Evaluation')
    parser.add_argument('--ablation-base', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/ablation_trellis',
                        help='Base directory containing ablation experiment outputs')
    parser.add_argument('--baseline-dir', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/Trellis',
                        help='Directory containing baseline TRELLIS outputs')
    parser.add_argument('--cond-base', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/datasets/Toys2h/renders_cond',
                        help='Base directory for condition images')
    parser.add_argument('--metadata-file', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/datasets/Toys2h/metadata.csv',
                        help='Path to metadata.csv')
    parser.add_argument('--output', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/ablation_trellis/ablation_clip_evaluation.json',
                        help='Output JSON file for evaluation results')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-L/14'],
                        help='CLIP model variant')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Rendering resolution')
    parser.add_argument('--num-views', type=int, default=8,
                        help='Number of azimuth views per elevation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load metadata
    log("Loading metadata mapping...")
    name_to_sha256, sha256_to_name = load_metadata_mapping(args.metadata_file)
    log(f"Loaded {len(name_to_sha256)} mappings")
    
    # Initialize evaluators
    log("Initializing CLIP evaluator...")
    clip_evaluator = CLIPScoreEvaluator(model_name=args.clip_model, device=args.device)
    renderer = MeshRenderer(resolution=args.resolution)
    
    ablation_base = Path(args.ablation_base)
    all_results = {}
    
    # Evaluate baseline first
    baseline_dir = Path(args.baseline_dir)
    if baseline_dir.exists():
        log("\n📊 Evaluating Baseline (steps=50, ss_cfg=7.5, slat_cfg=3.0)...")
        all_results["baseline"] = evaluate_experiment(
            baseline_dir, args.cond_base, name_to_sha256,
            clip_evaluator, renderer, args.num_views
        )
        mean_val = all_results['baseline']['summary'].get('mean')
        log(f"  Baseline CLIP: mean={mean_val:.4f}" if mean_val is not None else "  Baseline CLIP: mean=N/A")
    
    # Evaluate each ablation experiment
    for exp_dir in sorted(ablation_base.iterdir()):
        if exp_dir.is_dir() and (exp_dir.name.startswith("steps_") or exp_dir.name.startswith("cfg_")):
            log(f"\n📊 Evaluating {exp_dir.name}...")
            all_results[exp_dir.name] = evaluate_experiment(
                exp_dir, args.cond_base, name_to_sha256,
                clip_evaluator, renderer, args.num_views
            )
            summary = all_results[exp_dir.name]['summary']
            mean_val = summary.get('mean')
            log(f"  {exp_dir.name} CLIP: mean={mean_val:.4f}" if mean_val is not None else f"  {exp_dir.name} CLIP: mean=N/A")
    
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
    log("ABLATION STUDY CLIP SCORE RESULTS SUMMARY")
    log("="*80)
    log(f"{'Experiment':<30} {'Count':<8} {'Mean':<10} {'Std':<10} {'Median':<10}")
    log("-"*80)
    
    for name, result in all_results.items():
        s = result['summary']
        if s.get('count', 0) > 0:
            log(f"{name:<30} {s['count']:<8} {s['mean']:<10.4f} {s['std']:<10.4f} {s['median']:<10.4f}")
    
    log("="*80)


if __name__ == '__main__':
    main()
