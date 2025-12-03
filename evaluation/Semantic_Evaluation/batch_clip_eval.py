#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_clip_eval.py
Batch CLIP Score Evaluation for Hunyuan3D-2 and TRELLIS generated 3D models.

Computes CLIP score between input condition images (065.png) and 
rendered views of generated 3D models.

Author: Chris Chan & Copilot
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import clip
import trimesh
import pyrender


def log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# -----------------------------
# Metadata Mapping
# -----------------------------
def load_metadata_mapping(metadata_file: str) -> Dict[str, str]:
    """
    Load metadata.csv and create object_name -> sha256 mapping.
    (Reverse mapping for looking up sha256 from object_name)
    """
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


# -----------------------------
# Mesh Renderer
# -----------------------------
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
        
        scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=background_color)
        mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(mesh_pr)
        
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=np.eye(4))
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
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


# -----------------------------
# CLIP Score Evaluator
# -----------------------------
class CLIPScoreEvaluator:
    """Evaluate condition alignment using CLIP"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model_name = model_name
        
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


# -----------------------------
# Batch Evaluation
# -----------------------------
def batch_clip_evaluate(
    paths_file: str,
    metadata_file: str,
    hunyuan_dir: str,
    trellis_dir: str,
    output_json: str,
    clip_model: str = "ViT-B/32",
    num_views: int = 8,
    elevations: List[float] = [0.0, 30.0],
    resolution: int = 512,
    device: str = "cuda",
    post: bool = False,
):
    """
    Batch evaluate CLIP scores for Hunyuan3D-2 and TRELLIS models.
    
    Args:
        paths_file: Path to extract_test_paths.txt
        metadata_file: Path to metadata.csv
        hunyuan_dir: Directory containing Hunyuan3D-2 GLB files
        trellis_dir: Directory containing TRELLIS GLB files
        output_json: Path to save JSON results
        clip_model: CLIP model variant
        num_views: Number of azimuth views per elevation
        elevations: List of elevation angles
        resolution: Rendering resolution
        device: cuda or cpu
        post: If True, add "post_" prefix to GLB filenames
    """
    log("Loading metadata mapping...")
    name_to_sha256, sha256_to_name = load_metadata_mapping(metadata_file)
    log(f"Loaded {len(name_to_sha256)} mappings")
    
    # Read paths file to get sha256 folders
    with open(paths_file, 'r') as f:
        gt_paths = [line.strip() for line in f if line.strip()]
    log(f"Read {len(gt_paths)} ground truth paths")
    
    # Initialize evaluators
    log(f"Initializing CLIP evaluator ({clip_model})...")
    clip_evaluator = CLIPScoreEvaluator(model_name=clip_model, device=device)
    renderer = MeshRenderer(resolution=resolution)
    
    hunyuan_dir = Path(hunyuan_dir)
    trellis_dir = Path(trellis_dir)
    
    results = {
        "config": {
            "clip_model": clip_model,
            "num_views": num_views,
            "elevations": elevations,
            "resolution": resolution,
            "total_rendered_views": num_views * len(elevations),
        },
        "samples": [],
        "summary": {}
    }
    
    hunyuan_scores = []
    trellis_scores = []
    
    log(f"Starting batch CLIP evaluation...")
    
    for gt_path_str in tqdm(gt_paths, desc="Evaluating"):
        gt_path = Path(gt_path_str)
        sha256 = gt_path.name
        object_name = sha256_to_name.get(sha256)
        
        if not object_name:
            log(f"SKIP: sha256 {sha256} not found in metadata")
            continue
        
        # Find input condition image (065.png)
        input_image_path = gt_path / '065.png'
        if not input_image_path.exists():
            log(f"SKIP: input image not found: {input_image_path}")
            continue
        
        # Find GLB files (add "post_" prefix if post=True)
        glb_prefix = 'post_' if post else ''
        hunyuan_glb = hunyuan_dir / f'{glb_prefix}{object_name}.glb'
        trellis_glb = trellis_dir / f'{glb_prefix}{object_name}.glb'
        
        sample_result = {
            "object_name": object_name,
            "sha256": sha256,
            "input_image": str(input_image_path),
            "hunyuan3d2": None,
            "trellis": None,
            "error": None,
        }
        
        try:
            # Load condition image
            condition_image = Image.open(input_image_path).convert("RGB")
            
            # Evaluate Hunyuan3D-2
            if hunyuan_glb.exists():
                try:
                    mesh = renderer.load_mesh(str(hunyuan_glb))
                    views = renderer.render_views(mesh, num_views=num_views, elevations=elevations)
                    clip_result = clip_evaluator.compute_score(condition_image, views)
                    sample_result["hunyuan3d2"] = clip_result
                    hunyuan_scores.append(clip_result["clip_score_mean"])
                except Exception as e:
                    sample_result["hunyuan3d2"] = {"error": str(e)}
            else:
                sample_result["hunyuan3d2"] = {"error": "GLB not found"}
            
            # Evaluate TRELLIS
            if trellis_glb.exists():
                try:
                    mesh = renderer.load_mesh(str(trellis_glb))
                    views = renderer.render_views(mesh, num_views=num_views, elevations=elevations)
                    clip_result = clip_evaluator.compute_score(condition_image, views)
                    sample_result["trellis"] = clip_result
                    trellis_scores.append(clip_result["clip_score_mean"])
                except Exception as e:
                    sample_result["trellis"] = {"error": str(e)}
            else:
                sample_result["trellis"] = {"error": "GLB not found"}
                
        except Exception as e:
            sample_result["error"] = str(e)
        
        results["samples"].append(sample_result)
    
    # Compute summary statistics
    hunyuan_scores = np.array(hunyuan_scores)
    trellis_scores = np.array(trellis_scores)
    
    results["summary"] = {
        "hunyuan3d2": {
            "num_evaluated": len(hunyuan_scores),
            "clip_score_mean": float(hunyuan_scores.mean()) if len(hunyuan_scores) > 0 else None,
            "clip_score_std": float(hunyuan_scores.std()) if len(hunyuan_scores) > 0 else None,
            "clip_score_min": float(hunyuan_scores.min()) if len(hunyuan_scores) > 0 else None,
            "clip_score_max": float(hunyuan_scores.max()) if len(hunyuan_scores) > 0 else None,
            "clip_score_median": float(np.median(hunyuan_scores)) if len(hunyuan_scores) > 0 else None,
        },
        "trellis": {
            "num_evaluated": len(trellis_scores),
            "clip_score_mean": float(trellis_scores.mean()) if len(trellis_scores) > 0 else None,
            "clip_score_std": float(trellis_scores.std()) if len(trellis_scores) > 0 else None,
            "clip_score_min": float(trellis_scores.min()) if len(trellis_scores) > 0 else None,
            "clip_score_max": float(trellis_scores.max()) if len(trellis_scores) > 0 else None,
            "clip_score_median": float(np.median(trellis_scores)) if len(trellis_scores) > 0 else None,
        },
    }
    
    # Save results
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    log(f"Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLIP Score Evaluation Summary")
    print("="*60)
    print(f"\nHunyuan3D-2 ({len(hunyuan_scores)} samples):")
    if len(hunyuan_scores) > 0:
        print(f"  Mean:   {hunyuan_scores.mean():.4f}")
        print(f"  Std:    {hunyuan_scores.std():.4f}")
        print(f"  Median: {np.median(hunyuan_scores):.4f}")
        print(f"  Range:  [{hunyuan_scores.min():.4f}, {hunyuan_scores.max():.4f}]")
    
    print(f"\nTRELLIS ({len(trellis_scores)} samples):")
    if len(trellis_scores) > 0:
        print(f"  Mean:   {trellis_scores.mean():.4f}")
        print(f"  Std:    {trellis_scores.std():.4f}")
        print(f"  Median: {np.median(trellis_scores):.4f}")
        print(f"  Range:  [{trellis_scores.min():.4f}, {trellis_scores.max():.4f}]")
    
    if len(hunyuan_scores) > 0 and len(trellis_scores) > 0:
        diff = trellis_scores.mean() - hunyuan_scores.mean()
        print(f"\nDifference (TRELLIS - Hunyuan3D-2): {diff:+.4f}")
        winner = "TRELLIS" if diff > 0 else "Hunyuan3D-2"
        print(f"Better semantic alignment: {winner}")
    
    print("="*60 + "\n")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch CLIP Score Evaluation')
    parser.add_argument('--paths-file', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/extract_test_paths.txt',
                        help='Path to extract_test_paths.txt')
    parser.add_argument('--metadata-file', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/datasets/Toys2h/metadata.csv',
                        help='Path to metadata.csv')
    parser.add_argument('--hunyuan-dir', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/Hunyuan3D2',
                        help='Directory containing Hunyuan3D-2 GLB files')
    parser.add_argument('--trellis-dir', type=str,
                        default='/disk2/licheng/code/ARIN5201-CV-FinalProject/evaluation/Trellis',
                        help='Directory containing TRELLIS GLB files')
    parser.add_argument('--output', type=str,
                        default='evaluation/Semantic_Evaluation/clip_batch_results.json',
                        help='Output JSON file path')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-L/14'],
                        help='CLIP model variant')
    parser.add_argument('--num-views', type=int, default=8,
                        help='Number of azimuth views per elevation')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Rendering resolution')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--post', action='store_true',
                        help='If set, add "post_" prefix to GLB filenames')
    
    args = parser.parse_args()
    
    batch_clip_evaluate(
        paths_file=args.paths_file,
        metadata_file=args.metadata_file,
        hunyuan_dir=args.hunyuan_dir,
        trellis_dir=args.trellis_dir,
        output_json=args.output,
        clip_model=args.clip_model,
        num_views=args.num_views,
        resolution=args.resolution,
        device=args.device,
        post=args.post,
    )


if __name__ == '__main__':
    main()
