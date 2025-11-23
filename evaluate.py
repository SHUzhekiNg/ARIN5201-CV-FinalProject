#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py
Unified 3D Model Quality Evaluator

Supports both text-to-3D and image-to-3D evaluation without ground truth.
Compatible with TRELLIS and Hunyuan3D-2 output formats.

Evaluation Metrics:
1. CLIP Score: Condition alignment (text/image → 3D model)
2. Mesh Validity: Geometric integrity (watertight, manifold, self-intersection)
3. Aesthetic Score: Visual appeal of rendered views
4. Multi-view Consistency: 3D coherence across different viewpoints
5. Geometric Statistics: Basic mesh properties

Author: Chris Chan & Copilot
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import clip
import numpy as np
import trimesh
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm
import pyrender
import torchvision.transforms as transforms
from transformers import pipeline
import lpips

warnings.filterwarnings("ignore")


# -----------------------------
# Utils
# -----------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{_now()}] {msg}", flush=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------
# 3D Model Renderer
# -----------------------------
class MeshRenderer:
    """Render 3D mesh from multiple viewpoints using pyrender"""
    
    def __init__(self, resolution: int = 512, use_egl: bool = True):
        """
        Args:
            resolution: Image resolution (square)
            use_egl: Use EGL for headless rendering (recommended for servers)
        """
        self.resolution = resolution
        
        # Set up pyrender with EGL for headless rendering
        if use_egl:
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        
        
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
        """
        Render mesh from multiple viewpoints
        
        Args:
            mesh: trimesh.Trimesh object
            num_views: Number of azimuth angles
            elevations: List of elevation angles in degrees
            distance: Camera distance from origin
            background_color: RGBA background color
            
        Returns:
            List of PIL Images
        """
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
            # Generate views
            azimuths = np.linspace(0, 360, num_views, endpoint=False)
            
            for elev in elevations:
                for azim in azimuths:
                    # Compute camera pose
                    azim_rad = np.radians(azim)
                    elev_rad = np.radians(elev)
                    
                    # Spherical to Cartesian
                    x = distance * np.cos(elev_rad) * np.cos(azim_rad)
                    y = distance * np.cos(elev_rad) * np.sin(azim_rad)
                    z = distance * np.sin(elev_rad)
                    
                    # Camera pose (look at origin)
                    eye = np.array([x, y, z])
                    target = np.array([0, 0, 0])
                    up = np.array([0, 0, 1])
                    
                    # Compute view matrix
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
                    
                    # Add camera to scene
                    camera_node = scene.add(camera, pose=camera_pose)
                    
                    # Render
                    color, _ = renderer.render(scene)
                    
                    # Convert to PIL Image
                    img = Image.fromarray(color)
                    images.append(img)
                    
                    # Remove camera for next view
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
        """
        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-L/14)
            device: cuda or cpu
        """
        
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model_name = model_name
        
    def compute_score(
        self,
        condition: Union[str, Image.Image],
        rendered_views: List[Image.Image],
        condition_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Compute CLIP scores between condition and rendered views
        
        Args:
            condition: Text prompt or input image
            rendered_views: List of rendered images from different viewpoints
            condition_type: "text", "image", or "auto"
            
        Returns:
            Dictionary with scores and statistics
        """
        
        # Auto-detect condition type
        if condition_type == "auto":
            if isinstance(condition, str):
                condition_type = "text"
            else:
                condition_type = "image"
        
        # Encode condition
        with torch.no_grad():
            if condition_type == "text":
                text_tokens = clip.tokenize([condition]).to(self.device)
                condition_features = self.model.encode_text(text_tokens)
            else:
                condition_input = self.preprocess(condition).unsqueeze(0).to(self.device)
                condition_features = self.model.encode_image(condition_input)
            
            condition_features = condition_features / condition_features.norm(dim=-1, keepdim=True)
            
            # Encode rendered views
            scores = []
            for view in rendered_views:
                view_input = self.preprocess(view).unsqueeze(0).to(self.device)
                view_features = self.model.encode_image(view_input)
                view_features = view_features / view_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
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
            "condition_type": condition_type,
            "clip_model": self.model_name,
        }


# -----------------------------
# Mesh Validity Checker
# -----------------------------
class MeshValidityChecker:
    """Check geometric validity of 3D mesh"""
    
    @staticmethod
    def check(mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Perform comprehensive mesh validity checks
        
        Args:
            mesh: trimesh.Trimesh object
            
        Returns:
            Dictionary with validity metrics
        """
        results = {}
        
        # Basic checks (convert numpy bool to Python bool)
        results["is_watertight"] = bool(mesh.is_watertight)
        results["is_volume"] = bool(mesh.is_volume)
        results["is_empty"] = bool(mesh.is_empty)
        
        # Euler characteristic (for closed surfaces: V - E + F = 2)
        results["euler_number"] = int(mesh.euler_number)
        
        # Check for degenerate faces
        face_areas = mesh.area_faces
        degenerate_faces = np.sum(face_areas < 1e-10)
        results["num_degenerate_faces"] = int(degenerate_faces)
        results["degenerate_face_ratio"] = float(degenerate_faces / len(mesh.faces))
        
        # Check for duplicate faces
        results["num_duplicate_faces"] = int(len(mesh.faces) - len(np.unique(np.sort(mesh.faces, axis=1), axis=0)))
        
        # Check for non-manifold edges (edges shared by more than 2 faces)
        try:
            edges_face_count = mesh.edges_face
            non_manifold = np.sum(edges_face_count > 2)
            results["num_non_manifold_edges"] = int(non_manifold)
            results["is_manifold"] = bool(non_manifold == 0)
        except Exception:
            results["num_non_manifold_edges"] = None
            results["is_manifold"] = None
        
        # Check for self-intersection (expensive, use sampling)
        try:
            results["has_self_intersection"] = bool(mesh.is_self_intersecting)
        except Exception:
            results["has_self_intersection"] = None
        
        # Overall validity score (0-1)
        validity_checks = [
            results["is_watertight"],
            results["is_manifold"] if results["is_manifold"] is not None else False,
            not results["has_self_intersection"] if results["has_self_intersection"] is not None else True,
            results["degenerate_face_ratio"] < 0.01,
        ]
        results["validity_score"] = float(sum(validity_checks) / len(validity_checks))
        
        return results


# -----------------------------
# Aesthetic Score Evaluator
# -----------------------------
class AestheticScoreEvaluator:
    """Evaluate aesthetic quality of rendered images"""
    
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: cuda or cpu
        """
        try:
            
            self.scorer = pipeline(
                "image-classification",
                model="cafeai/cafe_aesthetic",
                device=0 if device == "cuda" and torch.cuda.is_available() else -1
            )
            self.available = True
        except Exception as e:
            log(f"[WARN] Aesthetic scorer not available: {e}")
            self.available = False
    
    def compute_scores(self, images: List[Image.Image]) -> Dict[str, Any]:
        """
        Compute aesthetic scores for multiple images
        
        Args:
            images: List of PIL Images
            
        Returns:
            Dictionary with aesthetic scores
        """
        if not self.available:
            return {
                "aesthetic_score_mean": None,
                "aesthetic_score_std": None,
                "aesthetic_score_min": None,
                "aesthetic_score_max": None,
                "per_view_scores": None,
            }
        
        scores = []
        for img in images:
            result = self.scorer(img)
            # Extract score from result (model-dependent format)
            if isinstance(result, list) and len(result) > 0:
                score = result[0].get("score", 0.5)
            else:
                score = 0.5
            scores.append(score)
        
        scores = np.array(scores)
        
        return {
            "aesthetic_score_mean": float(scores.mean()),
            "aesthetic_score_std": float(scores.std()),
            "aesthetic_score_min": float(scores.min()),
            "aesthetic_score_max": float(scores.max()),
            "per_view_scores": [float(s) for s in scores],
        }


# -----------------------------
# Multi-view Consistency Evaluator
# -----------------------------
class ConsistencyEvaluator:
    """Evaluate 3D consistency across different viewpoints"""
    
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: cuda or cpu
        """
        try:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            self.device = device
            self.available = True
        except Exception as e:
            log(f"[WARN] LPIPS model not available: {e}")
            self.available = False
    
    def compute_consistency(self, images: List[Image.Image]) -> Dict[str, Any]:
        """
        Compute LPIPS between adjacent views
        
        Args:
            images: List of PIL Images (should be in sequential order)
            
        Returns:
            Dictionary with consistency metrics
        """
        if not self.available or len(images) < 2:
            return {
                "consistency_score_mean": None,
                "consistency_score_std": None,
                "note": "LPIPS not available or insufficient views"
            }
        
        
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        scores = []
        with torch.no_grad():
            for i in range(len(images) - 1):
                img1 = transform(images[i]).unsqueeze(0).to(self.device)
                img2 = transform(images[i + 1]).unsqueeze(0).to(self.device)
                
                distance = self.lpips_model(img1, img2).item()
                scores.append(distance)
        
        scores = np.array(scores)
        
        return {
            "consistency_score_mean": float(scores.mean()),
            "consistency_score_std": float(scores.std()),
            "consistency_score_min": float(scores.min()),
            "consistency_score_max": float(scores.max()),
            "note": "Lower LPIPS = better consistency",
        }


# -----------------------------
# Geometric Statistics
# -----------------------------
class GeometricStatsCalculator:
    """Calculate basic geometric statistics"""
    
    @staticmethod
    def compute(mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Compute geometric statistics
        
        Args:
            mesh: trimesh.Trimesh object
            
        Returns:
            Dictionary with geometric stats
        """
        stats = {
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "num_edges": len(mesh.edges),
            "surface_area": float(mesh.area),
            "bounding_box_volume": float(mesh.bounding_box.volume),
        }
        
        # Volume (only for watertight meshes)
        if mesh.is_volume:
            stats["volume"] = float(mesh.volume)
        else:
            stats["volume"] = None
        
        # Bounding box extents
        extents = mesh.extents
        stats["bounding_box_extents"] = {
            "x": float(extents[0]),
            "y": float(extents[1]),
            "z": float(extents[2]),
        }
        
        return stats


# -----------------------------
# Unified Evaluator
# -----------------------------
class Universal3DEvaluator:
    """Unified evaluator for text-to-3D and image-to-3D models"""
    
    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        resolution: int = 512,
        num_views: int = 8,
        elevations: List[float] = [0.0, 30.0],
        device: str = "cuda",
        use_aesthetic: bool = False,
        use_consistency: bool = False,
    ):
        """
        Args:
            clip_model: CLIP model variant
            resolution: Rendering resolution
            num_views: Number of azimuth angles per elevation
            elevations: List of elevation angles
            device: cuda or cpu
            use_aesthetic: Enable aesthetic scoring (requires extra models)
            use_consistency: Enable consistency evaluation (requires LPIPS)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.resolution = resolution
        self.num_views = num_views
        self.elevations = elevations
        
        log(f"Initializing evaluator on device: {self.device}")
        
        # Initialize components
        self.renderer = MeshRenderer(resolution=resolution, use_egl=True)
        self.clip_evaluator = CLIPScoreEvaluator(model_name=clip_model, device=self.device)
        self.validity_checker = MeshValidityChecker()
        self.stats_calculator = GeometricStatsCalculator()
        
        # Optional components
        if use_aesthetic:
            self.aesthetic_evaluator = AestheticScoreEvaluator(device=self.device)
        else:
            self.aesthetic_evaluator = None
            
        if use_consistency:
            self.consistency_evaluator = ConsistencyEvaluator(device=self.device)
        else:
            self.consistency_evaluator = None
    
    def evaluate(
        self,
        model_path: str,
        condition: Union[str, Image.Image, None] = None,
        condition_type: str = "auto",
        save_renders: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a 3D model
        
        Args:
            model_path: Path to 3D model file (GLB, OBJ, PLY)
            condition: Text prompt or input image (optional)
            condition_type: "text", "image", or "auto"
            save_renders: Save rendered views to disk
            output_dir: Directory to save renders (if save_renders=True)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        start_time = time.time()
        
        log(f"Evaluating model: {model_path}")
        
        # Load mesh
        try:
            mesh = self.renderer.load_mesh(model_path)
        except Exception as e:
            return {"error": f"Failed to load mesh: {e}"}
        
        # Geometric statistics
        log("Computing geometric statistics...")
        geometric_stats = self.stats_calculator.compute(mesh)
        
        # Mesh validity
        log("Checking mesh validity...")
        validity_results = self.validity_checker.check(mesh)
        
        # Render views
        log(f"Rendering {self.num_views} views at {len(self.elevations)} elevations...")
        render_start = time.time()
        rendered_views = self.renderer.render_views(
            mesh,
            num_views=self.num_views,
            elevations=self.elevations
        )
        render_time = time.time() - render_start
        
        # Save renders if requested
        if save_renders and output_dir:
            output_path = Path(output_dir)
            ensure_dir(output_path)
            stem = Path(model_path).stem
            
            # Save as grid
            grid_cols = self.num_views
            grid_rows = len(self.elevations)
            grid_img = Image.new('RGB', 
                                (self.resolution * grid_cols, self.resolution * grid_rows))
            
            for idx, img in enumerate(rendered_views):
                row = idx // grid_cols
                col = idx % grid_cols
                grid_img.paste(img, (col * self.resolution, row * self.resolution))
            
            grid_path = output_path / f"{stem}_renders.png"
            grid_img.save(grid_path)
            log(f"Saved renders to: {grid_path}")
        
        # CLIP Score (if condition provided)
        clip_results = {}
        if condition is not None:
            log("Computing CLIP scores...")
            clip_results = self.clip_evaluator.compute_score(
                condition, rendered_views, condition_type
            )
        
        # Aesthetic Score (optional)
        aesthetic_results = {}
        if self.aesthetic_evaluator is not None:
            log("Computing aesthetic scores...")
            aesthetic_results = self.aesthetic_evaluator.compute_scores(rendered_views)
        
        # Consistency Score (optional)
        consistency_results = {}
        if self.consistency_evaluator is not None:
            log("Computing multi-view consistency...")
            consistency_results = self.consistency_evaluator.compute_consistency(rendered_views)
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            "model_path": str(model_path),
            "model_name": Path(model_path).name,
            "condition": str(condition) if isinstance(condition, str) else (
                "image" if condition is not None else None
            ),
            "evaluation_time": {
                "total": float(total_time),
                "rendering": float(render_time),
            },
            "rendering_config": {
                "num_views": self.num_views,
                "elevations": self.elevations,
                "resolution": self.resolution,
                "total_rendered_images": len(rendered_views),
            },
            "geometric_stats": geometric_stats,
            "mesh_validity": validity_results,
            "clip_score": clip_results if clip_results else None,
            "aesthetic_score": aesthetic_results if aesthetic_results else None,
            "consistency": consistency_results if consistency_results else None,
        }
        
        log(f"Evaluation complete in {total_time:.2f}s")
        
        return results
    
    def evaluate_from_inference_result(
        self,
        inference_result: Dict[str, Any],
        save_renders: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate 3D model from inference3d.py output
        
        Args:
            inference_result: Output dict from inference3d.py (or loaded from meta JSON)
            save_renders: Save rendered views
            output_dir: Output directory for renders
            
        Returns:
            Evaluation results dictionary
        """
        # Extract model path
        artifacts = inference_result.get("artifacts", {})
        model_path = artifacts.get("glb")
        
        if not model_path or not Path(model_path).exists():
            return {"error": "Model file not found in inference result"}
        
        # Extract input condition
        inputs = inference_result.get("inputs", {})
        image_path = inputs.get("image_path")
        
        condition = None
        condition_type = "image"
        
        if image_path and Path(image_path).exists():
            try:
                condition = Image.open(image_path).convert("RGB")
            except Exception as e:
                log(f"[WARN] Failed to load input image: {e}")
        
        # Run evaluation
        eval_result = self.evaluate(
            model_path=model_path,
            condition=condition,
            condition_type=condition_type,
            save_renders=save_renders,
            output_dir=output_dir,
        )
        
        # Add backend info
        eval_result["backend"] = inference_result.get("backend")
        eval_result["inference_params"] = inference_result.get("params")
        
        return eval_result


# -----------------------------
# Batch Evaluation
# -----------------------------
def batch_evaluate(
    models: List[str],
    conditions: List[Union[str, Image.Image, None]],
    output_csv: str,
    evaluator: Universal3DEvaluator,
    save_renders: bool = False,
    render_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Batch evaluate multiple models
    
    Args:
        models: List of model file paths
        conditions: List of conditions (text/image) for each model
        output_csv: Path to save CSV report
        evaluator: Universal3DEvaluator instance
        save_renders: Save rendered views
        render_dir: Directory for renders
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    for model_path, condition in tqdm(zip(models, conditions), total=len(models), desc="Evaluating"):
        try:
            result = evaluator.evaluate(
                model_path=model_path,
                condition=condition,
                save_renders=save_renders,
                output_dir=render_dir,
            )
            
            # Flatten for CSV
            flat_result = {
                "model_name": result["model_name"],
                "model_path": result["model_path"],
                "num_vertices": result["geometric_stats"]["num_vertices"],
                "num_faces": result["geometric_stats"]["num_faces"],
                "surface_area": result["geometric_stats"]["surface_area"],
                "is_watertight": result["mesh_validity"]["is_watertight"],
                "is_manifold": result["mesh_validity"]["is_manifold"],
                "validity_score": result["mesh_validity"]["validity_score"],
            }
            
            if result.get("clip_score"):
                flat_result.update({
                    "clip_score_mean": result["clip_score"]["clip_score_mean"],
                    "clip_score_std": result["clip_score"]["clip_score_std"],
                    "clip_score_min": result["clip_score"]["clip_score_min"],
                    "clip_score_max": result["clip_score"]["clip_score_max"],
                })
            
            if result.get("aesthetic_score") and result["aesthetic_score"]["aesthetic_score_mean"] is not None:
                flat_result.update({
                    "aesthetic_score_mean": result["aesthetic_score"]["aesthetic_score_mean"],
                    "aesthetic_score_std": result["aesthetic_score"]["aesthetic_score_std"],
                })
            
            if result.get("consistency") and result["consistency"]["consistency_score_mean"] is not None:
                flat_result.update({
                    "consistency_score_mean": result["consistency"]["consistency_score_mean"],
                    "consistency_score_std": result["consistency"]["consistency_score_std"],
                })
            
            results.append(flat_result)
            
        except Exception as e:
            log(f"[ERROR] Failed to evaluate {model_path}: {e}")
            results.append({
                "model_name": Path(model_path).name,
                "model_path": str(model_path),
                "error": str(e),
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    log(f"Saved batch results to: {output_csv}")
    
    return df


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified 3D Model Quality Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model with image condition
  python evaluate.py --model output.glb --condition input.png --type image
  
  # Evaluate from inference3d.py meta JSON
  python evaluate.py --meta-json outputs/meta_xxx.json --save-renders
  
  # Batch evaluate all GLB files in directory
  python evaluate.py --batch-dir outputs/ --pattern "*.glb" --output-csv results.csv
        """
    )
    
    # Single model evaluation
    parser.add_argument("--model", type=str, help="Path to 3D model file (GLB/OBJ/PLY)")
    parser.add_argument("--condition", type=str, help="Text prompt or path to input image")
    parser.add_argument("--type", type=str, choices=["text", "image", "auto"], default="auto",
                       help="Condition type")
    
    # Evaluation from inference result
    parser.add_argument("--meta-json", type=str, help="Path to inference3d.py meta JSON file")
    
    # Batch evaluation
    parser.add_argument("--batch-dir", type=str, help="Directory containing models to evaluate")
    parser.add_argument("--pattern", type=str, default="*.glb", help="File pattern for batch mode")
    
    # Output options
    parser.add_argument("--output-json", type=str, help="Path to save JSON results (single model)")
    parser.add_argument("--output-csv", type=str, help="Path to save CSV results (batch mode)")
    parser.add_argument("--save-renders", action="store_true", help="Save rendered views")
    parser.add_argument("--render-dir", type=str, default="eval_renders", help="Directory for renders")
    
    # Evaluator config
    parser.add_argument("--clip-model", type=str, default="ViT-B/32",
                       choices=["ViT-B/32", "ViT-L/14"], help="CLIP model variant")
    parser.add_argument("--resolution", type=int, default=512, help="Rendering resolution")
    parser.add_argument("--num-views", type=int, default=8, help="Number of azimuth views")
    parser.add_argument("--elevations", type=float, nargs="+", default=[0.0, 30.0],
                       help="Elevation angles (degrees)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--use-aesthetic", action="store_true", help="Enable aesthetic scoring")
    parser.add_argument("--use-consistency", action="store_true", help="Enable consistency evaluation")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize evaluator
    evaluator = Universal3DEvaluator(
        clip_model=args.clip_model,
        resolution=args.resolution,
        num_views=args.num_views,
        elevations=args.elevations,
        device=args.device,
        use_aesthetic=args.use_aesthetic,
        use_consistency=args.use_consistency,
    )
    
    # Mode 1: Evaluate from meta JSON
    if args.meta_json:
        log(f"Loading inference result from: {args.meta_json}")
        with open(args.meta_json, "r") as f:
            inference_result = json.load(f)
        
        result = evaluator.evaluate_from_inference_result(
            inference_result,
            save_renders=args.save_renders,
            output_dir=args.render_dir,
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {result['model_name']}")
        print(f"Backend: {result.get('backend', 'N/A')}")
        print(f"\n📊 Geometric Stats:")
        print(f"  Vertices: {result['geometric_stats']['num_vertices']:,}")
        print(f"  Faces: {result['geometric_stats']['num_faces']:,}")
        print(f"  Surface Area: {result['geometric_stats']['surface_area']:.4f}")
        print(f"\n✅ Mesh Validity:")
        print(f"  Watertight: {result['mesh_validity']['is_watertight']}")
        print(f"  Manifold: {result['mesh_validity']['is_manifold']}")
        print(f"  Validity Score: {result['mesh_validity']['validity_score']:.2f}")
        
        if result.get("clip_score"):
            print(f"\n🎯 CLIP Score:")
            print(f"  Mean: {result['clip_score']['clip_score_mean']:.4f}")
            print(f"  Std: {result['clip_score']['clip_score_std']:.4f}")
            print(f"  Range: [{result['clip_score']['clip_score_min']:.4f}, {result['clip_score']['clip_score_max']:.4f}]")
        
        print("="*60 + "\n")
        
        # Save JSON
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            log(f"Saved results to: {args.output_json}")
        else:
            # Auto-save next to meta JSON
            output_path = Path(args.meta_json).with_suffix(".eval.json")
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            log(f"Saved results to: {output_path}")
    
    # Mode 2: Evaluate single model
    elif args.model:
        if not Path(args.model).exists():
            print(f"Error: Model file not found: {args.model}")
            return
        
        # Load condition
        condition = None
        if args.condition:
            if args.type == "text" or (args.type == "auto" and not Path(args.condition).exists()):
                condition = args.condition
            else:
                condition = Image.open(args.condition).convert("RGB")
        
        result = evaluator.evaluate(
            model_path=args.model,
            condition=condition,
            condition_type=args.type,
            save_renders=args.save_renders,
            output_dir=args.render_dir,
        )
        
        # Print and save (similar to above)
        print(json.dumps(result, indent=2))
        
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            log(f"Saved results to: {args.output_json}")
    
    # Mode 3: Batch evaluation
    elif args.batch_dir:
        batch_dir = Path(args.batch_dir)
        if not batch_dir.exists():
            print(f"Error: Batch directory not found: {batch_dir}")
            return
        
        # Find all matching files
        model_files = sorted(batch_dir.rglob(args.pattern))
        log(f"Found {len(model_files)} models matching pattern: {args.pattern}")
        
        if len(model_files) == 0:
            print("No models found!")
            return
        
        # For batch mode, no conditions (pure quality assessment)
        conditions = [None] * len(model_files)
        
        output_csv = args.output_csv or "evaluation_results.csv"
        
        df = batch_evaluate(
            models=[str(f) for f in model_files],
            conditions=conditions,
            output_csv=output_csv,
            evaluator=evaluator,
            save_renders=args.save_renders,
            render_dir=args.render_dir,
        )
        
        print(f"\n✅ Batch evaluation complete. Results saved to: {output_csv}")
        print(f"\nSummary Statistics:")
        print(df.describe())
    
    else:
        print("Error: Must specify --model, --meta-json, or --batch-dir")
        print("Run with --help for usage examples")


if __name__ == "__main__":
    main()
