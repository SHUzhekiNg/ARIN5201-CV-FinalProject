#!/usr/bin/env python3
"""
Hausdorff distance evaluation between ground truth (.ply) and prediction (.glb).

Compute symmetric Hausdorff distance between two 3D meshes:
  - Ground truth: .ply file
  - Prediction: .glb file

Dependencies:
  - trimesh: mesh loading & point cloud sampling
  - scipy: Hausdorff distance computation
  - numpy: numerical operations
  - pandas: CSV handling
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import trimesh

logger = logging.getLogger(__name__)


def load_mesh_from_file(file_path: str) -> trimesh.Trimesh:
    """
    Load a 3D mesh from file (.ply, .glb, .gltf, etc.).
    
    Args:
        file_path: path to mesh file
    
    Returns:
        trimesh.Trimesh object
    
    Raises:
        RuntimeError if file cannot be loaded
    """
    file_path = str(file_path)
    logger.info(f'Loading mesh from: {file_path}')
    mesh = trimesh.load(file_path, force='mesh')
    
    # If we got a Scene, merge all geometries
    if isinstance(mesh, trimesh.Scene):
        logger.warning('Input is a Scene; merging all geometries.')
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    
    logger.info(f'Loaded mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces')
    return mesh


def sample_points_from_mesh(mesh: trimesh.Trimesh, num_points: int = 100000) -> np.ndarray:
    """
    Sample points uniformly from mesh surface.
    
    Args:
        mesh: trimesh.Trimesh object
        num_points: number of points to sample
    
    Returns:
        (num_points, 3) array of 3D points
    """
    points, _ = trimesh.sample.sample_surface(mesh, count=num_points)
    logger.info(f'Sampled {len(points)} points from mesh')
    return points


def hausdorff_distance(points1: np.ndarray, points2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute symmetric Hausdorff distance and directed distances.
    
    Hausdorff distance: max(d(P1, P2), d(P2, P1))
    where d(P1, P2) = max over p in P1 of min over q in P2 of ||p - q||
    
    Args:
        points1: (N, 3) array of 3D points
        points2: (M, 3) array of 3D points
    
    Returns:
        tuple: (symmetric_hausdorff_distance, directed_d1_to_d2, directed_d2_to_d1)
    """
    from scipy.spatial.distance import cdist
    
    logger.info(f'Computing Hausdorff distance between {len(points1)} and {len(points2)} points...')
    
    # Compute pairwise distances
    distances = cdist(points1, points2, metric='euclidean')  # (N, M)
    
    # d(P1, P2): for each point in P1, find min distance to P2
    d1_to_2 = np.min(distances, axis=1)  # (N,)
    directed_d1_to_d2 = np.max(d1_to_2)
    
    # d(P2, P1): for each point in P2, find min distance to P1
    d2_to_1 = np.min(distances, axis=0)  # (M,)
    directed_d2_to_d1 = np.max(d2_to_1)
    
    # Symmetric Hausdorff distance
    symmetric_hausdorff = max(directed_d1_to_d2, directed_d2_to_d1)
    
    logger.info(f'Directed Hausdorff (GT -> Pred): {directed_d1_to_d2:.6f}')
    logger.info(f'Directed Hausdorff (Pred -> GT): {directed_d2_to_d1:.6f}')
    logger.info(f'Symmetric Hausdorff distance: {symmetric_hausdorff:.6f}')
    
    return symmetric_hausdorff, directed_d1_to_d2, directed_d2_to_d1


def evaluate_hausdorff(ground_truth_path: str, prediction_path: str,
                       num_points: int = 100000,
                       output_json: str = None,
                       visualize: bool = False) -> dict:
    """
    Compute Hausdorff distance between ground truth and prediction meshes.
    
    Args:
        ground_truth_path: path to .blend (ground truth)
        prediction_path: path to .glb (prediction)
        num_points: number of surface points to sample per mesh
        blender_bin: (optional) path to blender executable
        output_json: (optional) path to save results as JSON
        visualize: whether to save a visualization (requires open3d)
    
    Returns:
        dict with evaluation results
    """
    results = {
        'ground_truth': str(ground_truth_path),
        'prediction': str(prediction_path),
        'num_sample_points': num_points,
    }
    
    try:
        # Load ground truth mesh (.ply)
        logger.info('Loading ground truth mesh...')
        gt_mesh = load_mesh_from_file(ground_truth_path)
        
        # Load prediction mesh (.glb)
        logger.info('Loading prediction mesh...')
        pred_mesh = load_mesh_from_file(prediction_path)
        
        # Sample points from surfaces
        logger.info('Sampling surface points...')
        gt_points = sample_points_from_mesh(gt_mesh, num_points=num_points)
        pred_points = sample_points_from_mesh(pred_mesh, num_points=num_points)
        
        # Compute Hausdorff distance
        logger.info('Computing Hausdorff distance...')
        sym_hausdorff, d_gt_to_pred, d_pred_to_gt = hausdorff_distance(gt_points, pred_points)
        
        results.update({
            'symmetric_hausdorff': float(sym_hausdorff),
            'directed_hausdorff_gt_to_pred': float(d_gt_to_pred),
            'directed_hausdorff_pred_to_gt': float(d_pred_to_gt),
            'gt_vertices': int(gt_mesh.vertices.shape[0]),
            'gt_faces': int(gt_mesh.faces.shape[0]),
            'pred_vertices': int(pred_mesh.vertices.shape[0]),
            'pred_faces': int(pred_mesh.faces.shape[0]),
            'success': True,
        })
        
        # Optional: save visualization
        if visualize:
            try:
                import open3d as o3d
                logger.info('Creating visualization...')
                vis_path = Path(output_json).parent / 'hausdorff_vis.ply' if output_json else 'hausdorff_vis.ply'
                
                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
                pcd_gt.paint_uniform_color([0, 1, 0])  # Green for GT
                
                pcd_pred = o3d.geometry.PointCloud()
                pcd_pred.points = o3d.utility.Vector3dVector(pred_points)
                pcd_pred.paint_uniform_color([1, 0, 0])  # Red for prediction
                
                o3d.io.write_point_cloud(str(vis_path), pcd_gt + pcd_pred)
                logger.info(f'Visualization saved to: {vis_path}')
                results['visualization_path'] = str(vis_path)
            except ImportError:
                logger.warning('open3d not available; skipping visualization')
    
    except Exception as e:
        logger.error(f'Evaluation failed: {e}', exc_info=True)
        results.update({
            'success': False,
            'error': str(e),
        })
    
    # Save results to JSON if requested
    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Results saved to: {output_json}')
    
    return results


def batch_evaluate_hausdorff(paths_file: str, metadata_file: str, evaluation_dir: str,
                             num_points: int = 100000, output_csv: str = 'hd_eval.csv') -> None:
    """
    Batch evaluate Hausdorff distance for all predictions.
    
    Args:
        paths_file: path to txt file with ground truth folder paths (one per line)
        metadata_file: path to metadata.csv to get object names
        evaluation_dir: base directory containing Hunyuan3D2 and Trellis subdirectories
        num_points: number of sample points per mesh
        output_csv: output CSV file path
    """
    # Read metadata to map sha256 to object name
    metadata_df = pd.read_csv(metadata_file)
    sha256_to_name = {}
    for _, row in metadata_df.iterrows():
        file_id = str(row['file_identifier'])
        # Extract object name: last part after '/' and before '.'
        object_name = file_id.split('/')[-1].replace('.blend', '')
        sha256 = str(row['sha256'])
        sha256_to_name[sha256] = object_name
    
    logger.info(f'Loaded {len(sha256_to_name)} object names from metadata')
    
    # Read paths file
    with open(paths_file, 'r') as f:
        gt_paths = [line.strip() for line in f if line.strip()]
    
    logger.info(f'Read {len(gt_paths)} ground truth paths')
    
    # Prepare evaluation directory
    eval_base = Path(evaluation_dir)
    hunyuan_dir = eval_base / 'Hunyuan3D2'
    trellis_dir = eval_base / 'Trellis'
    
    if not hunyuan_dir.exists() or not trellis_dir.exists():
        raise RuntimeError(f'Evaluation directories not found: {hunyuan_dir}, {trellis_dir}')
    
    # Collect results
    results = []
    
    for gt_path in gt_paths:
        gt_path = Path(gt_path)
        
        # Extract sha256 from path (folder name)
        sha256 = gt_path.name
        object_name = sha256_to_name.get(sha256, sha256)
        
        # Ground truth mesh
        gt_ply = gt_path / 'mesh.ply'
        if not gt_ply.exists():
            logger.warning(f'Ground truth PLY not found: {gt_ply}')
            results.append({
                'object_name': object_name,
                'hunyuan3d2_hd': None,
                'trellis_hd': None,
                'error': 'GT PLY not found'
            })
            continue
        
        # Prediction GLBs - use object_name (e.g., apple_028.glb) not sha256
        hunyuan_glb = hunyuan_dir / f'{object_name}.glb'
        trellis_glb = trellis_dir / f'{object_name}.glb'
        
        hunyuan_hd = None
        trellis_hd = None
        error_msg = ''
        
        # Compute Hunyuan distance
        if hunyuan_glb.exists():
            try:
                logger.info(f'Computing Hunyuan distance for {object_name}')
                result_hunyuan = evaluate_hausdorff(
                    ground_truth_path=str(gt_ply),
                    prediction_path=str(hunyuan_glb),
                    num_points=num_points,
                    output_json=None,
                    visualize=False
                )
                if result_hunyuan['success']:
                    hunyuan_hd = result_hunyuan['symmetric_hausdorff']
                else:
                    error_msg += f'Hunyuan: {result_hunyuan.get("error", "unknown")}; '
            except Exception as e:
                logger.error(f'Hunyuan evaluation failed for {object_name}: {e}')
                error_msg += f'Hunyuan: {str(e)}; '
        else:
            logger.warning(f'Hunyuan GLB not found: {hunyuan_glb}')
            error_msg += 'Hunyuan GLB not found; '
        
        # Compute Trellis distance
        if trellis_glb.exists():
            try:
                logger.info(f'Computing Trellis distance for {object_name}')
                result_trellis = evaluate_hausdorff(
                    ground_truth_path=str(gt_ply),
                    prediction_path=str(trellis_glb),
                    num_points=num_points,
                    output_json=None,
                    visualize=False
                )
                if result_trellis['success']:
                    trellis_hd = result_trellis['symmetric_hausdorff']
                else:
                    error_msg += f'Trellis: {result_trellis.get("error", "unknown")}'
            except Exception as e:
                logger.error(f'Trellis evaluation failed for {object_name}: {e}')
                error_msg += f'Trellis: {str(e)}'
        else:
            logger.warning(f'Trellis GLB not found: {trellis_glb}')
            error_msg += 'Trellis GLB not found'
        
        results.append({
            'object_name': object_name,
            'hunyuan3d2_hd': hunyuan_hd,
            'trellis_hd': trellis_hd,
            'error': error_msg.strip() if error_msg else ''
        })
    
    # Write results to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    
    # Also save as JSON with summary statistics
    json_output_path = output_path.with_suffix('.json')
    
    # Compute summary statistics
    summary = {
        'total_objects': len(results),
        'successful_hunyuan': int(df_results["hunyuan3d2_hd"].notna().sum()),
        'successful_trellis': int(df_results["trellis_hd"].notna().sum()),
    }
    
    if df_results["hunyuan3d2_hd"].notna().sum() > 0:
        summary['hunyuan3d2_stats'] = {
            'mean': float(df_results["hunyuan3d2_hd"].mean()),
            'std': float(df_results["hunyuan3d2_hd"].std()),
            'min': float(df_results["hunyuan3d2_hd"].min()),
            'max': float(df_results["hunyuan3d2_hd"].max()),
            'median': float(df_results["hunyuan3d2_hd"].median()),
        }
    
    if df_results["trellis_hd"].notna().sum() > 0:
        summary['trellis_stats'] = {
            'mean': float(df_results["trellis_hd"].mean()),
            'std': float(df_results["trellis_hd"].std()),
            'min': float(df_results["trellis_hd"].min()),
            'max': float(df_results["trellis_hd"].max()),
            'median': float(df_results["trellis_hd"].median()),
        }
    
    json_output = {
        'summary': summary,
        'results': results
    }
    
    with open(json_output_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    logger.info(f'Results saved to: {output_path} and {json_output_path}')
    print(f'\n{"="*80}')
    print(f'Batch Evaluation Complete')
    print(f'{"="*80}')
    print(f'Total objects: {len(results)}')
    print(f'Successful Hunyuan: {df_results["hunyuan3d2_hd"].notna().sum()}')
    print(f'Successful Trellis: {df_results["trellis_hd"].notna().sum()}')
    print(f'CSV saved to: {output_path}')
    print(f'JSON saved to: {json_output_path}')
    print(f'{"="*80}\n')
    
    # Print summary statistics
    if df_results["hunyuan3d2_hd"].notna().sum() > 0:
        print(f'Hunyuan3D2 Hausdorff Distance Statistics:')
        print(f'  Mean: {df_results["hunyuan3d2_hd"].mean():.6f}')
        print(f'  Std:  {df_results["hunyuan3d2_hd"].std():.6f}')
        print(f'  Min:  {df_results["hunyuan3d2_hd"].min():.6f}')
        print(f'  Max:  {df_results["hunyuan3d2_hd"].max():.6f}')
    
    if df_results["trellis_hd"].notna().sum() > 0:
        print(f'Trellis Hausdorff Distance Statistics:')
        print(f'  Mean: {df_results["trellis_hd"].mean():.6f}')
        print(f'  Std:  {df_results["trellis_hd"].std():.6f}')
        print(f'  Min:  {df_results["trellis_hd"].min():.6f}')
        print(f'  Max:  {df_results["trellis_hd"].max():.6f}')


def main():
    parser = argparse.ArgumentParser(
        description='Compute Hausdorff distance between ground truth (.ply) and prediction (.glb) meshes.'
    )
    parser.add_argument('ground_truth', nargs='?', default=None, help='Path to ground truth .ply file')
    parser.add_argument('prediction', nargs='?', default=None, help='Path to prediction .glb file')
    parser.add_argument('--num-points', type=int, default=100000,
                        help='Number of surface points to sample per mesh (default: 100000)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save results as JSON (single mode) or CSV (batch mode)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate visualization (requires open3d)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    
    # Batch mode arguments
    parser.add_argument('--batch', action='store_true',
                        help='Run batch evaluation mode')
    parser.add_argument('--paths-file', type=str, default=None,
                        help='Path to txt file with ground truth folder paths (for batch mode)')
    parser.add_argument('--metadata-file', type=str, default=None,
                        help='Path to metadata.csv file (for batch mode)')
    parser.add_argument('--eval-dir', type=str, default=None,
                        help='Base directory with Hunyuan3D2 and Trellis subdirectories (for batch mode)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='[%(levelname)s] %(name)s: %(message)s'
    )
    
    # Batch mode
    if args.batch:
        if not args.paths_file or not args.metadata_file or not args.eval_dir:
            parser.error('Batch mode requires --paths-file, --metadata-file, and --eval-dir')
        output_csv = args.output or 'hd_eval.csv'
        batch_evaluate_hausdorff(
            paths_file=args.paths_file,
            metadata_file=args.metadata_file,
            evaluation_dir=args.eval_dir,
            num_points=args.num_points,
            output_csv=output_csv
        )
        return 0
    
    # Single mode
    if not args.ground_truth or not args.prediction:
        parser.error('Single mode requires ground_truth and prediction arguments')
    
    # Auto-generate JSON output path if not provided
    if not args.output:
        pred_name = Path(args.prediction).stem
        args.output = f'hausdorff_{pred_name}.json'
    
    # Run evaluation
    results = evaluate_hausdorff(
        ground_truth_path=args.ground_truth,
        prediction_path=args.prediction,
        num_points=args.num_points,
        output_json=args.output,
        visualize=args.visualize,
    )
    
    # Print summary
    print('\n' + '='*60)
    print('HAUSDORFF DISTANCE EVALUATION RESULTS')
    print('='*60)
    if results['success']:
        print(f'Ground Truth:           {results["ground_truth"]}')
        print(f'Prediction:             {results["prediction"]}')
        print(f'Sample Points:          {results["num_sample_points"]}')
        print(f'GT Mesh:                {results["gt_vertices"]} verts, {results["gt_faces"]} faces')
        print(f'Pred Mesh:              {results["pred_vertices"]} verts, {results["pred_faces"]} faces')
        print(f'\nHausdorff Distances:')
        print(f'  Symmetric:            {results["symmetric_hausdorff"]:.6f}')
        print(f'  GT → Pred:            {results["directed_hausdorff_gt_to_pred"]:.6f}')
        print(f'  Pred → GT:            {results["directed_hausdorff_pred_to_gt"]:.6f}')
        if 'visualization_path' in results:
            print(f'\nVisualization:          {results["visualization_path"]}')
        print(f'\n✅ Results saved to:    {args.output}')
    else:
        print(f'FAILED: {results["error"]}')
        sys.exit(1)
    print('='*60 + '\n')
    
    return 0 if results['success'] else 1


def test():
    results = {}

    # Dragon
    results['dragon_007'] = evaluate_hausdorff(
        ground_truth_path='test_renders/dragon_007.glb',
        prediction_path='evaluation/Hunyuan3D2/dragon_007.glb',
        num_points=100000
    )

    # Hammer
    results['hammer_075'] = evaluate_hausdorff(
        ground_truth_path='test_renders/hammer_075.glb',
        prediction_path='evaluation/Hunyuan3D2/hammer_075.glb',
        num_points=100000
    )

    with open('evaluation/hausdorff_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    sys.exit(main())
