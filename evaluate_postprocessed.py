#!/usr/bin/env python3
"""
Evaluate Hausdorff distance for post-processed 3D models.

Compares post-processed GLB files in Post-Processed/post/ 
against ground truth mesh.ply files in datasets/Toys2h/renders/{sha256}/
"""

import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from hausdorff_eval import evaluate_hausdorff

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def build_name_to_sha256_mapping(metadata_file: str) -> Dict[str, str]:
    """
    Build object_name -> sha256 mapping from metadata.csv.
    """
    df = pd.read_csv(metadata_file)
    mapping = {}
    for _, row in df.iterrows():
        sha256 = str(row['sha256'])
        file_id = str(row['file_identifier'])
        # Extract object name: e.g., "hammer/hammer_075/hammer_075.blend" -> "hammer_075"
        object_name = file_id.split('/')[-1].replace('.blend', '')
        mapping[object_name] = sha256
    return mapping


def evaluate_postprocessed(
    post_dir: str = "Post-Processed/post",
    gt_base_dir: str = "datasets/Toys2h/renders",
    metadata_file: str = "datasets/Toys2h/metadata.csv",
    output_json: str = "evaluation/hausdorff_postprocessed.json",
    num_points: int = 100000,
) -> Dict:
    """
    Evaluate all post-processed GLB files against ground truth mesh.ply.
    """
    post_dir = Path(post_dir)
    gt_base_dir = Path(gt_base_dir)
    
    # Build name -> sha256 mapping
    logger.info("Building object name to sha256 mapping...")
    name_to_sha256 = build_name_to_sha256_mapping(metadata_file)
    logger.info(f"Loaded {len(name_to_sha256)} mappings")
    
    # Find all post-processed GLB files
    post_files = sorted(post_dir.glob("post_*.glb"))
    logger.info(f"Found {len(post_files)} post-processed GLB files")
    
    results = {}
    successful = 0
    failed = 0
    
    for post_file in post_files:
        # Extract object name: post_hammer_075.glb -> hammer_075
        object_name = post_file.stem.replace("post_", "")
        
        # Get sha256
        sha256 = name_to_sha256.get(object_name)
        if not sha256:
            logger.warning(f"SKIP: {object_name} not found in metadata")
            results[object_name] = {"success": False, "error": "Not found in metadata"}
            failed += 1
            continue
        
        # Ground truth path
        gt_ply = gt_base_dir / sha256 / "mesh.ply"
        if not gt_ply.exists():
            logger.warning(f"SKIP: GT mesh not found: {gt_ply}")
            results[object_name] = {"success": False, "error": f"GT not found: {gt_ply}"}
            failed += 1
            continue
        
        # Evaluate
        logger.info(f"Evaluating {object_name}...")
        try:
            result = evaluate_hausdorff(
                ground_truth_path=str(gt_ply),
                prediction_path=str(post_file),
                num_points=num_points,
                output_json=None,
                visualize=False
            )
            results[object_name] = result
            if result.get("success"):
                successful += 1
                logger.info(f"  ✅ {object_name}: HD = {result['symmetric_hausdorff']:.6f}")
            else:
                failed += 1
                logger.error(f"  ❌ {object_name}: {result.get('error')}")
        except Exception as e:
            logger.error(f"  ❌ {object_name}: {e}")
            results[object_name] = {"success": False, "error": str(e)}
            failed += 1
    
    # Summary statistics
    hd_values = [r["symmetric_hausdorff"] for r in results.values() if r.get("success")]
    
    summary = {
        "total": len(post_files),
        "successful": successful,
        "failed": failed,
        "statistics": {}
    }
    
    if hd_values:
        import numpy as np
        summary["statistics"] = {
            "mean": float(np.mean(hd_values)),
            "std": float(np.std(hd_values)),
            "min": float(np.min(hd_values)),
            "max": float(np.max(hd_values)),
            "median": float(np.median(hd_values)),
        }
    
    output = {
        "summary": summary,
        "results": results
    }
    
    # Save results
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("POST-PROCESSED EVALUATION COMPLETE")
    print("="*80)
    print(f"Total:      {summary['total']}")
    print(f"Successful: {summary['successful']} ✅")
    print(f"Failed:     {summary['failed']} ❌")
    
    if summary["statistics"]:
        print("\nHausdorff Distance Statistics:")
        print(f"  Mean:   {summary['statistics']['mean']:.6f}")
        print(f"  Std:    {summary['statistics']['std']:.6f}")
        print(f"  Min:    {summary['statistics']['min']:.6f}")
        print(f"  Max:    {summary['statistics']['max']:.6f}")
        print(f"  Median: {summary['statistics']['median']:.6f}")
    
    print(f"\nResults saved to: {output_path}")
    print("="*80 + "\n")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate post-processed 3D models")
    parser.add_argument("--post-dir", type=str, default="Post-Processed/post_hunyuan3d2",
                        help="Directory containing post_*.glb files")
    parser.add_argument("--gt-dir", type=str, default="datasets/Toys2h/renders",
                        help="Base directory for ground truth meshes")
    parser.add_argument("--metadata", type=str, default="datasets/Toys2h/metadata.csv",
                        help="Path to metadata.csv")
    parser.add_argument("--output", "-o", type=str, default="evaluation/hausdorff_postprocessed.json",
                        help="Output JSON file path")
    parser.add_argument("--num-points", type=int, default=100000,
                        help="Number of sample points per mesh")
    
    args = parser.parse_args()
    
    evaluate_postprocessed(
        post_dir=args.post_dir,
        gt_base_dir=args.gt_dir,
        metadata_file=args.metadata,
        output_json=args.output,
        num_points=args.num_points,
    )
