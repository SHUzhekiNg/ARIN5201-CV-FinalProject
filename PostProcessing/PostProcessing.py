"""
GLB Post-Processing Module for Blender
======================================

功能：
- Mesh Cleanup: 移除重复顶点、删除孤立顶点和边、删除退化面
- Geometry Repair: 修复法线方向、修复非流形边、填充孔洞
- Topology Optimization: Decimate简化、Surface smooth平滑

使用方式：
1. 批处理: blender --background --python PostProcessing.py -- --batch --input <folder> [options]
2. 单文件: blender --background --python PostProcessing.py -- --input <file.glb> [options]

"""

import bpy
import bmesh
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

DEFAULT_MERGE_DISTANCE = 0.0001 # 合并顶点的距离阈值
DEFAULT_DECIMATE_RATIO = 0.8 # Decimate 简化比率 (0-1)
DEFAULT_SMOOTH_ITERATIONS = 5 # 平滑迭代次数


class MeshStats:
    """网格统计信息"""
    
    def __init__(self):
        self.vertices = 0
        self.edges = 0
        self.faces = 0
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "vertices": self.vertices,
            "edges": self.edges,
            "faces": self.faces
        }
    
    @staticmethod
    def from_mesh_objects(objects: List[bpy.types.Object]) -> 'MeshStats':
        """从网格对象列表获取统计信息"""
        stats = MeshStats()
        for obj in objects:
            if obj.type == 'MESH':
                mesh = obj.data
                stats.vertices += len(mesh.vertices)
                stats.edges += len(mesh.edges)
                stats.faces += len(mesh.polygons)
        return stats


class ProcessingReport:
    """单个文件的处理报告"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.before = MeshStats()
        self.after = MeshStats()
        self.operations = {
            "merged_vertices": 0,
            "deleted_loose_elements": 0,
            "deleted_degenerate_faces": 0,
            "recalculated_normals": 0,
            "repaired_non_manifold": 0,
            "filled_holes": 0,
            "decimate_applied": False,
            "smooth_applied": False
        }
        self.success = False
        self.error_message = ""
        self.processing_time = 0.0
    
    def calculate_drop_ratio(self) -> Dict[str, float]:
        """计算各项的减少比例"""
        def safe_ratio(before: int, after: int) -> float:
            if before == 0:
                return 0.0
            return round((before - after) / before * 100, 2)
        
        return {
            "vertices_drop_percent": safe_ratio(self.before.vertices, self.after.vertices),
            "edges_drop_percent": safe_ratio(self.before.edges, self.after.edges),
            "faces_drop_percent": safe_ratio(self.before.faces, self.after.faces)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        drop_ratio = self.calculate_drop_ratio()
        return {
            "filename": self.filename,
            "success": self.success,
            "error_message": self.error_message,
            "processing_time_seconds": round(self.processing_time, 3),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "drop_ratio": drop_ratio,
            "operations": self.operations
        }


class BatchReport:
    """批处理报告"""
    
    def __init__(self):
        self.reports: List[ProcessingReport] = []
        self.input_folder = ""
        self.output_folder = ""
        self.parameters = {}
        self.timestamp = ""
    
    def add_report(self, report: ProcessingReport):
        self.reports.append(report)
    
    def calculate_average(self) -> Dict[str, Any]:
        """计算平均值"""
        successful_reports = [r for r in self.reports if r.success]
        if not successful_reports:
            return {}
        
        count = len(successful_reports)
        
        avg = {
            "total_files": len(self.reports),
            "successful_files": count,
            "failed_files": len(self.reports) - count,
            "average_before": {
                "vertices": round(sum(r.before.vertices for r in successful_reports) / count, 2),
                "edges": round(sum(r.before.edges for r in successful_reports) / count, 2),
                "faces": round(sum(r.before.faces for r in successful_reports) / count, 2)
            },
            "average_after": {
                "vertices": round(sum(r.after.vertices for r in successful_reports) / count, 2),
                "edges": round(sum(r.after.edges for r in successful_reports) / count, 2),
                "faces": round(sum(r.after.faces for r in successful_reports) / count, 2)
            },
            "average_drop_ratio": {
                "vertices_drop_percent": round(sum(r.calculate_drop_ratio()["vertices_drop_percent"] for r in successful_reports) / count, 2),
                "edges_drop_percent": round(sum(r.calculate_drop_ratio()["edges_drop_percent"] for r in successful_reports) / count, 2),
                "faces_drop_percent": round(sum(r.calculate_drop_ratio()["faces_drop_percent"] for r in successful_reports) / count, 2)
            },
            "average_processing_time_seconds": round(sum(r.processing_time for r in successful_reports) / count, 3),
            "total_operations": {
                "merged_vertices": sum(r.operations["merged_vertices"] for r in successful_reports),
                "deleted_loose_elements": sum(r.operations["deleted_loose_elements"] for r in successful_reports),
                "deleted_degenerate_faces": sum(r.operations["deleted_degenerate_faces"] for r in successful_reports),
                "recalculated_normals": sum(r.operations["recalculated_normals"] for r in successful_reports),
                "repaired_non_manifold": sum(r.operations["repaired_non_manifold"] for r in successful_reports),
                "filled_holes": sum(r.operations["filled_holes"] for r in successful_reports)
            }
        }
        return avg
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_processing_report": True,
            "timestamp": self.timestamp,
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "parameters": self.parameters,
            "summary": self.calculate_average(),
            "files": [r.to_dict() for r in self.reports]
        }
    
    def save(self, filepath: str):
        """保存报告到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[REPORT] Saved batch report to: {filepath}")


class MeshProcessor:    
    def __init__(
        self,
        merge_distance: float = DEFAULT_MERGE_DISTANCE,
        decimate_ratio: float = DEFAULT_DECIMATE_RATIO,
        smooth_iterations: int = DEFAULT_SMOOTH_ITERATIONS,
        enable_smooth: bool = True,
        enable_decimate: bool = True,
        enable_merge: bool = True
    ):
        self.merge_distance = merge_distance
        self.decimate_ratio = decimate_ratio
        self.smooth_iterations = smooth_iterations
        self.enable_smooth = enable_smooth
        self.enable_decimate = enable_decimate
        self.enable_merge = enable_merge
        self.current_report: Optional[ProcessingReport] = None
    
    def get_parameters_dict(self) -> Dict[str, Any]:
        """获取当前参数配置"""
        return {
            "merge_distance": self.merge_distance,
            "decimate_ratio": self.decimate_ratio,
            "smooth_iterations": self.smooth_iterations,
            "enable_smooth": self.enable_smooth,
            "enable_decimate": self.enable_decimate,
            "enable_merge": self.enable_merge
        }
    
    def clear_scene(self):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)
        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)
        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)
    
    def import_glb(self, filepath: str) -> bool:
        try:
            bpy.ops.import_scene.gltf(filepath=filepath)
            print(f"[INFO] Successfully imported: {filepath}")
            return True
        except Exception as e:
            print(f"[ERROR] Import failed {filepath}: {e}")
            return False
    
    def export_glb(self, filepath: str) -> bool:
        try:
            bpy.ops.export_scene.gltf(
                filepath=filepath,
                export_format='GLB',
                use_selection=False,
                export_apply=True
            )
            print(f"[INFO] Successfully exported: {filepath}")
            return True
        except Exception as e:
            print(f"[ERROR] Export failed {filepath}: {e}")
            return False
    
    def get_mesh_objects(self) -> List[bpy.types.Object]:
        return [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    # ==================== Mesh Cleanup ====================
    
    def remove_duplicate_vertices(self, obj: bpy.types.Object) -> int:
        """合并顶点，返回合并的数量"""
        if not self.enable_merge:
            print(f"  [Cleanup] Merge Vertices: Skipped (disabled)")
            return 0
        
        before_count = len(obj.data.vertices)
        
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=self.merge_distance)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        after_count = len(obj.data.vertices)
        merged = before_count - after_count
        
        print(f"  [Cleanup] Merge Vertices: {merged} merged (threshold={self.merge_distance})")
        return merged
    
    def delete_isolated_vertices_edges(self, obj: bpy.types.Object) -> int:
        """删除孤立的顶点和边，返回删除的数量"""
        before_count = len(obj.data.vertices)
        
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_loose()
        bpy.ops.mesh.delete(type='VERT')
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        after_count = len(obj.data.vertices)
        deleted = before_count - after_count
        
        print(f"  [Cleanup] Delete Isolated: {deleted} elements deleted")
        return deleted
    
    def delete_degenerate_faces(self, obj: bpy.types.Object) -> int:
        """删除退化面，返回删除的数量"""
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        bm = bmesh.from_edit_mesh(obj.data)
        
        degenerate_faces = [f for f in bm.faces if f.calc_area() < 1e-8]
        deleted_count = len(degenerate_faces)
        bmesh.ops.delete(bm, geom=degenerate_faces, context='FACES')
        
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        print(f"  [Cleanup] Delete Degenerate Faces: {deleted_count} faces deleted")
        return deleted_count
    
    # ==================== Geometry Repair ====================
    
    def fix_normal_direction(self, obj: bpy.types.Object) -> int:
        """修复法线方向，返回处理的面数"""
        face_count = len(obj.data.polygons)
        
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        print(f"  [Repair] Fix Normal Direction: {face_count} faces processed")
        return face_count
    
    def repair_non_manifold_edges(self, obj: bpy.types.Object) -> int:
        """修复非流形边，返回修复的数量"""
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_non_manifold()
        
        # 获取选中的非流形元素数量
        bm = bmesh.from_edit_mesh(obj.data)
        non_manifold_count = sum(1 for v in bm.verts if v.select)
        
        try:
            bpy.ops.mesh.fill()
        except:
            pass
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        print(f"  [Repair] Repair Non-Manifold: {non_manifold_count} elements found")
        return non_manifold_count
    
    def fill_holes(self, obj: bpy.types.Object) -> int:
        """填充孔洞，返回填充的数量"""
        before_faces = len(obj.data.polygons)
        
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        
        bpy.ops.mesh.select_all(action='DESELECT')
        
        bpy.ops.mesh.select_non_manifold(
            extend=False, 
            use_wire=False, 
            use_boundary=True, 
            use_multi_face=False,
            use_non_contiguous=False, 
            use_verts=False
        )
        
        try:
            bpy.ops.mesh.fill_holes(sides=0)
        except:
            pass
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        after_faces = len(obj.data.polygons)
        filled = after_faces - before_faces
        
        print(f"  [Repair] Fill Holes: {filled} faces added")
        return max(0, filled)
    
    # ==================== Topology Optimization ====================
    
    def decimate(self, obj: bpy.types.Object) -> bool:
        """Decimate简化网格"""
        if not self.enable_decimate:
            print(f"  [Optimize] Decimate Skipped (disabled)")
            return False
        
        modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
        modifier.ratio = self.decimate_ratio
        modifier.use_collapse_triangulate = True
        
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier="Decimate")
        print(f"  [Optimize] Decimate applied (ratio={self.decimate_ratio})")
        return True
    
    def surface_smooth(self, obj: bpy.types.Object) -> bool:
        """Surface smooth 表面平滑"""
        if not self.enable_smooth:
            print(f"  [Optimize] Surface smooth Skipped (disabled)")
            return False
        
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        
        bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=self.smooth_iterations)
        
        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"  [Optimize] Surface smooth applied (iterations={self.smooth_iterations})")
        return True
    
    # ==================== 主处理流程 ====================
    
    def process_object(self, obj: bpy.types.Object):
        """处理单个网格对象"""
        print(f"\n[Processing] Processing Object: {obj.name}")
        
        obj.hide_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # Step 1: Mesh Cleanup
        print("  --- Mesh Cleanup Processing ---")
        merged = self.remove_duplicate_vertices(obj)
        deleted_loose = self.delete_isolated_vertices_edges(obj)
        deleted_degenerate = self.delete_degenerate_faces(obj)
        
        if self.current_report:
            self.current_report.operations["merged_vertices"] += merged
            self.current_report.operations["deleted_loose_elements"] += deleted_loose
            self.current_report.operations["deleted_degenerate_faces"] += deleted_degenerate
        
        # Step 2: Geometry Repair
        print("  --- Geometry Repair Processing ---")
        normals_count = self.fix_normal_direction(obj)
        non_manifold = self.repair_non_manifold_edges(obj)
        holes_filled = self.fill_holes(obj)
        
        if self.current_report:
            self.current_report.operations["recalculated_normals"] += normals_count
            self.current_report.operations["repaired_non_manifold"] += non_manifold
            self.current_report.operations["filled_holes"] += holes_filled
        
        # Step 3: Topology Optimization
        print("  --- Topology Optimization Processing ---")
        decimate_applied = self.decimate(obj)
        smooth_applied = self.surface_smooth(obj)
        
        if self.current_report:
            self.current_report.operations["decimate_applied"] = decimate_applied
            self.current_report.operations["smooth_applied"] = smooth_applied
    
    def process_file(self, input_path: str, output_path: Optional[str] = None) -> ProcessingReport:
        """
        处理单个GLB文件
        
        Args:
            input_path: 输入GLB文件路径
            output_path: 输出GLB文件路径，如果为None则自动生成
        
        Returns:
            ProcessingReport: 处理报告
        """
        import time
        start_time = time.time()
        
        input_path = os.path.abspath(input_path)
        filename = os.path.basename(input_path)
        
        # 创建报告
        report = ProcessingReport(filename)
        self.current_report = report
        
        if output_path is None:
            input_dir = os.path.dirname(input_path)
            input_name = os.path.basename(input_path)
            output_name = f"post_{input_name}"
            output_path = os.path.join(input_dir, output_name)
        
        print(f"\n{'='*60}")
        print(f"[START] Processing File: {input_path}")
        print(f"[OUTPUT] Output Path: {output_path}")
        print(f"{'='*60}")
        
        # 清空场景
        self.clear_scene()
        
        # 导入GLB
        if not self.import_glb(input_path):
            report.success = False
            report.error_message = "Failed to import GLB file"
            report.processing_time = time.time() - start_time
            return report
        
        # 获取所有网格对象
        mesh_objects = self.get_mesh_objects()
        if not mesh_objects:
            print("[WARNING] No mesh objects found")
            report.success = False
            report.error_message = "No mesh objects found in file"
            report.processing_time = time.time() - start_time
            return report
        
        # 记录处理前的统计
        report.before = MeshStats.from_mesh_objects(mesh_objects)
        print(f"\n[STATS] Before Processing:")
        print(f"  Vertices: {report.before.vertices}")
        print(f"  Edges: {report.before.edges}")
        print(f"  Faces: {report.before.faces}")
        
        # 处理所有对象
        for obj in mesh_objects:
            self.process_object(obj)
        
        # 记录处理后的统计
        mesh_objects = self.get_mesh_objects()  # 重新获取（可能有变化）
        report.after = MeshStats.from_mesh_objects(mesh_objects)
        print(f"\n[STATS] After Processing:")
        print(f"  Vertices: {report.after.vertices}")
        print(f"  Edges: {report.after.edges}")
        print(f"  Faces: {report.after.faces}")
        
        # 计算drop ratio
        drop_ratio = report.calculate_drop_ratio()
        print(f"\n[STATS] Drop Ratio:")
        print(f"  Vertices: {drop_ratio['vertices_drop_percent']}%")
        print(f"  Edges: {drop_ratio['edges_drop_percent']}%")
        print(f"  Faces: {drop_ratio['faces_drop_percent']}%")
        
        # 导出GLB
        if not self.export_glb(output_path):
            report.success = False
            report.error_message = "Failed to export GLB file"
            report.processing_time = time.time() - start_time
            return report
        
        report.success = True
        report.processing_time = time.time() - start_time
        
        print(f"\n[DONE] Processing Completed: {output_path}")
        print(f"[TIME] Processing time: {report.processing_time:.3f}s")
        
        # 保存单文件报告到 reports 文件夹
        output_dir = os.path.dirname(output_path)
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_filename = os.path.basename(output_path).replace('.glb', '_report.json')
        report_path = os.path.join(reports_dir, report_filename)
        self._save_single_report(report, report_path)
        
        self.current_report = None
        return report
    
    def _save_single_report(self, report: ProcessingReport, filepath: str):
        """保存单个文件的报告"""
        report_data = {
            "single_file_report": True,
            "timestamp": datetime.now().isoformat(),
            "parameters": self.get_parameters_dict(),
            **report.to_dict()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"[REPORT] Saved report to: {filepath}")
    
    def process_folder(self, folder_path: str, output_folder: Optional[str] = None) -> BatchReport:
        """
        批量处理文件夹中的所有GLB文件
        
        Args:
            folder_path: 输入文件夹路径
            output_folder: 输出文件夹路径,如果为None则输出到同一文件夹
        
        Returns:
            BatchReport: 批处理报告
        """
        folder_path = os.path.abspath(folder_path)
        
        if output_folder is None:
            output_folder = folder_path
        else:
            output_folder = os.path.abspath(output_folder)
            os.makedirs(output_folder, exist_ok=True)
        
        batch_report = BatchReport()
        batch_report.input_folder = folder_path
        batch_report.output_folder = output_folder
        batch_report.parameters = self.get_parameters_dict()
        batch_report.timestamp = datetime.now().isoformat()
        
        glb_files = list(Path(folder_path).glob("*.glb"))
        glb_files = [f for f in glb_files if not f.name.startswith("post_")]
        
        print(f"\n{'#'*60}")
        print(f"[BATCH] Batch Processing Mode")
        print(f"[INPUT] Input Folder: {folder_path}")
        print(f"[OUTPUT] Output Folder: {output_folder}")
        print(f"[COUNT] Found {len(glb_files)} GLB files")
        print(f"{'#'*60}")
        
        for i, glb_file in enumerate(glb_files, 1):
            print(f"\n[PROGRESS] {i}/{len(glb_files)}")
            
            output_name = f"post_{glb_file.name}"
            output_path = os.path.join(output_folder, output_name)
            
            report = self.process_file(str(glb_file), output_path)
            batch_report.add_report(report)
        
        summary = batch_report.calculate_average()
        print(f"\n{'#'*60}")
        print(f"[SUMMARY] Batch Processing Completed")
        print(f"  Total files: {summary.get('total_files', 0)}")
        print(f"  Successful: {summary.get('successful_files', 0)}")
        print(f"  Failed: {summary.get('failed_files', 0)}")
        if summary.get('average_drop_ratio'):
            print(f"\n  Average Drop Ratio:")
            print(f"    Vertices: {summary['average_drop_ratio']['vertices_drop_percent']}%")
            print(f"    Edges: {summary['average_drop_ratio']['edges_drop_percent']}%")
            print(f"    Faces: {summary['average_drop_ratio']['faces_drop_percent']}%")
        print(f"{'#'*60}")
        
        # 保存批处理报告到 reports 文件夹
        reports_dir = os.path.join(output_folder, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, "batch_report.json")
        batch_report.save(report_path)
        
        return batch_report


def parse_arguments():
    """解析命令行参数"""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(
        description="GLB Post-Processing Tool for Blender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Process single file
  blender --background --python PostProcessing.py -- --input model.glb

  # Batch process folder
  blender --background --python PostProcessing.py -- --batch --input ./models/

  # Custom parameters
  blender --background --python PostProcessing.py -- --input model.glb --no-smooth --decimate-ratio 0.5

  # Specify output path
  blender --background --python PostProcessing.py -- --input model.glb --output processed_model.glb
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file path (single file mode) or folder path (batch processing mode)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path or folder path (optional, default is post_originalname.glb)"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Enable batch processing mode, process all GLB files in the folder"
    )
    
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=DEFAULT_MERGE_DISTANCE,
        help=f"Merge vertices distance threshold (default: {DEFAULT_MERGE_DISTANCE})"
    )
    parser.add_argument(
        "--decimate-ratio",
        type=float,
        default=DEFAULT_DECIMATE_RATIO,
        help=f"Decimate simplification ratio, between 0 and 1 (default: {DEFAULT_DECIMATE_RATIO})"
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=DEFAULT_SMOOTH_ITERATIONS,
        help=f"Smooth iterations (default: {DEFAULT_SMOOTH_ITERATIONS})"
    )
    
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable surface smoothing"
    )
    parser.add_argument(
        "--no-decimate",
        action="store_true",
        help="Disable Decimate simplification"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merging duplicate vertices"
    )
    
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    
    processor = MeshProcessor(
        merge_distance=args.merge_distance,
        decimate_ratio=args.decimate_ratio,
        smooth_iterations=args.smooth_iterations,
        enable_smooth=not args.no_smooth,
        enable_decimate=not args.no_decimate,
        enable_merge=not args.no_merge
    )
    
    print("\n" + "="*60)
    print("GLB Post-Processing Tool")
    print("="*60)
    print(f"Configuration Parameters:")
    print(f"  - Merge Distance: {processor.merge_distance}")
    print(f"  - Decimate Ratio: {processor.decimate_ratio}")
    print(f"  - Smooth Iterations: {processor.smooth_iterations}")
    print(f"  - Enable Merge: {processor.enable_merge}")
    print(f"  - Enable Smooth: {processor.enable_smooth}")
    print(f"  - Enable Decimate: {processor.enable_decimate}")
    print("="*60)
    
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"[ERROR] Batch processing mode requires a folder path: {args.input}")
            sys.exit(1)
        processor.process_folder(args.input, args.output)
    else:
        if not os.path.isfile(args.input):
            print(f"[ERROR] File does not exist: {args.input}")
            sys.exit(1)
        report = processor.process_file(args.input, args.output)
        if not report.success:
            sys.exit(1)


if __name__ == "__main__":
    main()
