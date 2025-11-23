
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference3d.py
A unified 3D inference dispatcher that supports two backends:
- TRELLIS (microsoft/TRELLIS-image-large)
- Hunyuan3D-2 (hy3dgen pipelines)

Design highlights:
- Single-file integration with environment isolation by directly invoking target conda env's python binary (no `conda run`).
- Streaming logs from worker subprocess (no stdout buffering), with timeout control.
- Strong validation for Hunyuan3D-2 local model path to avoid accidental HuggingFace downloads.
- Optional PYTHONPATH injection for hy3dgen repo, plus worker-side `sys.path` fallback.
- CLI & Python API use.

Author: Chris Chan & Copilot
"""

import os
import sys
import json
import time
import uuid
import shlex
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


# -----------------------------
# Utils: time, logging, mkdir
# -----------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[{_now()}] {msg}", flush=True)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def analyze_glb(path: str) -> Dict[str, Any]:
    """Lightweight GLB inspector: returns buffer and image counts/sizes.

    Reads GLB header, parses JSON chunk and inspects `buffers` and `images` entries.
    This mirrors the controller's analyzer so worker logs include the same diagnostics.
    """
    import struct
    import json as _json
    import base64

    info: Dict[str, Any] = {"buffers": [], "images": [], "meshes": 0}
    try:
        with open(path, 'rb') as f:
            data = f.read()
        if len(data) < 12:
            return {"error": "file too small"}
        magic, version, length = struct.unpack_from('<4sII', data, 0)
        if magic != b'glTF':
            return {"error": "not glb"}
        offset = 12
        json_chunk = None
        bin_chunk = None
        while offset + 8 <= len(data):
            chunk_len, chunk_type = struct.unpack_from('<I4s', data, offset)
            offset += 8
            chunk_data = data[offset: offset + chunk_len]
            offset += chunk_len
            if chunk_type == b'JSON':
                try:
                    json_chunk = _json.loads(chunk_data.decode('utf-8'))
                except Exception:
                    json_chunk = None
            elif chunk_type == b'BIN\x00':
                bin_chunk = chunk_data

        if json_chunk is None:
            return {"error": "no json chunk"}

        for b in json_chunk.get('buffers', []):
            blen = b.get('byteLength')
            info['buffers'].append({'byteLength': blen})

        for im in json_chunk.get('images', []):
            uri = im.get('uri')
            if not uri:
                info['images'].append({'uri': None, 'note': 'bufferView'})
            elif uri.startswith('data:'):
                try:
                    header, b64 = uri.split(',', 1)
                    raw = base64.b64decode(b64)
                    info['images'].append({'uri': 'data', 'size': len(raw)})
                except Exception:
                    info['images'].append({'uri': 'data', 'size': None})
            else:
                info['images'].append({'uri': uri, 'size': None})

        info['meshes'] = len(json_chunk.get('meshes', []))
        return info
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# TRELLIS inline runner
# -----------------------------


def run_trellis_inline(
    prompt: str,
    output_dir: str,
    seed: int = 1,
    trellis_model: str = "microsoft/TRELLIS-image-large",
    # 新增：文本/图像输入选择，与 Hunyuan3D-2 保持一致
    input_type: str = "image",
    # 新增：文生图模型（优先使用本地 Diffusers 管线目录）
    t2i_model: str = "/disk2/licheng/models/HunyuanDiT-v1.1-Diffusers-Distilled",
    # 新增：可选去背景，保持与 Hunyuan3D-2 一致
    do_rembg_if_rgb: bool = True,
    render_target: str = "mesh",  # 'gaussian' | 'radiance_field' | 'mesh'
    render_channel: str = "normal",  # 'color' | 'normal'
    attn_backend: Optional[str] = None,  # 'flash-attn' | 'xformers' | None
    spconv_algo: str = "native",  # 'native' | 'auto'
    sparse_structure_steps: Optional[int] = None,
    sparse_structure_cfg: Optional[float] = None,
    slat_steps: Optional[int] = None,
    slat_cfg: Optional[float] = None,
    texture_size: int = 1024,
    simplify_ratio: float = 0.95,
    cuda_visible_devices: Optional[str] = None,
    render_video: bool = False,
) -> Dict[str, Any]:
    """
    TRELLIS inference: supports image-to-3D or text-to-3D (Text→Image→3D).
    """
    # Env controls
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    if attn_backend:
        os.environ["ATTN_BACKEND"] = attn_backend
    os.environ["SPCONV_ALGO"] = spconv_algo
    os.environ["NVDIFRAST_USE_EGL"] = "1"

    import warnings
    warnings.filterwarnings("ignore", message="xFormers is available")

    from pathlib import Path  # 提前导入，避免未定义 Path 的错误
    import sys
    import torch
    from PIL import Image

    # trellis
    sys.path.insert(0, "/disk2/licheng/code/ARIN5201-CV-FinalProject/TRELLIS")
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    # 使用同一套 rembg 以与 Hunyuan3D-2 保持一致
    try:
        from hy3dgen.rembg import BackgroundRemover
    except Exception:
        # 在某些环境（例如只安装了 TRELLIS 的虚拟环境）可能没有 hy3dgen 包。
        # 为了兼容性，提供一个轻量回退实现 —— 不做去背景，只返回原图。
        class BackgroundRemover:
            def __init__(self):
                pass
            def __call__(self, img):
                # 返回原始 PIL.Image，不做处理
                return img
        log("[WARN] hy3dgen.rembg not available in this environment; background removal disabled.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bg = BackgroundRemover()
    out_dir = ensure_dir(Path(output_dir))

    # ---- 统一输入处理：支持 image / text，与 Hunyuan3D-2 保持一致 ----
    image = None
    if input_type.lower() == "image":
        stem = Path(prompt).stem
        log(f"Reading image: {prompt}")
        image = Image.open(prompt)
        if image.mode == "RGB" and do_rembg_if_rgb:
            log("Running background remover (image is RGB)...")
            image = bg(image)
    elif input_type.lower() == "text":
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Empty text prompt for TRELLIS.")
        # 文生图：优先本地 Diffusers 管线目录（含 model_index.json）
        t2i_dir = Path(t2i_model)
        use_local = t2i_dir.is_dir() and (t2i_dir / "model_index.json").exists()
        log(f"[T2I] Using {'local' if use_local else 'remote'} HunyuanDiT-v1.1-Distilled for text→image (TRELLIS).")
        from diffusers import HunyuanDiTPipeline
        if use_local:
            pipe = HunyuanDiTPipeline.from_pretrained(
                str(t2i_dir), torch_dtype=torch.float16, local_files_only=True
            ).to(device)
        else:
            pipe = HunyuanDiTPipeline.from_pretrained(
                "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
                torch_dtype=torch.float16
            ).to(device)
        log(f"[T2I] Generating image for prompt: '{prompt}'")
        image_t2i = pipe(prompt).images[0]  # PIL RGB
        if image_t2i.mode == "RGB" and do_rembg_if_rgb:
            log("[T2I] Running background remover on generated RGB image...")
            image_t2i = bg(image_t2i)
        image = image_t2i  # TRELLIS 原始读取为 PIL.Image
        stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in prompt)[:40] or uuid.uuid4().hex[:8]
        t2i_png = out_dir / "temp.png"
        try:
            image.save(str(t2i_png))
            log(f"[T2I] Saved generated image: {t2i_png}")
        except Exception as e:
            log(f"[T2I] Save temp.png failed: {e}")
    else:
        raise ValueError("input_type must be 'image' or 'text'.")

    glb_path = out_dir / f"{stem}_trellis.glb"
    mp4_path = out_dir / f"{stem}_trellis_{render_target}.mp4"
    ply_path = out_dir / f"{stem}_trellis_gs.ply"

    log("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(trellis_model)
    pipeline.cuda()

    # Always pass dicts (avoid NoneType merging inside Trellis)
    sparse_structure_sampler_params: Dict[str, Any] = {}
    if sparse_structure_steps is not None:
        sparse_structure_sampler_params["steps"] = int(sparse_structure_steps)
    if sparse_structure_cfg is not None:
        sparse_structure_sampler_params["cfg_strength"] = float(sparse_structure_cfg)

    slat_sampler_params: Dict[str, Any] = {}
    if slat_steps is not None:
        slat_sampler_params["steps"] = int(slat_steps)
    if slat_cfg is not None:
        slat_sampler_params["cfg_strength"] = float(slat_cfg)

    log("Running TRELLIS inference...")
    outputs = pipeline.run(
        image,
        seed=seed,
        sparse_structure_sampler_params=sparse_structure_sampler_params,
        slat_sampler_params=slat_sampler_params,
    )

    # outputs: 'gaussian', 'radiance_field', 'mesh'
    if render_video:
        try:
            os.environ["NVDIFRAST_USE_EGL"] = "1"
            from trellis.utils import render_utils
            log(f"Rendering {render_target} video ({render_channel})...")
            if render_target not in outputs:
                raise ValueError(f"render_target '{render_target}' not in outputs: {list(outputs.keys())}")
            video = render_utils.render_video(outputs[render_target][0])[render_channel]
            import imageio
            imageio.mimsave(str(mp4_path), video, fps=30)
        except Exception as e:
            log(f"[WARN] Render video failed: {e}. Continue to export GLB/PLY.")
            mp4_path = None  # 标记未生成

    log("Exporting GLB...")
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=simplify_ratio,
        texture_size=texture_size,
    )
    # Try to force embedded images when exporter supports it; fall back to default export.
    try:
        # common variants: embed_images, embed
        try:
            glb.export(str(glb_path), embed_images=True)
        except TypeError:
            try:
                glb.export(str(glb_path), embed=True)
            except TypeError:
                glb.export(str(glb_path))
    except Exception as e:
        log(f"[WARN] glb.export with embed args failed: {e}; falling back to default export.")
        glb.export(str(glb_path))

    # Analyze exported GLB for debugging parity issues
    try:
        analysis = analyze_glb(str(glb_path))
        log(f"GLB analysis (trellis): {analysis}")
    except Exception as e:
        log(f"GLB analysis failed: {e}")

    log("Saving Gaussian PLY...")
    outputs["gaussian"][0].save_ply(str(ply_path))

    artifacts = {
        "glb": str(glb_path),
        "ply": str(ply_path),
    }
    # 文本入口时，附带 t2i 结果
    if input_type.lower() == "text":
        png_guess = out_dir / "temp.png"
        if png_guess.exists():
            artifacts["t2i_image"] = str(png_guess)

    log("Done.")

    # include artifact sizes and analysis in result to help controller compare runs
    try:
        art_sizes = {}
        if glb_path.exists():
            art_sizes['glb'] = glb_path.stat().st_size
        if ply_path.exists():
            art_sizes['ply'] = ply_path.stat().st_size
        result['artifact_sizes'] = art_sizes
        if 'analysis' in locals():
            result['glb_analysis'] = analysis
    except Exception:
        pass

    result = {
        "backend": "trellis",
        "inputs": {
            "input": str(prompt),
            "input_type": input_type,
            "seed": seed,
        },
        "artifacts": artifacts,
        "params": {
            "render_target": render_target,
            "render_channel": render_channel,
            "texture_size": texture_size,
            "simplify_ratio": simplify_ratio,
            "attn_backend": attn_backend,
            "spconv_algo": spconv_algo,
            "sparse_structure_sampler_params": sparse_structure_sampler_params,
            "slat_sampler_params": slat_sampler_params,
            "cuda_visible_devices": cuda_visible_devices,
            "render_video": render_video,
            "t2i_model": t2i_model,
            "do_rembg_if_rgb": do_rembg_if_rgb,
        },
    }
    return result


# -----------------------------
# Hunyuan3D-2 inline runner
# -----------------------------
def run_hunyuan3d2_inline(
    prompt: str,                             
    output_dir: str,
    model_path: str,
    input_type: str = "image",               
    do_rembg_if_rgb: bool = True,
    repo_dir: Optional[str] = None,          
    cuda_visible_devices: Optional[str] = None,
    # 文生图模型目录：若是本地 Diffusers 管线目录（含 model_index.json）将优先使用；否则回落到官方 repo_id
    t2i_model: str = "/disk2/licheng/models/HunyuanDiT-v1.1-Diffusers-Distilled",
    seed: int = 1234,
    octree_resolution: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    mc_algo: str = "mc",
) -> Dict[str, Any]:
    """
    Hunyuan3D-2 inference: supports image-to-3D or text-to-3D (Text→Image→3D).
    """
    from pathlib import Path
    # Env controls
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    # 仓库路径兜底
    if repo_dir:
        p = Path(repo_dir)
        if not p.exists():
            raise FileNotFoundError(f"repo_dir does not exist: {repo_dir}")
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

    # 正常导入 hy3dgen
    import torch
    from PIL import Image
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    # 检查模型目录结构（3D 权重）
    root = Path(model_path)
    dit_ckpt = root / "hunyuan3d-dit-v2-0" / "model.fp16.safetensors"
    paint_dir = root / "hunyuan3d-paint-v2-0"
    if not dit_ckpt.exists():
        raise FileNotFoundError(f"[Hunyuan3D-2] 缺少形状生成权重文件: {dit_ckpt}")
    if not paint_dir.exists():
        raise FileNotFoundError(f"[Hunyuan3D-2] 缺少纹理生成目录: {paint_dir}")

    out_dir = ensure_dir(Path(output_dir))
    bg = BackgroundRemover()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(int(seed))

    # 1) 解析输入：图像或文本
    image = None
    if input_type.lower() == "image":
        image_path = prompt
        stem = Path(image_path).stem
        log(f"Reading image: {image_path}")
        image = Image.open(image_path).convert("RGBA")
        if image.mode == "RGB" and do_rembg_if_rgb:
            log("Running background remover (image is RGB)...")
            image = bg(image)

    elif input_type.lower() == "text":
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Empty text prompt.")
        # ---- 文生图：优先使用本地 Diffusers 管线目录（含 model_index.json）；否则使用官方 repo_id ----
        t2i_dir = Path(t2i_model)
        use_local = t2i_dir.is_dir() and (t2i_dir / "model_index.json").exists()

        log(f"[T2I] Using {'local' if use_local else 'remote'} HunyuanDiT-v1.1-Distilled for text→image.")
        from diffusers import HunyuanDiTPipeline  # 官方文档推荐用法 [1](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)

        if use_local:
            pipe = HunyuanDiTPipeline.from_pretrained(
                str(t2i_dir), torch_dtype=torch.float16, local_files_only=True
            ).to(device)
        else:
            pipe = HunyuanDiTPipeline.from_pretrained(
                "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",  # 官方模型 repo_id
                torch_dtype=torch.float16
            ).to(device)

        # 官方示例：pipe(prompt).images[0] 生成 PIL.Image 
        log(f"[T2I] Generating image for prompt: '{prompt}'")
        image_t2i = pipe(prompt).images[0]  # PIL RGB

        # 可选：去背景，与图生3D路径保持一致
        if image_t2i.mode == "RGB" and do_rembg_if_rgb:
            log("[T2I] Running background remover on generated RGB image...")
            image_t2i = bg(image_t2i)

        # 转 RGBA 以统一下游
        image = image_t2i.convert("RGBA")

        # 保存 temp.png 到 output
        stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in prompt)[:40] or uuid.uuid4().hex[:8]
        t2i_png = out_dir / "temp.png"
        try:
            image.save(str(t2i_png))
            log(f"[T2I] Saved generated image: {t2i_png}")
        except Exception as e:
            log(f"[T2I] Save temp.png failed: {e}")
    else:
        raise ValueError("input_type must be 'image' or 'text'.")

    # 2) Hunyuan3D-2 形状与贴图
    glb_path = out_dir / f"{stem}_hunyuan3d2.glb"

    log("Loading Hunyuan3D-2 pipelines...")
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    # （保留你的安全检查临时补丁，避免 transformers 在 torch<2.6 且加载 .bin 时拦截）
    try:
        import transformers.utils.import_utils as _iu
        if hasattr(_iu, "check_torch_load_is_safe"):
            _iu.check_torch_load_is_safe = lambda: None
            log("[Patch] Disabled transformers torch.load safety check in-process (temporary).")
    except Exception as _e:
        log(f"[Patch] Failed to disable safety check: {_e}")

    pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)

    # 形状生成参数
    # Build params only with values explicitly provided; otherwise let pipeline use its defaults
    params: Dict[str, Any] = {"generator": gen, "mc_algo": mc_algo}
    if octree_resolution is not None:
        params["octree_resolution"] = int(octree_resolution)
    if num_inference_steps is not None:
        params["num_inference_steps"] = int(num_inference_steps)
    if guidance_scale is not None:
        params["guidance_scale"] = float(guidance_scale)

    log(f"Hunyuan3D-2 shapegen params: {params}")

    log("Running Hunyuan3D-2 shape generation...")
    mesh = pipeline_shapegen(image=image, **params)[0]

    log("Running Hunyuan3D-2 texture painting...")
    mesh = pipeline_texgen(mesh, image=image)

    log("Exporting GLB...")
    mesh.export(str(glb_path))

    artifacts = {"glb": str(glb_path)}
    # 若是文本入口，附带 temp.png
    if input_type.lower() == "text":
        png_guess = out_dir / "temp.png"
        if png_guess.exists():
            artifacts["t2i_image"] = str(png_guess)

    log("Done.")

    return {
        "backend": "hunyuan3d-2",
        "inputs": {"input": prompt, "input_type": input_type, "model_path": str(model_path)},
        "artifacts": artifacts,
        "params": {
            "do_rembg_if_rgb": do_rembg_if_rgb,
            "repo_dir": repo_dir,
            "cuda_visible_devices": cuda_visible_devices,
            "t2i_model": t2i_model,
            "seed": seed,
            "octree_resolution": octree_resolution,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "mc_algo": mc_algo,
        }
    }


# -----------------------------
# Dispatcher: env-aware runner
# -----------------------------
def infer(
    model: str,
    image_path: str,
    output_dir: str,
    input_type: str = "image",
    hunyuan_env: str = "hunyuan3d2",
    trellis_env: str = "trellis",
    hy3dgen_repo: Optional[str] = None,  # repo root containing hy3dgen
    # directly call target conda env python (no conda run)
    python_bin_hy3d2: Optional[str] = "/disk2/licheng/miniconda3/envs/hunyuan3d2/bin/python",
    python_bin_trellis: Optional[str] = "/disk2/licheng/miniconda3/envs/trellis/bin/python",
    spawn_timeout_sec: int = 900,        # 15 minutes timeout
    cuda_visible_devices: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Public API for Flask or external callers.
    If current environment matches the required one, run inline.
    Otherwise, spawn a worker subprocess by invoking the target env's python binary
    (no `conda run`), streaming logs in real-time and enforcing a timeout.
    """
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    required_env = trellis_env if model.lower() == "trellis" else hunyuan_env

    # Prepare output dir
    out_dir = ensure_dir(Path(output_dir))
    meta_path = out_dir / f"meta_{uuid.uuid4().hex}.json"

    # Inline path (env matches)
    if current_env == required_env:
        log(f"Current env '{current_env}' == required '{required_env}', running inline...")
        if model.lower() == "trellis":
            result = run_trellis_inline(
                prompt=image_path,
                output_dir=output_dir,
                input_type=input_type,
                cuda_visible_devices=cuda_visible_devices,
                **kwargs
            )
        elif model.lower() == "hunyuan3d-2":
            if "model_path" not in kwargs:
                raise ValueError("Hunyuan3D-2 requires 'model_path' in kwargs.")
            result = run_hunyuan3d2_inline(
                prompt=image_path,
                output_dir=output_dir,
                repo_dir=hy3dgen_repo,
                input_type=input_type,
                cuda_visible_devices=cuda_visible_devices,
                **kwargs
            )
        else:
            raise ValueError("model should be 'trellis' or 'hunyuan3d-2'")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    # Spawn worker in target env by invoking the target python binary
    log(f"Current env '{current_env}' != required '{required_env}', spawning subprocess via target python...")

    # Select python path
    if model.lower() == "trellis":
        python_bin = python_bin_trellis
    elif model.lower() == "hunyuan3d-2":
        python_bin = python_bin_hy3d2
        # pass repo_dir to worker via kwargs
        if hy3dgen_repo:
            kwargs["repo_dir"] = hy3dgen_repo
    else:
        raise ValueError("model should be 'trellis' or 'hunyuan3d-2'")

    # Validate python bin
    if not python_bin or not Path(python_bin).exists():
        raise FileNotFoundError(f"Target python binary not found: {python_bin}")

    # Build worker command (use -u to disable buffering)
    worker_payload = json.dumps(kwargs)
    
    cmd_list = [
    str(python_bin), "-u", str(os.path.abspath(__file__)),
        "--worker",
        "--model", str(model),
        "--input", str(image_path),
        "--input-type", str(input_type or "image"),
        "--output", str(output_dir),
        "--hunyuan-env", str(hunyuan_env),
        "--trellis-env", str(trellis_env),
        "--kwargs-json", str(worker_payload),
    ]


    # Environment for subprocess
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    # Inject PYTHONPATH for Hunyuan3D-2 (repo root containing hy3dgen)
    if model.lower() == "hunyuan3d-2" and hy3dgen_repo:
        env["PYTHONPATH"] = f"{hy3dgen_repo}:{env.get('PYTHONPATH', '')}"

    log(f"Spawn cmd: {' '.join(shlex.quote(x) for x in cmd_list)}")
    log(f"Env overrides: {{'PYTHONPATH': {env.get('PYTHONPATH')}, 'PYTHONUNBUFFERED': {env.get('PYTHONUNBUFFERED')}, 'CUDA_VISIBLE_DEVICES': {env.get('CUDA_VISIBLE_DEVICES')}}}")

    # Stream logs + timeout
    proc = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    stdout_lines = []
    start = time.time()
    while True:
        if proc.poll() is not None:
            break
        line = proc.stdout.readline()
        if line:
            stdout_lines.append(line)
            print(line, end="")
        if time.time() - start > spawn_timeout_sec:
            proc.kill()
            raise TimeoutError(f"Subprocess timeout after {spawn_timeout_sec}s (env={required_env}).")

    ret = proc.wait()
    if ret != 0:
        tail = "".join(stdout_lines[-50:])
        raise RuntimeError(f"Subprocess failed (env={required_env}). Tail logs:\n{tail}")

    # Parse last valid JSON line from worker stdout
    result = None
    for i in range(len(stdout_lines)-1, -1, -1):
        s = stdout_lines[i].strip()
        try:
            result = json.loads(s)
            break
        except Exception:
            continue
    if result is None:
        raise RuntimeError("Worker did not output valid JSON. Check streamed logs above.")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# -----------------------------
# CLI & Worker entrypoints
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified 3D inference for TRELLIS and Hunyuan3D-2")

    # Common
    p.add_argument("--model", required=True, choices=["trellis", "hunyuan3d-2"], help="Which backend to use.")
    p.add_argument("--input", required=True, help="Path to input image or text prompt.")
    p.add_argument("--input-type", default="image", help="Type of input(image or text).")
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument("--worker", action="store_true", help="Internal: run inside target env.")
    p.add_argument("--hunyuan-env", default="hunyuan3d2", help="Conda env for Hunyuan3D-2.")
    p.add_argument("--trellis-env", default="trellis", help="Conda env for TRELLIS.")
    p.add_argument("--kwargs-json", default="{}", help="Model-specific kwargs as JSON string.")
    p.add_argument("--hy3dgen-repo", default=None, help="Path to Hunyuan3D-2 repo root (contains 'hy3dgen').")
    p.add_argument("--python-bin-hy3d2", default="/disk2/licheng/miniconda3/envs/hunyuan3d2/bin/python", help="Python binary for hunyuan3d2 env.")
    p.add_argument("--python-bin-trellis", default="/disk2/licheng/miniconda3/envs/trellis/bin/python", help="Python binary for trellis env.")
    p.add_argument("--spawn-timeout-sec", type=int, default=900, help="Timeout for worker subprocess in seconds.")
    p.add_argument("--cuda-visible-devices", default=None, help="Optional: set CUDA_VISIBLE_DEVICES (e.g., '0').")

    return p.parse_args()


def main():
    args = parse_args()
    kwargs = json.loads(args.kwargs_json)

    # Worker mode: execute inline in the currently active env
    if args.worker:
        if args.model == "trellis":
            result = run_trellis_inline(
                prompt=args.input,
                output_dir=args.output,
                input_type=args.input_type,
                cuda_visible_devices=args.cuda_visible_devices,
                **kwargs
            )
        else:
            if "model_path" not in kwargs:
                raise ValueError("Hunyuan3D-2 requires 'model_path' in kwargs.")
            # Avoid duplicate passing of repo_dir (pop from kwargs, pass explicitly)
            repo_dir = kwargs.pop("repo_dir", None)
            result = run_hunyuan3d2_inline(
                prompt=args.input,
                output_dir=args.output,
                input_type=args.input_type,
                repo_dir=repo_dir,
                cuda_visible_devices=args.cuda_visible_devices,
                **kwargs
            )
        # Print result JSON on the last line for controller to parse
        print(json.dumps(result, ensure_ascii=False))
        return

    # Controller mode: decide inline vs spawn
    result = infer(
        model=args.model,
        image_path=args.input,
        output_dir=args.output,
        input_type=args.input_type,
        hunyuan_env=args.hunyuan_env,
        trellis_env=args.trellis_env,
        hy3dgen_repo=args.hy3dgen_repo,
        python_bin_hy3d2=args.python_bin_hy3d2,
        python_bin_trellis=args.python_bin_trellis,
        spawn_timeout_sec=args.spawn_timeout_sec,
        cuda_visible_devices=args.cuda_visible_devices,
        **kwargs
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
    