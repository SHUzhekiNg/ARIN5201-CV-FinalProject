
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


# -----------------------------
# TRELLIS inline runner
# -----------------------------
def run_trellis_inline(
    image_path: str,
    output_dir: str,
    seed: int = 1,
    trellis_model: str = "microsoft/TRELLIS-image-large",
    render_target: str = "mesh",        # 'gaussian' | 'radiance_field' | 'mesh'
    render_channel: str = "normal",     # 'color' | 'normal'
    attn_backend: Optional[str] = None, # 'flash-attn' | 'xformers' | None
    spconv_algo: str = "native",        # 'native' | 'auto'
    sparse_structure_steps: Optional[int] = None,
    sparse_structure_cfg: Optional[float] = None,
    slat_steps: Optional[int] = None,
    slat_cfg: Optional[float] = None,
    texture_size: int = 1024,
    simplify_ratio: float = 0.95,
    cuda_visible_devices: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute TRELLIS inference inside the current Python process (assumes trellis env).
    Returns dict with output file paths.
    """
    # Env controls
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    if attn_backend:
        os.environ['ATTN_BACKEND'] = attn_backend
    os.environ['SPCONV_ALGO'] = spconv_algo

    # Optional: silence noisy warnings (xFormers availability)
    import warnings
    warnings.filterwarnings("ignore", message="xFormers is available")

    from PIL import Image
    import imageio
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import render_utils, postprocessing_utils

    out_dir = ensure_dir(Path(output_dir))
    stem = Path(image_path).stem
    glb_path = out_dir / f"{stem}_trellis.glb"
    mp4_path = out_dir / f"{stem}_trellis_{render_target}.mp4"
    ply_path = out_dir / f"{stem}_trellis_gs.ply"

    log("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(trellis_model)
    pipeline.cuda()

    log(f"Reading image: {image_path}")
    image = Image.open(image_path)

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
    # outputs keys: 'gaussian', 'radiance_field', 'mesh' (each a list)

    log(f"Rendering {render_target} video ({render_channel})...")
    if render_target not in outputs:
        raise ValueError(f"render_target '{render_target}' not in outputs: {list(outputs.keys())}")
    video = render_utils.render_video(outputs[render_target][0])[render_channel]
    imageio.mimsave(str(mp4_path), video, fps=30)

    log("Exporting GLB...")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=simplify_ratio,
        texture_size=texture_size,
    )
    glb.export(str(glb_path))

    log("Saving Gaussian PLY...")
    outputs['gaussian'][0].save_ply(str(ply_path))

    result = {
        "backend": "trellis",
        "inputs": {
            "image_path": str(image_path),
            "seed": seed,
        },
        "artifacts": {
            "mp4": str(mp4_path),
            "glb": str(glb_path),
            "ply": str(ply_path),
        },
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
        }
    }
    return result


# -----------------------------
# Hunyuan3D-2 inline runner
# -----------------------------
def run_hunyuan3d2_inline(
    image_path: str,
    output_dir: str,
    model_path: str,
    do_rembg_if_rgb: bool = True,
    repo_dir: Optional[str] = None,        # repo root (contains 'hy3dgen' directory)
    cuda_visible_devices: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute Hunyuan3D-2 inference inside current Python process (assumes hunyuan3d2 env).
    Returns dict with output file paths.
    """

    # Env controls
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    # Fallback: inject repo_dir into sys.path before importing hy3dgen
    if repo_dir:
        if not Path(repo_dir).exists():
            raise FileNotFoundError(f"repo_dir does not exist: {repo_dir}")
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

    # Proper import: hy3dgen is top-level package in the repo
    from PIL import Image
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    # Strong validation for local model path structure
    root = Path(model_path)  # e.g., /disk2/licheng/models/Hunyuan3D-2/
    dit_ckpt = root / "hunyuan3d-dit-v2-0" / "model.fp16.safetensors"
    paint_dir = root / "hunyuan3d-paint-v2-0"
    if not dit_ckpt.exists():
        raise FileNotFoundError(
            f"[Hunyuan3D-2] Missing shape-generation checkpoint: {dit_ckpt}\n"
            f"Expected file at: {dit_ckpt}\n"
            f"Current model_path: {model_path}"
        )
    if not paint_dir.exists():
        raise FileNotFoundError(
            f"[Hunyuan3D-2] Missing texture-generation directory: {paint_dir}\n"
            f"Expected directory at: {paint_dir}\n"
            f"Current model_path: {model_path}"
        )

    out_dir = ensure_dir(Path(output_dir))
    stem = Path(image_path).stem
    glb_path = out_dir / f"{stem}_hunyuan3d2.glb"

    log("Loading Hunyuan3D-2 pipelines...")
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)

    log(f"Reading image: {image_path}")
    image = Image.open(image_path).convert("RGBA")

    if image.mode == "RGB" and do_rembg_if_rgb:
        log("Running background remover (image is RGB)...")
        rembg = BackgroundRemover()
        image = rembg(image)

    log("Running Hunyuan3D-2 shape generation...")
    mesh = pipeline_shapegen(image=image)[0]
    log("Running Hunyuan3D-2 texture painting...")
    mesh = pipeline_texgen(mesh, image=image)

    log("Exporting GLB...")
    mesh.export(str(glb_path))

    result = {
        "backend": "hunyuan3d-2",
        "inputs": {
            "image_path": str(image_path),
            "model_path": str(model_path),
        },
        "artifacts": {
            "glb": str(glb_path),
        },
        "params": {
            "do_rembg_if_rgb": do_rembg_if_rgb,
            "repo_dir": repo_dir,
            "cuda_visible_devices": cuda_visible_devices,
        }
    }
    return result


# -----------------------------
# Dispatcher: env-aware runner
# -----------------------------
def infer(
    model: str,
    image_path: str,
    output_dir: str,
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
                image_path=image_path,
                output_dir=output_dir,
                cuda_visible_devices=cuda_visible_devices,
                **kwargs
            )
        elif model.lower() == "hunyuan3d-2":
            if "model_path" not in kwargs:
                raise ValueError("Hunyuan3D-2 requires 'model_path' in kwargs.")
            result = run_hunyuan3d2_inline(
                image_path=image_path,
                output_dir=output_dir,
                repo_dir=hy3dgen_repo,
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
        python_bin, "-u", os.path.abspath(__file__),
        "--worker",
        "--model", model,
        "--image", image_path,
        "--output", output_dir,
        "--hunyuan-env", hunyuan_env,
        "--trellis-env", trellis_env,
        "--kwargs-json", worker_payload,
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
    p.add_argument("--image", required=True, help="Path to input image.")
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
                image_path=args.image,
                output_dir=args.output,
                cuda_visible_devices=args.cuda_visible_devices,
                **kwargs
            )
        else:
            if "model_path" not in kwargs:
                raise ValueError("Hunyuan3D-2 requires 'model_path' in kwargs.")
            # Avoid duplicate passing of repo_dir (pop from kwargs, pass explicitly)
            repo_dir = kwargs.pop("repo_dir", None)
            result = run_hunyuan3d2_inline(
                image_path=args.image,
                output_dir=args.output,
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
        image_path=args.image,
        output_dir=args.output,
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
    