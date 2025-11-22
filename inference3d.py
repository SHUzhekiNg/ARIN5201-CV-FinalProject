
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference3d.py
A unified 3D inference dispatcher that supports two backends:
- TRELLIS (microsoft/TRELLIS-image-large)
- Hunyuan3D-2 (hy3dgen pipelines)

Features:
- Single-file integration with environment isolation via `conda run`.
- CLI & Python API use.
- Returns structured JSON describing output artifacts (GLB/MP4/PLY paths).
"""
import os
import sys
import json
import time
import uuid
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


# -----------------------------
# Utility: Safe mkdir, logging
# -----------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[{_now()}] {msg}", flush=True)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------
# Inline runners (import inside)
# -----------------------------
def run_trellis_inline(
    image_path: str,
    output_dir: str,
    seed: int = 1,
    trellis_model: str = "microsoft/TRELLIS-image-large",
    render_target: str = "mesh",        # 'gaussian' | 'radiance_field' | 'mesh'
    render_channel: str = "normal",     # 'color' | 'normal'
    attn_backend: Optional[str] = None, # 'flash-attn' | 'xformers'
    spconv_algo: str = "native",        # 'native' | 'auto'
    sparse_structure_steps: Optional[int] = None,
    sparse_structure_cfg: Optional[float] = None,
    slat_steps: Optional[int] = None,
    slat_cfg: Optional[float] = None,
    texture_size: int = 1024,
    simplify_ratio: float = 0.95,
) -> Dict[str, Any]:
    """
    Execute TRELLIS inference inside the current Python process (assumes trellis env).
    Returns dict with output file paths.
    """
    # Environment hints
    if attn_backend:
        os.environ['ATTN_BACKEND'] = attn_backend
    os.environ['SPCONV_ALGO'] = spconv_algo

    from PIL import Image
    import imageio
    from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
    from TRELLIS.trellis.utils import render_utils, postprocessing_utils

    out_dir = ensure_dir(Path(output_dir))
    stem = Path(image_path).stem
    glb_path = out_dir / f"{stem}_trellis.glb"
    mp4_path = out_dir / f"{stem}_trellis_{render_target}.mp4"
    ply_path = out_dir / f"{stem}_trellis_gs.ply"

    # Load pipeline
    log("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(trellis_model)
    pipeline.cuda()

    # Load image
    log(f"Reading image: {image_path}")
    image = Image.open(image_path)

    # Optional sampler params
    sparse_structure_sampler_params = {}
    if (sparse_structure_steps is not None) or (sparse_structure_cfg is not None):
        if sparse_structure_steps is not None:
            sparse_structure_sampler_params["steps"] = int(sparse_structure_steps)
        if sparse_structure_cfg is not None:
            sparse_structure_sampler_params["cfg_strength"] = float(sparse_structure_cfg)

    slat_sampler_params = {}
    if (slat_steps is not None) or (slat_cfg is not None):
        if slat_steps is not None:
            slat_sampler_params["steps"] = int(slat_steps)
        if slat_cfg is not None:
            slat_sampler_params["cfg_strength"] = float(slat_cfg)

    # 运行
    outputs = pipeline.run(
        image,
        seed=seed,
        sparse_structure_sampler_params=sparse_structure_sampler_params,
        slat_sampler_params=slat_sampler_params,
    )

    # outputs keys: 'gaussian', 'radiance_field', 'mesh' (each a list)

    # Render video
    log(f"Rendering {render_target} video ({render_channel})...")
    if render_target not in outputs:
        raise ValueError(f"render_target '{render_target}' not in outputs: {list(outputs.keys())}")
    video = render_utils.render_video(outputs[render_target][0])[render_channel]
    imageio.mimsave(str(mp4_path), video, fps=30)

    # Export GLB
    log("Exporting GLB...")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=simplify_ratio,
        texture_size=texture_size,
    )
    glb.export(str(glb_path))

    # Save Gaussian PLY
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
        }
    }
    return result


def run_hunyuan3d2_inline(
    image_path: str,
    output_dir: str,
    model_path: str,
    do_rembg_if_rgb: bool = True,
) -> Dict[str, Any]:
    """
    Execute Hunyuan3D-2 inference inside current Python process (assumes hunyuan3d2 env).
    Returns dict with output file paths.
    """
    from PIL import Image
    from Hunyuan3D_2.hy3dgen.rembg import BackgroundRemover
    from Hunyuan3D_2.hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from Hunyuan3D_2.hy3dgen.texgen import Hunyuan3DPaintPipeline

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
            "do_rembg_if_rgb": do_rembg_if_rgb
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
    conda_sh: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Public API for Flask or external callers.
    If current environment matches the required one, run inline.
    Otherwise, spawn via `conda run -n <env>` as a worker.

    Parameters:
    - model: 'hunyuan3d-2' | 'trellis'
    - image_path: input image path
    - output_dir: where to write outputs
    - hunyuan_env: conda env name for Hunyuan3D-2
    - trellis_env: conda env name for TRELLIS
    - conda_sh: optional path to conda.sh for `source` (e.g., /opt/anaconda3/etc/profile.d/conda.sh)

    kwargs: forwarded to the inline functions (model-specific)
    """
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    required_env = trellis_env if model.lower() == "trellis" else hunyuan_env

    # Prepare output dir
    out_dir = ensure_dir(Path(output_dir))
    meta_path = out_dir / f"meta_{uuid.uuid4().hex}.json"

    # If already in the right env, run inline
    if current_env == required_env:
        log(f"Current env '{current_env}' == required '{required_env}', running inline...")
        if model.lower() == "trellis":
            result = run_trellis_inline(image_path=image_path, output_dir=output_dir, **kwargs)
        elif model.lower() == "hunyuan3d-2":
            if "model_path" not in kwargs:
                raise ValueError("Hunyuan3D-2 requires 'model_path' in kwargs.")
            result = run_hunyuan3d2_inline(image_path=image_path, output_dir=output_dir, **kwargs)
        else:
            raise ValueError("model should be 'trellis' or 'hunyuan3d-2'")

        # write meta
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    # Otherwise, spawn worker in target env
    log(f"Current env '{current_env}' != required '{required_env}', spawning subprocess via conda run...")

    # Build command line args for the worker
    worker_args = [
        sys.executable,  # we will replace this with 'python' inside conda run
        os.path.abspath(__file__),
        "--worker",
        "--model", model,
        "--image", image_path,
        "--output", output_dir,
        "--hunyuan-env", hunyuan_env,
        "--trellis-env", trellis_env,
    ]
    # Forward kwargs as JSON
    worker_payload = json.dumps(kwargs)
    worker_args += ["--kwargs-json", worker_payload]

    cmd_core = f'conda run -n {required_env} python "{os.path.abspath(__file__)}" --worker ' \
               f'--model {model} --image "{image_path}" --output "{output_dir}" ' \
               f'--hunyuan-env "{hunyuan_env}" --trellis-env "{trellis_env}" ' \
               f'--kwargs-json \'{worker_payload}\''

    # If conda_sh provided, source it to ensure `conda` is available
    if conda_sh:
        full_cmd = f'source "{conda_sh}" && {cmd_core}'
        proc = subprocess.run(full_cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    else:
        # assumes `conda` is already in PATH
        proc = subprocess.run(cmd_core, shell=True, capture_output=True, text=True)

    if proc.returncode != 0:
        log(proc.stdout)
        log(proc.stderr)
        raise RuntimeError(f"Subprocess failed (env={required_env}). See logs above.")

    # Worker prints JSON to stdout
    try:
        result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        raise RuntimeError(f"Worker did not return valid JSON. Raw stdout:\n{proc.stdout}")

    # Persist meta for traceability
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# -----------------------------
# CLI entry (also worker mode)
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified 3D inference for TRELLIS and Hunyuan3D-2")
    p.add_argument("--model", required=True, choices=["trellis", "hunyuan3d-2"], help="Which backend to use.")
    p.add_argument("--image", required=True, help="Path to input image.")
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument("--worker", action="store_true", help="Internal: run inside target env.")
    p.add_argument("--hunyuan-env", default="hunyuan3d2", help="Conda env for Hunyuan3D-2.")
    p.add_argument("--trellis-env", default="trellis", help="Conda env for TRELLIS.")
    p.add_argument("--conda-sh", default=None, help="Optional: path to conda.sh to source before conda run.")

    # Model-specific options via JSON (to avoid huge CLI)
    p.add_argument("--kwargs-json", default="{}", help="Model-specific kwargs as JSON string.")

    return p.parse_args()


def main():
    args = parse_args()
    kwargs = json.loads(args.kwargs_json)

    if args.worker:
        # Execute inline inside the target env and print JSON to stdout for the controller
        if args.model == "trellis":
            result = run_trellis_inline(
                image_path=args.image,
                output_dir=args.output,
                **kwargs
            )
        else:
            if "model_path" not in kwargs:
                raise ValueError("Hunyuan3D-2 requires 'model_path' in kwargs.")
            result = run_hunyuan3d2_inline(
                image_path=args.image,
                output_dir=args.output,
                **kwargs
            )
        print(json.dumps(result, ensure_ascii=False))
        return

    # Controller mode: infer() will decide whether to run inline or spawn
    result = infer(
        model=args.model,
        image_path=args.image,
        output_dir=args.output,
        hunyuan_env=args.hunyuan_env,
        trellis_env=args.trellis_env,
        conda_sh=args.conda_sh,
        **kwargs
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
