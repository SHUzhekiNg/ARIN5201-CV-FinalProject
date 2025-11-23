
# app.py
# Flask 服务：图/文生 3D 推理页面 + 实时日志/进度 + 文件下载
import os
import sys
import json
import uuid
import queue
import shutil
import signal
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, Response, send_file, abort, url_for, redirect

# ---- 基本配置（按你的环境修改） ----
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_ROOT = BASE_DIR / "outputs"
INFER_SCRIPT = str(BASE_DIR / "inference3d.py")

# 临时上传用于 viewer 的 GLB 文件
UPLOADED_DIR = OUTPUT_ROOT / "viewer_uploads"
UPLOADED_DIR.mkdir(parents=True, exist_ok=True)
uploaded_files: Dict[str, str] = {}
uploaded_files_lock = threading.Lock()

# 目标 Conda 环境中的 Python 可执行路径（务必正确）
PYTHON_TRELLIS = "/disk2/licheng/miniconda3/envs/trellis/bin/python"
PYTHON_HY3D2 = "/disk2/licheng/miniconda3/envs/hunyuan3d2/bin/python"

# SSE 心跳间隔（秒）
SSE_HEARTBEAT_SEC = 5

# 创建 Flask
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# 确保目录存在
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---- Job 状态容器 ----
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()


def new_job_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def sse_format(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---- 首页 ----
@app.get("/")
def index():
    return render_template("index.html")


# ---- 启动作业 ----
@app.post("/start")
def start_job():
    # 前端值：'trellis' 或 'hunyuan3d-2'
    model = request.form.get("model", "").strip()
    gpu = request.form.get("gpu", "0").strip()               # e.g. "0" 或 "0,1"
    prompt = request.form.get("prompt", "").strip()
    output_dir_root = "/disk2/licheng/code/ARIN5201-CV-FinalProject/outputs"
    image_file = request.files.get("image")

    # 统一输入模式：image/text
    input_type = None         # "image" or "text"
    input_value = None        # 对应路径或文本（prompt）

    # 创建作业目录
    job_id = new_job_id()
    job_dir = Path(output_dir_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image_path = None
    if image_file and image_file.filename:
        suffix = Path(image_file.filename).suffix.lower()
        if suffix not in [".png", ".jpg", ".jpeg", ".webp"]:
            return Response("仅支持 PNG/JPG/WEBP 格式图片", status=400)
        image_path = str(job_dir / f"input{suffix}")
        image_file.save(image_path)
        input_type = "image"
        input_value = image_path
    else:
        # 未上传图片：若提供了 prompt，则按 text 模式
        if prompt:
            input_type = "text"
            input_value = prompt
        else:
            return Response("请上传图片或填写文本 prompt", status=400)

    # 模型特定参数
    kwargs: Dict[str, Any] = {}

    if model == "hunyuan3d-2":
        hy3dgen_repo = "/disk2/licheng/code/ARIN5201-CV-FinalProject/Hunyuan3D_2"
        model_path = "/disk2/licheng/models/Hunyuan3D-2/"
        kwargs.update({
            "model_path": model_path,
            "do_rembg_if_rgb": True,
        })
        if hy3dgen_repo:
            kwargs["repo_dir"] = hy3dgen_repo

    elif model == "trellis":
        trellis_model = request.form.get("trellis_model", "microsoft/TRELLIS-image-large").strip()
        render_target = request.form.get("render_target", "mesh").strip()
        render_channel = request.form.get("render_channel", "normal").strip()
        spconv_algo = request.form.get("spconv_algo", "native").strip()
        texture_size = int(request.form.get("texture_size", "1024"))
        simplify_ratio = float(request.form.get("simplify_ratio", "0.95"))
        kwargs.update({
            "seed": 1,
            "trellis_model": trellis_model,
            "render_target": render_target,
            "render_channel": render_channel,
            "spconv_algo": spconv_algo,
            "texture_size": texture_size,
            "simplify_ratio": simplify_ratio,
            "do_rembg_if_rgb": True,
        })
    else:
        return Response("无效的模型选择", status=400)

    # 记录 Job
    log_queue = queue.Queue()
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "model": model,
            "gpu": gpu,
            "prompt": prompt,
            "image_path": image_path,           # 可能为 None（text 模式）
            "input_type": input_type,           # "image"/"text"
            "input_value": input_value,         # 路径或文本
            "output_dir": str(job_dir),
            "kwargs": kwargs,
            "status": "running",
            "progress": 0,
            "artifacts": {},
            "log_queue": log_queue,
            "proc": None,
        }

    # 启动后台线程运行推理
    th = threading.Thread(target=_run_inference_worker, args=(job_id,), daemon=True)
    th.start()

    return jsonify({"job_id": job_id})


def _run_inference_worker(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return

    model = job["model"]
    gpu = job["gpu"]
    output_dir = job["output_dir"]
    input_type = job["input_type"]
    input_value = job["input_value"]
    kwargs = dict(job["kwargs"])  # copy
    log_queue: queue.Queue = job["log_queue"]

    # Normalize input/output paths to absolute to avoid relative-path differences
    try:
        if input_value:
            input_value = str(Path(input_value).resolve())
            job["input_value"] = input_value
        output_dir = str(Path(output_dir).resolve())
        job["output_dir"] = output_dir
        # Normalize repo_dir inside kwargs if present
        repo_dir = kwargs.get("repo_dir")
        if repo_dir:
            kwargs["repo_dir"] = str(Path(repo_dir).resolve())
    except Exception:
        pass

    # 选择目标环境的 Python
    if model == "trellis":
        python_bin = PYTHON_TRELLIS
    else:
        python_bin = PYTHON_HY3D2

    # 构建命令。为了兼容你本地能跑通的用法：
    # - 对于 hunyuan3d-2，我们生成你期望的命令样式（不带 --worker，带 --hy3dgen-repo、--cuda-visible-devices）
    # - 其他模型继续使用原来的 --worker inline worker 风格
    worker_kwargs = dict(kwargs)
    # 提取 repo_dir（若存在），但不要把它放回 kwargs-json 当中——我们会单独传给 --hy3dgen-repo
    hy3d_repo = None
    if "repo_dir" in worker_kwargs:
        hy3d_repo = worker_kwargs.pop("repo_dir")

    payload = json.dumps(worker_kwargs, ensure_ascii=False)

    if model == "hunyuan3d-2":
        # Build command exactly like your expected invocation
        cmd_list = [
            python_bin, "-u", INFER_SCRIPT,
            "--model", model,
            "--input", input_value,
            "--input-type", input_type,
            "--output", output_dir,
        ]
        if hy3d_repo:
            cmd_list += ["--hy3dgen-repo", str(hy3d_repo)]
        # pass CUDA devices as CLI arg as you expect
        if gpu:
            cmd_list += ["--cuda-visible-devices", str(gpu)]
        cmd_list += ["--kwargs-json", payload]
    else:
        # default: spawn worker mode to run inline in target env
        cmd_list = [
            python_bin, "-u", INFER_SCRIPT,
            "--worker",
            "--model", model,
            "--input", input_value,
            "--input-type", input_type,
            "--output", output_dir,
            "--kwargs-json", payload,
            "--hunyuan-env", "hunyuan3d2",
            "--trellis-env", "trellis",
        ]

    # 子进程环境变量
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # 若为 Hunyuan3D-2 且提供了 repo_dir，则注入 PYTHONPATH
    repo_dir = kwargs.get("repo_dir")
    if model == "hunyuan3d-2" and repo_dir:
        env["PYTHONPATH"] = f"{repo_dir}:{env.get('PYTHONPATH', '')}"

    # 记录执行时的关键参数，便于排查（会出现在实时日志）
    log_queue.put({"type": "log", "line": f"Worker kwargs: {json.dumps(kwargs, ensure_ascii=False)}"})
    log_queue.put({"type": "log", "line": f"Worker env CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')} PYTHONPATH={env.get('PYTHONPATH','')}"})

    # 启动子进程
    log_queue.put({"type": "log", "line": f"Spawn: {' '.join(shlex_quote(x) for x in cmd_list)}"})
    log_queue.put({"type": "log", "line": f"Env CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')} PYTHONPATH={env.get('PYTHONPATH','')}"})

    proc = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    job["proc"] = proc

    stdout_lines = []
    try:
        while True:
            if proc.poll() is not None:
                break
            line = proc.stdout.readline()
            if line:
                stdout_lines.append(line)
                log_queue.put({"type": "log", "line": line.rstrip()})

                # 简单进度触发：基于关键日志标记推进（可按需精细化）
                if "Loading TRELLIS pipeline" in line or "Loading Hunyuan3D-2 pipelines" in line:
                    job["progress"] = 10
                    log_queue.put({"type": "progress", "value": 10})
                if ("Running TRELLIS inference" in line or
                    "Running Hunyuan3D-2 shape generation" in line):
                    job["progress"] = 50
                    log_queue.put({"type": "progress", "value": 50})
                if "Running Hunyuan3D-2 texture painting" in line:
                    job["progress"] = 70
                    log_queue.put({"type": "progress", "value": 70})
                if "Exporting GLB" in line:
                    job["progress"] = 85
                    log_queue.put({"type": "progress", "value": 85})
                if "Done." in line:
                    job["progress"] = 100
                    log_queue.put({"type": "progress", "value": 100})
            else:
                # 心跳
                time_sleep(0.05)

        ret = proc.wait()

        # 解析最后一条有效 JSON（worker 输出的结果）
        result = None
        for i in range(len(stdout_lines) - 1, -1, -1):
            s = stdout_lines[i].strip()
            try:
                result = json.loads(s)
                break
            except Exception:
                continue

        if ret == 0 and result:
            job["status"] = "ok"
            job["progress"] = 100
            job["artifacts"] = result.get("artifacts", {})

            # 保存 worker 输出 JSON 到作业目录，便于比对（meta_web.json）
            try:
                out_dir = Path(job.get("output_dir", ""))
                meta_out = out_dir / "meta_web.json"
                with open(meta_out, "w", encoding="utf-8") as mf:
                    json.dump(result, mf, ensure_ascii=False, indent=2)
                log_queue.put({"type": "log", "line": f"Saved worker meta: {meta_out}"})
            except Exception as e:
                log_queue.put({"type": "log", "line": f"Failed saving worker meta: {e}"})

            # 记录 artifacts 尺寸，便于比较（bytes）
            sizes = {}
            for k, v in job.get("artifacts", {}).items():
                try:
                    p = Path(v)
                    if p.exists():
                        sizes[k] = p.stat().st_size
                        log_queue.put({"type": "log", "line": f"Artifact {k}: {p} size={sizes[k]} bytes"})
                        # If GLB, analyze internals (buffers/images)
                        if str(p).lower().endswith('.glb'):
                            try:
                                analysis = analyze_glb(str(p))
                                log_queue.put({"type": "log", "line": f"GLB analysis {k}: {analysis}"})
                            except Exception as _e:
                                log_queue.put({"type": "log", "line": f"GLB analysis {k} failed: {_e}"})
                    else:
                        log_queue.put({"type": "log", "line": f"Artifact {k}: {v} (not found)"})
                except Exception as e:
                    log_queue.put({"type": "log", "line": f"Artifact {k}: error getting size: {e}"})
            job["artifact_sizes"] = sizes

            log_queue.put({"type": "status", "value": "ok"})
            log_queue.put({"type": "progress", "value": 100})
            log_queue.put({"type": "result", "artifacts": job["artifacts"]})
        else:
            job["status"] = "error"
            log_queue.put({"type": "status", "value": "error"})

    except Exception as e:
        job["status"] = "error"
        log_queue.put({"type": "log", "line": f"Worker exception: {e}"})
        log_queue.put({"type": "status", "value": "error"})
        try:
            proc.kill()
        except Exception:
            pass


# ---- SSE 实时日志 ----
@app.get("/logs/<job_id>")
def sse_logs(job_id: str):
    def event_stream():
        last_heartbeat = time_now()
        while True:
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                yield sse_format({"type": "status", "value": "error"})
                break
            q: queue.Queue = job["log_queue"]
            try:
                item = q.get(timeout=1.0)
                yield sse_format(item)
            except queue.Empty:
                # 心跳保持连接
                if time_now() - last_heartbeat > SSE_HEARTBEAT_SEC:
                    yield sse_format({"type": "heartbeat"})
                    last_heartbeat = time_now()

            # 结束条件：作业完成且队列清空后，退出
            if job["status"] in ("ok", "error") and q.empty():
                break

    return Response(event_stream(), mimetype="text/event-stream")


# ---- 状态查询 ----
@app.get("/status/<job_id>")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "job not found"}), 404
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "artifacts": job["artifacts"],
    })


# ---- 文件下载 ----
@app.get("/download/<job_id>/<kind>")
def download(job_id: str, kind: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return abort(404)
    artifacts = job.get("artifacts", {})
    p = artifacts.get(kind)
    if not p or not Path(p).exists():
        return abort(404)
    # 统一下载文件名
    filename = f"{job_id}_{kind}{Path(p).suffix}"
    return send_file(p, as_attachment=True, download_name=filename)


# ---- 文件可视化 ----
@app.get("/view/<job_id>")
def view_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return abort(404)
        artifacts = job.get("artifacts", {})

    kind = request.args.get("kind")
    path = artifacts.get(kind) if kind else None

    # 自动寻找 .glb
    if not path:
        for k, v in artifacts.items():
            if str(v).lower().endswith(".glb") and Path(v).exists():
                kind, path = k, v
                break

    # 如果 artifacts 中未找到，尝试在作业输出目录里查找 .glb 文件
    if not path:
        out_dir = Path(job.get("output_dir", ""))
        if out_dir.exists():
            for p in out_dir.rglob('*.glb'):
                path = str(p)
                kind = Path(p).name
                break

    waiting = not path
    # 为 model-viewer 提供非 attachment 的内嵌访问 URL
    glb_url = url_for("view_glb", job_id=job_id, kind=kind) if path else None
    return render_template("viewer.html", job=job, kind=kind, glb_url=glb_url, waiting=waiting)


# 无 job id 的预览页：允许上传 GLB
@app.get("/view")
def view_noid():
    # 空的 viewer，会显示上传按钮
    return render_template("viewer.html", job=None, kind=None, glb_url=None, waiting=True)


@app.post('/view/upload')
def view_upload():
    f = request.files.get('glb')
    if not f or not f.filename:
        return Response('No file', status=400)
    fname = f.filename
    if not fname.lower().endswith('.glb'):
        return Response('Only .glb files allowed', status=400)
    token = uuid.uuid4().hex
    dest = UPLOADED_DIR / f"{token}.glb"
    f.save(str(dest))
    with uploaded_files_lock:
        uploaded_files[token] = str(dest)
    return redirect(url_for('view_uploaded', token=token))


@app.get('/view_uploaded/<token>')
def view_uploaded(token: str):
    with uploaded_files_lock:
        path = uploaded_files.get(token)
    if not path or not Path(path).exists():
        return abort(404)
    glb_url = url_for('uploaded_file', token=token)
    return render_template('viewer.html', job=None, kind='uploaded', glb_url=glb_url, waiting=False)


@app.get('/uploaded/<token>')
def uploaded_file(token: str):
    with uploaded_files_lock:
        path = uploaded_files.get(token)
    if not path or not Path(path).exists():
        return abort(404)
    return send_file(path, as_attachment=False)


@app.get('/view_glb/<job_id>/<kind>')
def view_glb(job_id: str, kind: str):
    # Serve the GLB for inline viewing (no Content-Disposition: attachment)
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return abort(404)
    artifacts = job.get('artifacts', {})
    p = artifacts.get(kind)
    if not p or not Path(p).exists():
        # try searching output dir
        out_dir = Path(job.get('output_dir', ''))
        if out_dir.exists():
            found = None
            for pp in out_dir.rglob('*'):
                if pp.is_file() and pp.name == kind:
                    found = pp
                    break
            if found:
                p = str(found)
    if not p or not Path(p).exists():
        return abort(404)
    return send_file(p, as_attachment=False)


# ---- 工具函数 ----
def shlex_quote(x: str) -> str:
    import shlex
    return shlex.quote(x)


def time_sleep(sec: float):
    import time
    time.sleep(sec)


def time_now() -> float:
    import time
    return time.time()


def analyze_glb(path: str) -> Dict[str, Any]:
    """Lightweight GLB inspector: returns buffer and image counts/sizes.

    This does not require external libs: it reads GLB header, parses JSON chunk,
    and inspects `buffers` and `images` entries. For embedded data URIs it
    decodes base64 to get byte sizes.
    """
    import struct
    import json as _json
    import base64

    info: Dict[str, Any] = {"buffers": [], "images": [], "meshes": 0}
    try:
        with open(path, 'rb') as f:
            data = f.read()
        # GLB header: 12 bytes
        if len(data) < 12:
            return {"error": "file too small"}
        magic, version, length = struct.unpack_from('<4sII', data, 0)
        if magic != b'glTF':
            return {"error": "not glb"}
        offset = 12
        # Read chunks
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

        # buffers
        for b in json_chunk.get('buffers', []):
            blen = b.get('byteLength')
            info['buffers'].append({'byteLength': blen})

        # images
        for im in json_chunk.get('images', []):
            uri = im.get('uri')
            if not uri:
                # likely refers to buffer view; we cannot get exact size easily
                info['images'].append({'uri': None, 'note': 'bufferView'})
            elif uri.startswith('data:'):
                try:
                    header, b64 = uri.split(',', 1)
                    raw = base64.b64decode(b64)
                    info['images'].append({'uri': 'data', 'size': len(raw)})
                except Exception:
                    info['images'].append({'uri': 'data', 'size': None})
            else:
                # external file reference (rare inside GLB)
                info['images'].append({'uri': uri, 'size': None})

        info['meshes'] = len(json_chunk.get('meshes', []))
        return info
    except Exception as e:
        return {"error": str(e)}


# ---- 启动 ----
if __name__ == "__main__":
    # 生产部署可用 gunicorn 等，这里简单跑开发服务
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
