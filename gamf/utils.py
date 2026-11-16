import contextlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(name)


def autocast(device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


# ------------------------------------------------------------------ crash safety
# All result files are written atomically (temporary file, fsync, os.replace), so a
# power failure can never leave a truncated checkpoint, JSON file or image behind.

def _fsync_replace(tmp, path):
    os.replace(tmp, path)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    _fsync_replace(tmp, path)


def atomic_torch_save(obj, path):
    """Save a checkpoint atomically and keep the previous one as <name>.prev."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        os.replace(path, path.with_name(path.name + ".prev"))
    _fsync_replace(tmp, path)


def load_checkpoint(path, map_location="cpu"):
    """Load <name>, falling back to <name>.prev if the newest file is missing or unreadable."""
    path = Path(path)
    for cand in (path, path.with_name(path.name + ".prev")):
        if cand.exists():
            try:
                return torch.load(cand, map_location=map_location, weights_only=False)
            except Exception as err:  # truncated or corrupted file
                print(f"warning: {cand.name} unreadable ({err}); trying the backup")
    return None


def save_image_atomic(img, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    img.save(tmp, format="PNG")
    _fsync_replace(tmp, path)


def rng_state():
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    # generator states must be CPU ByteTensors; checkpoints loaded with map_location=<cuda device>
    # move them to the GPU, so bring them back first
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu())
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


def mark_done(path, **info):
    """Completion marker: a step counts as finished only when its marker exists."""
    write_json({"done": True, **info}, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hardware_info():
    """Exact software/hardware description for the manuscript (reviewer R3-6)."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cpu": platform.processor() or platform.machine(),
        "os": platform.platform(),
        # number of jobs the orchestrator runs concurrently (one per GPU); they share the CPU
        "parallel_jobs": int(os.environ.get("GAMF_PARALLEL_JOBS", "1")),
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info["gpu"] = prop.name
        info["gpu_memory_gb"] = round(prop.total_memory / 1024 ** 3, 1)
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        pass
    return info


def count_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters()
               if p.requires_grad or not trainable_only)
