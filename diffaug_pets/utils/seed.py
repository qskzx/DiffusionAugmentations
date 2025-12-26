import gc, random
from contextlib import nullcontext
import numpy as np
import torch

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_tf32_if_cuda(device: str):
    if device != "cuda":
        return
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    except Exception:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def cleanup(device: str):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def autocast_ctx(device: str):
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()

def make_cpu_generator(seed: int) -> torch.Generator:
    return torch.Generator("cpu").manual_seed(int(seed))
