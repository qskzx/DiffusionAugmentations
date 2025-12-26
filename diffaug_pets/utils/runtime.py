from __future__ import annotations

import gc
import warnings
from typing import Optional

import numpy as np
import torch


def setup_warnings_quiet() -> None:
    """Reduce warning/log spam from HF/diffusers in Colab."""
    warnings.filterwarnings("ignore", message=r".*Flax classes are deprecated.*")
    warnings.filterwarnings("ignore", message=r".*enable_vae_slicing.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*enable_vae_tiling.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*HF_TOKEN.*does not exist.*")
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        from diffusers.utils import logging as diff_log
        diff_log.set_verbosity_error()
    except Exception:
        pass
    try:
        from transformers.utils import logging as tr_log
        tr_log.set_verbosity_error()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import logging as hub_log
        hub_log.set_verbosity_error()
    except Exception:
        pass


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_tf32_if_cuda(device: Optional[str] = None) -> None:
    """Enable TF32 where supported (safe speed-up on Ampere+ GPUs)."""
    device = device or get_device()
    if device != "cuda":
        return
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"  # new API
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


def seed_everything(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cleanup(device: Optional[str] = None) -> None:
    """Free Python + CUDA memory (useful between diffusion + training blocks)."""
    device = device or get_device()
    gc.collect()
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def autocast_ctx(device: Optional[str] = None):
    """Autocast context: fp16 on CUDA, no-op on CPU."""
    device = device or get_device()
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    from contextlib import nullcontext
    return nullcontext()


def setup_runtime(seed: int = 42, quiet: bool = True) -> str:
    """Convenience: quiet logs + TF32 + seeding. Returns device string."""
    if quiet:
        setup_warnings_quiet()
    device = get_device()
    set_tf32_if_cuda(device)
    seed_everything(seed)
    return device


def print_cuda_mem(prefix: str = "") -> None:
    """Utility for Colab/debug: print current CUDA memory usage (allocated/reserved)."""
    import torch
    if not torch.cuda.is_available():
        print(f"{prefix}CUDA not available")
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"{prefix}allocated: {alloc:.3f} GB | reserved: {reserved:.3f} GB")
