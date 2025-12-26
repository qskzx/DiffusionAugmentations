from __future__ import annotations
from dataclasses import dataclass
import torch
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    DPMSolverMultistepScheduler,
)

@dataclass
class Pipes:
    t2i: any
    i2i: any
    inpaint: any


def tune_pipe(pipe, device: str):
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    pipe.enable_attention_slicing("max")
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    return pipe


@torch.no_grad()
def load_pipes(t2i_model: str, i2i_model: str, inpaint_model: str, device: str) -> Pipes:
    dtype = torch.float16 if device == "cuda" else torch.float32
    t2i = AutoPipelineForText2Image.from_pretrained(
        t2i_model, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False,
        use_safetensors=True,
        variant=("fp16" if device == "cuda" else None),
    )
    i2i = AutoPipelineForImage2Image.from_pretrained(
        i2i_model, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False,
        use_safetensors=True,
        variant=("fp16" if device == "cuda" else None),
    )
    inp = AutoPipelineForInpainting.from_pretrained(
        inpaint_model, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False,
        use_safetensors=False,
        variant=("fp16" if device == "cuda" else None),
    )
    return Pipes(
        t2i=tune_pipe(t2i, device),
        i2i=tune_pipe(i2i, device),
        inpaint=tune_pipe(inp, device),
    )
