from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from PIL import Image
import torch

from ..utils.seed import autocast_ctx, make_cpu_generator
from ..diffusion.pipes import Pipes
from .prompts import BG_LIST, class_prompt, neg_with_confusables, safe_inpaint_prompt
from .matting import mask_background_from_trimap, trimap_to_fg_alpha, composite_fg_bg


def to_sd_size(pil: Image.Image, size: int) -> Image.Image:
    size = int(size); size = size - (size % 8)
    return pil.convert("RGB").resize((size, size), Image.BICUBIC)

def to_sd_mask(mask: Image.Image, size: int) -> Image.Image:
    size = int(size); size = size - (size % 8)
    return mask.convert("L").resize((size, size), Image.NEAREST)

def is_black_image(pil_img: Image.Image, mean_thr=2.0, std_thr=2.0) -> bool:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
    return (arr.mean() < float(mean_thr)) and (arr.std() < float(std_thr))


@torch.no_grad()
def inpaint_retry(pipes: Pipes, device: str, img_in: Image.Image, mask_in: Image.Image,
                  prompt: str, negative: str, steps: int, guidance: float,
                  strength: float, seed0: int, max_tries: int = 6) -> Image.Image:
    last = None
    for k in range(int(max_tries)):
        gen = make_cpu_generator(int(seed0) + k)
        with autocast_ctx(device):
            out = pipes.inpaint(
                prompt=prompt,
                negative_prompt=negative,
                image=img_in,
                mask_image=mask_in,
                strength=float(strength),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=gen
            ).images[0]
        last = out
        if not is_black_image(out):
            return out
    return last


@torch.no_grad()
def img2img_retry(pipes: Pipes, device: str, img_in: Image.Image,
                  prompt: str, negative: str, steps: int, guidance: float,
                  strength: float, seed0: int, max_tries: int = 4) -> Image.Image:
    last = None
    for k in range(int(max_tries)):
        gen = make_cpu_generator(int(seed0) + k)
        with autocast_ctx(device):
            out = pipes.i2i(
                prompt=prompt,
                negative_prompt=negative,
                image=img_in,
                strength=float(strength),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=gen
            ).images[0]
        last = out
        if not is_black_image(out):
            return out
    return last


@torch.no_grad()
def t2i_retry(pipes: Pipes, device: str, prompt: str, negative: str,
              steps: int, guidance: float, size: int,
              seed0: int, max_tries: int = 6) -> Image.Image:
    last = None
    for k in range(int(max_tries)):
        gen = make_cpu_generator(int(seed0) + k)
        with autocast_ctx(device):
            out = pipes.t2i(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=gen,
                height=int(size),
                width=int(size)
            ).images[0]
        last = out
        if not is_black_image(out):
            return out
    return last


@dataclass
class Augmenter:
    cfg: any
    device: str
    class_names: List[str]
    conf_map: Dict[int, List[int]]
    species_token_fn: Callable[[str], str]
    pipes: Pipes

    @torch.no_grad()
    def inpaint_background(self, img: Image.Image, trimap: Image.Image, class_idx: int, seed: int) -> Image.Image:
        size = int(self.cfg.gen_size)
        img2 = to_sd_size(img, size)
        mask = to_sd_mask(mask_background_from_trimap(trimap, dilate=7), size)

        bg = BG_LIST[int(seed) % len(BG_LIST)]
        prompt = safe_inpaint_prompt(self.class_names[int(class_idx)], self.species_token_fn, bg)
        neg = neg_with_confusables(int(class_idx), self.class_names, self.conf_map)

        out = inpaint_retry(
            pipes=self.pipes, device=self.device,
            img_in=img2, mask_in=mask,
            prompt=prompt, negative=neg,
            steps=max(int(self.cfg.steps_inpaint), 30),
            guidance=6.5,
            strength=float(self.cfg.strength_bg_inpaint),
            seed0=int(seed),
        )

        out_size = out.size[0]
        img2s = img2.resize((out_size, out_size), Image.BICUBIC)
        alpha = trimap_to_fg_alpha(trimap, size=out_size, feather=11)
        comp = composite_fg_bg(img2s, out, alpha_fg=alpha)
        return comp.resize((size, size), Image.BICUBIC)

    @torch.no_grad()
    def small_edit(self, img: Image.Image, class_idx: int, seed: int) -> Image.Image:
        size = int(self.cfg.gen_size)
        img2 = to_sd_size(img, size)
        prompt = class_prompt(self.class_names[int(class_idx)], self.species_token_fn) + ", same animal, small pose change"
        neg = neg_with_confusables(int(class_idx), self.class_names, self.conf_map)
        out = img2img_retry(
            pipes=self.pipes, device=self.device,
            img_in=img2, prompt=prompt, negative=neg,
            steps=max(int(self.cfg.steps_img2img), 16),
            guidance=float(self.cfg.guidance),
            strength=float(self.cfg.strength_small_edits),
            seed0=int(seed),
        )
        return out.resize((size, size), Image.BICUBIC)

    @torch.no_grad()
    def text2img(self, class_idx: int, seed: int) -> Image.Image:
        size = int(self.cfg.gen_size)
        prompt = class_prompt(self.class_names[int(class_idx)], self.species_token_fn)
        neg = neg_with_confusables(int(class_idx), self.class_names, self.conf_map)
        out = t2i_retry(
            pipes=self.pipes, device=self.device,
            prompt=prompt, negative=neg,
            steps=max(int(self.cfg.steps_t2i), 28),
            guidance=float(self.cfg.guidance),
            size=size,
            seed0=int(seed),
        )
        return out.resize((size, size), Image.BICUBIC)
