from __future__ import annotations
import numpy as np
from PIL import Image
import cv2


def _trimap_as_uint8(trimap: Image.Image) -> np.ndarray:
    t = np.array(trimap, dtype=np.uint8)
    vals = set(np.unique(t).tolist())
    # If trimap is {0,1,2} (bg/fg/border), remap to {1=fg,2=border,3=bg}.
    if vals.issubset({0, 1, 2}):
        t2 = np.zeros_like(t, dtype=np.uint8)
        t2[t == 1] = 1
        t2[t == 2] = 2
        t2[t == 0] = 3
        return t2
    return t


def mask_background_from_trimap(trimap: Image.Image, dilate: int = 7) -> Image.Image:
    t = _trimap_as_uint8(trimap)
    m = np.zeros_like(t, dtype=np.uint8)
    m[(t == 2) | (t == 3)] = 255  # mask border+background
    if dilate and dilate > 0:
        k = np.ones((int(dilate), int(dilate)), np.uint8)
        m = cv2.dilate(m, k, iterations=1)
    return Image.fromarray(m.astype(np.uint8)).convert("L")


def trimap_to_fg_alpha(trimap: Image.Image, size: int, feather: int = 11) -> np.ndarray:
    t = _trimap_as_uint8(trimap)
    alpha = np.zeros_like(t, dtype=np.float32)
    alpha[t == 1] = 1.0
    alpha[t == 3] = 0.5
    alpha = cv2.resize(alpha, (int(size), int(size)), interpolation=cv2.INTER_NEAREST)
    if feather and feather > 0:
        k = int(feather)
        k = k if k % 2 == 1 else k + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return np.clip(alpha, 0.0, 1.0)


def composite_fg_bg(orig: Image.Image, gen: Image.Image, alpha_fg: np.ndarray) -> Image.Image:
    o = np.array(orig.convert("RGB"), dtype=np.float32)
    g = np.array(gen.convert("RGB"), dtype=np.float32)
    a = alpha_fg[..., None].astype(np.float32)
    out = a * o + (1.0 - a) * g
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
