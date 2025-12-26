from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from PIL import Image

from ..data.pets import build_mat_indices_by_class


def save_img(pil: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def load_or_make(man_path: Path, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    man_path.parent.mkdir(parents=True, exist_ok=True)
    if man_path.exists():
        return pd.read_csv(man_path)
    df = fn()
    df.to_csv(man_path, index=False)
    return df


def generate_manifest_inpaint_bg(
    augmenter,
    mat_ds,
    class_names: List[str],
    target_classes: List[int],
    syn_root: Path,
    base_seed: int,
    n_seed_images_per_class: int,
    n_variations_per_seed: int,
) -> pd.DataFrame:
    idxs_by_class = build_mat_indices_by_class(mat_ds, len(class_names))
    rows = []
    rng = random.Random(int(base_seed))
    for c in target_classes:
        avail = idxs_by_class.get(int(c), [])
        if not avail:
            continue
        take = min(int(n_seed_images_per_class), len(avail))
        seed_idxs = rng.sample(avail, take)
        for si, midx in enumerate(seed_idxs):
            img, trimap, y, uid = mat_ds[int(midx)]
            y = int(y); uid = int(uid)
            for k in range(int(n_variations_per_seed)):
                seed = int(base_seed + 100000 * y + 1000 * uid + 10 * k)
                out = augmenter.inpaint_background(img, trimap, y, seed=seed)
                p = syn_root / "inpaint_bg" / class_names[y] / f"uid{uid}_s{si}_k{k}.png"
                save_img(out, p)
                rows.append({"path": str(p), "y": y, "class_name": class_names[y], "method": "inpaint_bg", "uid": uid, "seed": seed})
    return pd.DataFrame(rows)


def generate_manifest_small_edit(
    augmenter,
    mat_ds,
    class_names: List[str],
    target_classes: List[int],
    syn_root: Path,
    base_seed: int,
    n_seed_images_per_class: int,
    n_variations_per_seed: int,
) -> pd.DataFrame:
    idxs_by_class = build_mat_indices_by_class(mat_ds, len(class_names))
    rows = []
    rng = random.Random(int(base_seed))
    for c in target_classes:
        avail = idxs_by_class.get(int(c), [])
        if not avail:
            continue
        take = min(int(n_seed_images_per_class), len(avail))
        seed_idxs = rng.sample(avail, take)
        for si, midx in enumerate(seed_idxs):
            img, _trimap, y, uid = mat_ds[int(midx)]
            y = int(y); uid = int(uid)
            for k in range(int(n_variations_per_seed)):
                seed = int(base_seed + 100000 * y + 1000 * uid + 10 * k + 123)
                out = augmenter.small_edit(img, y, seed=seed)
                p = syn_root / "small_edit" / class_names[y] / f"uid{uid}_s{si}_k{k}.png"
                save_img(out, p)
                rows.append({"path": str(p), "y": y, "class_name": class_names[y], "method": "small_edit", "uid": uid, "seed": seed})
    return pd.DataFrame(rows)


def generate_manifest_text2img(
    augmenter,
    class_names: List[str],
    target_classes: List[int],
    syn_root: Path,
    base_seed: int,
    n_text2img_per_class: int,
) -> pd.DataFrame:
    rows = []
    for c in target_classes:
        y = int(c)
        for k in range(int(n_text2img_per_class)):
            seed = int(base_seed + 100000 * y + 7 * k + 999)
            out = augmenter.text2img(y, seed=seed)
            p = syn_root / "text2img" / class_names[y] / f"t2i_k{k}.png"
            save_img(out, p)
            rows.append({"path": str(p), "y": y, "class_name": class_names[y], "method": "text2img", "uid": -1, "seed": seed})
    return pd.DataFrame(rows)
