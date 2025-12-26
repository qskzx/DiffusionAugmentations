from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import imagehash
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel

from ..utils.seed import autocast_ctx


def is_bad_image(pil: Image.Image, min_std: float) -> bool:
    arr = np.array(pil.convert("RGB"), dtype=np.float32)
    if arr.mean() < 5:
        return True
    if arr.std() < float(min_std):
        return True
    return False


@torch.no_grad()
def load_clip(model_name: str, device: str):
    tok = CLIPTokenizer.from_pretrained(model_name, use_fast=False)
    imgp = CLIPImageProcessor.from_pretrained(model_name, use_fast=False)
    model = CLIPModel.from_pretrained(
        model_name,
        torch_dtype=(torch.float16 if device == "cuda" else torch.float32),
        use_safetensors=True,
    ).to(device)
    model.eval()
    return tok, imgp, model


def phash_str(pil: Image.Image) -> str:
    return str(imagehash.phash(pil))


def clip_text_for_class(class_idx: int, class_names: List[str], species_token_fn) -> str:
    cls = class_names[int(class_idx)].replace("_", " ")
    sp = species_token_fn(class_names[int(class_idx)])
    return f"a photo of a {cls} {sp}"


@torch.no_grad()
def compute_clip_margin_scores(
    df: pd.DataFrame,
    device: str,
    clip_tok,
    clip_imgp,
    clip_model,
    class_names: List[str],
    conf_map: Dict[int, List[int]],
    species_token_fn,
    batch_size: int = 16,
) -> pd.DataFrame:
    if df.empty:
        return df

    needed_cls = set(df["y"].tolist())
    needed_conf = set()
    for y in needed_cls:
        for j in conf_map.get(int(y), []):
            needed_conf.add(int(j))

    all_text_classes = sorted(list(needed_cls | needed_conf))
    text_list = [clip_text_for_class(i, class_names, species_token_fn) for i in all_text_classes]

    text_inputs = clip_tok(text_list, padding=True, truncation=True, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with autocast_ctx(device):
        te = clip_model.get_text_features(**text_inputs)
    te = te / te.norm(dim=-1, keepdim=True)

    cls2pos = {c: i for i, c in enumerate(all_text_classes)}

    clip_own: List[float] = []
    clip_compmax: List[float] = []
    paths = df["path"].tolist()
    ys = df["y"].tolist()

    for i in range(0, len(paths), int(batch_size)):
        p_batch = paths[i : i + int(batch_size)]
        y_batch = ys[i : i + int(batch_size)]
        imgs = [Image.open(p).convert("RGB") for p in p_batch]

        img_inputs = clip_imgp(images=imgs, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}

        with autocast_ctx(device):
            ie = clip_model.get_image_features(**img_inputs)
        ie = ie / ie.norm(dim=-1, keepdim=True)

        own_pos = torch.tensor([cls2pos[int(y)] for y in y_batch], device=device)
        te_own = te[own_pos]
        s_own = (ie * te_own).sum(dim=-1)

        s_cmp = []
        for b, y in enumerate(y_batch):
            confs = conf_map.get(int(y), [])
            if not confs:
                s_cmp.append(torch.tensor(0.0, device=device))
                continue
            conf_pos = torch.tensor([cls2pos[int(j)] for j in confs if int(j) in cls2pos], device=device)
            te_conf = te[conf_pos]
            sims = (te_conf @ ie[b].unsqueeze(-1)).squeeze(-1)
            s_cmp.append(sims.max())
        s_cmp = torch.stack(s_cmp, dim=0)

        clip_own.extend(s_own.detach().float().cpu().numpy().tolist())
        clip_compmax.extend(s_cmp.detach().float().cpu().numpy().tolist())

    out = df.copy()
    out["clip_own"] = clip_own
    out["clip_compmax"] = clip_compmax
    out["clip_margin"] = out["clip_own"] - out["clip_compmax"]
    return out


def dedup_phash_group(g: pd.DataFrame, phash_hamming_thr: int) -> pd.DataFrame:
    g = g.sort_values(["clip_margin", "clip_own"], ascending=False).reset_index(drop=True)
    kept = []
    seen = []
    for _, row in g.iterrows():
        h = imagehash.hex_to_hash(row["phash"])
        ok = True
        for hs in seen:
            if h - hs <= int(phash_hamming_thr):
                ok = False
                break
        if ok:
            kept.append(row)
            seen.append(h)
    return pd.DataFrame(kept)


@dataclass
class FilterCfg:
    min_std: float
    phash_hamming_thr: int
    clip_margin_thr: float
    clip_own_min: float
    keep_per_class: int


def filter_manifest(
    df: pd.DataFrame,
    device: str,
    class_names: List[str],
    conf_map: Dict[int, List[int]],
    species_token_fn,
    clip_model_name: str,
    fcfg: FilterCfg,
    batch_size: int = 16,
) -> pd.DataFrame:
    rows = []
    for r in df.to_dict("records"):
        p = r["path"]
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB")
        if is_bad_image(im, min_std=float(fcfg.min_std)):
            continue
        r["phash"] = phash_str(im)
        rows.append(r)

    df2 = pd.DataFrame(rows)
    if df2.empty:
        return df2

    clip_tok, clip_imgp, clip_model = load_clip(clip_model_name, device)

    df2 = compute_clip_margin_scores(
        df2, device=device,
        clip_tok=clip_tok, clip_imgp=clip_imgp, clip_model=clip_model,
        class_names=class_names, conf_map=conf_map, species_token_fn=species_token_fn,
        batch_size=batch_size,
    )

    # Free a bit of VRAM/ram
    del clip_model, clip_tok, clip_imgp
    if device == "cuda":
        torch.cuda.empty_cache()

    df2 = df2[(df2["clip_margin"] >= float(fcfg.clip_margin_thr)) & (df2["clip_own"] >= float(fcfg.clip_own_min))].copy()
    if df2.empty:
        return df2

    kept_groups = []
    for (y, method), g in df2.groupby(["y", "method"]):
        kept_groups.append(dedup_phash_group(g, phash_hamming_thr=int(fcfg.phash_hamming_thr)))
    df3 = pd.concat(kept_groups, ignore_index=True) if kept_groups else df2

    out = []
    for y, g in df3.groupby("y"):
        gg = g.sort_values(["clip_margin", "clip_own"], ascending=False).head(int(fcfg.keep_per_class)).copy()
        out.append(gg)

    return pd.concat(out, ignore_index=True) if out else df3
