"""
Lightweight visualization helpers for the lecture Colab.

These functions are intentionally "dependency-light": you pass in your augmentation
callbacks and the dataset. This keeps the Colab notebook clean while still allowing
flexible selection by:
  - uid (int)
  - class name (str) -> first sample of that class
  - None -> random sample (optionally only among target_classes)

Expected dataset item format (mat_ds):
    img, trimap, y, uid = mat_ds[i]
where img/trimap are PIL.Images, y is int class index, uid is unique int id.
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Union, Callable, Any

import random
import numpy as np
import matplotlib.pyplot as plt


Selector = Union[int, str, None]


def show_grid(rows, row_titles, col_titles):
    """Simple matplotlib grid: rows[r][c] should be a PIL.Image or numpy array."""
    R = len(rows)
    C = len(rows[0]) if R else 0
    fig, axes = plt.subplots(R, C, figsize=(C * 2.6, R * 2.6))
    if R == 1:
        axes = np.array([axes])
    for r in range(R):
        for c in range(C):
            ax = axes[r, c]
            ax.imshow(rows[r][c])
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
        axes[r, 0].text(
            -0.02, 0.5, row_titles[r],
            transform=axes[r, 0].transAxes,
            fontsize=10, rotation=90, va="center", ha="right",
        )
    plt.tight_layout()
    plt.show()


def _normalize_cls(s: str) -> str:
    return s.strip().replace("_", " ").casefold()


def _build_uid_to_index(mat_ds) -> dict:
    uid2i = {}
    for i in range(len(mat_ds)):
        _img, _tri, _y, uid = mat_ds[i]
        uid2i[int(uid)] = int(i)
    return uid2i


def _build_classname_to_idx(class_names: Sequence[str]) -> dict:
    m = {}
    for i, n in enumerate(class_names):
        m[_normalize_cls(n)] = int(i)
    return m


def _find_first_index_by_class(mat_ds, class_idx: int) -> Optional[int]:
    for i in range(len(mat_ds)):
        _img, _tri, y, _uid = mat_ds[i]
        if int(y) == int(class_idx):
            return int(i)
    return None


def _candidate_indices(mat_ds, target_classes: Optional[Sequence[int]], only_targets: bool) -> List[int]:
    if only_targets and target_classes is not None and len(target_classes) > 0:
        tset = set(map(int, target_classes))
        cand = []
        for i in range(len(mat_ds)):
            _img, _tri, y, _uid = mat_ds[i]
            if int(y) in tset:
                cand.append(int(i))
        if cand:
            return cand
    return list(range(len(mat_ds)))


def visualize_selectors_or_random(
    *,
    mat_ds,
    class_names: Sequence[str],
    cfg,
    # augmentation callbacks
    aug_inpaint_background: Callable[[Any, Any, int, int], Any],
    aug_small_edit: Callable[[Any, int, int], Any],
    aug_text2img: Callable[[int, int], Any],
    to_sd_size: Callable[[Any, int], Any],
    # selection / layout
    selectors: Union[None, Selector, Sequence[Selector]] = None,
    n_rows: int = 3,
    n_per_type: int = 2,
    base_seed: int = 42,
    target_classes: Optional[Sequence[int]] = None,
    only_targets_random: bool = True,
    class_pick: str = "first",  # "first" | "random" (random not implemented here, but reserved)
):
    """
    selectors:
      - None -> pick n_rows random examples
      - int  -> uid
      - str  -> class name (берём первый пример этого класса)
      - list -> каждый элемент: uid | class_name | None
    """
    rng = random.Random(int(base_seed))
    uid2i = _build_uid_to_index(mat_ds)
    name2idx = _build_classname_to_idx(class_names)
    cand = _candidate_indices(mat_ds, target_classes=target_classes, only_targets=only_targets_random)

    # normalize selectors input -> slots
    if selectors is None:
        slots = [None] * int(n_rows)
    elif isinstance(selectors, (int, np.integer, str)):
        slots = [selectors]
    else:
        slots = list(selectors)
        if len(slots) < int(n_rows):
            slots = slots + [None] * (int(n_rows) - len(slots))
        else:
            slots = slots[: int(n_rows)]

    used_idxs = set()

    def pick_random_idx():
        for _ in range(200):
            i = rng.choice(cand)
            if i not in used_idxs:
                return i
        return rng.choice(cand)

    def pick_by_class(class_idx: int) -> Optional[int]:
        # first (as requested)
        i = _find_first_index_by_class(mat_ds, class_idx)
        return i

    chosen_idxs = []
    for s in slots:
        if s is None:
            i = pick_random_idx()
            chosen_idxs.append(i); used_idxs.add(i)
            continue

        if isinstance(s, (int, np.integer)):
            uid = int(s)
            if uid not in uid2i:
                print(f"[warn] uid={uid} not found -> using random")
                i = pick_random_idx()
            else:
                i = uid2i[uid]
                if i in used_idxs:
                    i = pick_random_idx()
            chosen_idxs.append(i); used_idxs.add(i)
            continue

        if isinstance(s, str):
            key = _normalize_cls(s)
            if key not in name2idx:
                print(f"[warn] class '{s}' not found -> using random")
                i = pick_random_idx()
            else:
                cidx = name2idx[key]
                i = pick_by_class(cidx)
                if i is None:
                    print(f"[warn] no samples for class '{s}' in mat_ds -> using random")
                    i = pick_random_idx()
                elif i in used_idxs:
                    i = pick_random_idx()
            chosen_idxs.append(i); used_idxs.add(i)
            continue

        print(f"[warn] unsupported selector type={type(s)} -> using random")
        i = pick_random_idx()
        chosen_idxs.append(i); used_idxs.add(i)

    # build grid
    size = int(getattr(cfg, "gen_size", 384))
    col_titles = ["orig"]
    col_titles += [f"inpaint_bg {k+1}" for k in range(int(n_per_type))]
    col_titles += [f"small_edit {k+1}" for k in range(int(n_per_type))]
    col_titles += [f"text2img {k+1}" for k in range(int(n_per_type))]

    rows = []
    row_titles = []
    for r, idx in enumerate(chosen_idxs):
        img, tri, y, uid = mat_ds[int(idx)]
        y = int(y); uid = int(uid)

        row = [to_sd_size(img, size)]
        for k in range(int(n_per_type)):
            row.append(aug_inpaint_background(img, tri, y, seed=int(base_seed + 10 * r + k)))
        for k in range(int(n_per_type)):
            row.append(aug_small_edit(img, y, seed=int(base_seed + 100 + 10 * r + k)))
        for k in range(int(n_per_type)):
            row.append(aug_text2img(y, seed=int(base_seed + 200 + 10 * r + k)))

        rows.append(row)
        row_titles.append(f"{class_names[y]} (uid={uid})")

    show_grid(rows, row_titles, col_titles)
