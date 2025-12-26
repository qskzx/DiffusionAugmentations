from __future__ import annotations
import random
from typing import Any, List, Optional, Sequence, Union, Dict

import numpy as np
import matplotlib.pyplot as plt

from ..aug.augmenter import to_sd_size


def show_grid(rows, row_titles, col_titles):
    R = len(rows); C = len(rows[0])
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
        axes[r, 0].text(-0.02, 0.5, row_titles[r], transform=axes[r, 0].transAxes,
                        fontsize=10, rotation=90, va="center", ha="right")
    plt.tight_layout()
    plt.show()


def _build_uid_to_index(mat_ds):
    uid2i = {}
    for i in range(len(mat_ds)):
        _img, _tri, _y, uid = mat_ds[i]
        uid2i[int(uid)] = int(i)
    return uid2i


def _normalize_cls(s: str) -> str:
    return s.strip().replace("_", " ").casefold()


def _build_classname_to_idx(class_names: List[str]):
    return {_normalize_cls(n): int(i) for i, n in enumerate(class_names)}


def _find_first_index_by_class(mat_ds, class_idx: int):
    for i in range(len(mat_ds)):
        _img, _tri, y, _uid = mat_ds[i]
        if int(y) == int(class_idx):
            return int(i)
    return None


def _candidate_indices(mat_ds, target_classes: Optional[Sequence[int]] = None):
    if target_classes is None:
        return list(range(len(mat_ds)))
    tset = set(map(int, target_classes))
    cand = []
    for i in range(len(mat_ds)):
        _img, _tri, y, _uid = mat_ds[i]
        if int(y) in tset:
            cand.append(int(i))
    return cand if len(cand) else list(range(len(mat_ds)))


Selector = Union[None, int, str]


def visualize_selectors_or_random(
    augmenter,  # Augmenter
    mat_ds,
    class_names: List[str],
    target_classes: Optional[Sequence[int]] = None,
    selectors: Optional[Union[Selector, Sequence[Selector]]] = None,
    n_rows: int = 3,
    n_per_type: int = 2,
    base_seed: int = 42,
    only_targets_random: bool = True,
    class_pick: str = "first",  # "first" | "random"
):
    rng = random.Random(int(base_seed))
    uid2i = _build_uid_to_index(mat_ds)
    name2idx = _build_classname_to_idx(class_names)
    cand = _candidate_indices(mat_ds, target_classes if only_targets_random else None)

    # normalize selectors -> slots
    if selectors is None:
        slots: List[Selector] = [None] * int(n_rows)
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

    def pick_by_class(class_idx: int):
        if class_pick == "random":
            idxs = []
            for i in range(len(mat_ds)):
                _img, _tri, y, _uid = mat_ds[i]
                if int(y) == int(class_idx):
                    idxs.append(int(i))
            if not idxs:
                return None
            rng.shuffle(idxs)
            for i in idxs:
                if i not in used_idxs:
                    return i
            return idxs[0]
        else:
            return _find_first_index_by_class(mat_ds, class_idx)

    chosen_idxs = []
    for s in slots:
        if s is None:
            i = pick_random_idx()
            chosen_idxs.append(i); used_idxs.add(i)
            continue

        if isinstance(s, (int, np.integer)):
            uid = int(s)
            if uid not in uid2i:
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
                i = pick_random_idx()
            else:
                cidx = name2idx[key]
                i = pick_by_class(cidx)
                if i is None or i in used_idxs:
                    i = pick_random_idx()
            chosen_idxs.append(i); used_idxs.add(i)
            continue

        i = pick_random_idx()
        chosen_idxs.append(i); used_idxs.add(i)

    size = int(augmenter.cfg.gen_size)
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
            row.append(augmenter.inpaint_background(img, tri, y, seed=int(base_seed + 10 * r + k)))
        for k in range(int(n_per_type)):
            row.append(augmenter.small_edit(img, y, seed=int(base_seed + 100 + 10 * r + k)))
        for k in range(int(n_per_type)):
            row.append(augmenter.text2img(y, seed=int(base_seed + 200 + 10 * r + k)))

        rows.append(row)
        row_titles.append(f"{class_names[y]} (uid={uid})")

    show_grid(rows, row_titles, col_titles)
