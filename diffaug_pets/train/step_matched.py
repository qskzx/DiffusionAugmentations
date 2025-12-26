from __future__ import annotations

import math
from typing import Callable, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset
from sklearn.metrics import f1_score

from ..data.pets import make_loader
from ..utils.seed import seed_everything


class SyntheticClsDataset(Dataset):
    """Dataset wrapper for synthetic images listed in a manifest DataFrame.

    Expected df columns: path, y (int).
    """
    def __init__(self, df: pd.DataFrame, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[int(idx)]
        x = Image.open(r["path"]).convert("RGB")
        x = self.tf(x) if self.tf else x
        return x, int(r["y"])


@torch.no_grad()
def evaluate_full_and_target(
    model,
    dl,
    num_classes: int,
    target_set: Set[int],
    device: str,
) -> dict:
    model.eval()
    all_y, all_p = [], []
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total += float(loss_fn(logits, y).item()) * x.size(0)
        pred = logits.argmax(1)
        all_y.append(y.detach().cpu().numpy())
        all_p.append(pred.detach().cpu().numpy())

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)

    overall_acc = float((p == y).mean())
    overall_f1 = float(f1_score(y, p, average="macro", labels=list(range(int(num_classes)))))

    mask = np.isin(y, list(target_set))
    if mask.any():
        y_t = y[mask]
        p_t = p[mask]
        target_acc = float((p_t == y_t).mean())
        target_f1 = float(f1_score(y_t, p_t, average="macro", labels=sorted(list(target_set))))
    else:
        target_acc, target_f1 = float("nan"), float("nan")

    return {
        "loss": float(total / len(dl.dataset)),
        "acc": overall_acc,
        "f1_macro": overall_f1,
        "target_acc": target_acc,
        "target_f1_macro": target_f1,
    }


def infinite_loader(dl):
    while True:
        for batch in dl:
            yield batch


def _train_steps_with_eval(
    model,
    train_dl,
    val_dl,
    *,
    total_steps: int,
    eval_every: int,
    cfg,
    num_classes: int,
    target_set: Set[int],
    device: str,
    verbose: bool,
    print_fn: Callable[[str], None],
):
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.CrossEntropyLoss()
    it = infinite_loader(train_dl)

    best_state = None
    best_val = -1.0

    model.train()
    for step in range(1, int(total_steps) + 1):
        x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

        if (step % int(eval_every) == 0) or (step == int(total_steps)):
            va = evaluate_full_and_target(
                model, val_dl, num_classes=int(num_classes), target_set=target_set, device=device
            )
            if verbose:
                print_fn(
                    f"Step {step}/{total_steps} | val_acc={va['acc']:.3f} | val_f1={va['f1_macro']:.3f} "
                    f"| target_acc={va['target_acc']:.3f} | target_f1={va['target_f1_macro']:.3f}"
                )
            if va["f1_macro"] > best_val:
                best_val = va["f1_macro"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_step_matched(
    *,
    model_ctor,
    train_real_ds,
    val_ds,
    test_ds,
    train_tf,
    syn_df: pd.DataFrame,
    cfg,
    target_set: Set[int],
    device: str,
    title: str,
    num_classes: int,
    seed: Optional[int] = None,
    verbose: bool = True,
    print_fn: Callable[[str], None] = print,
) -> Tuple[dict, dict, int, int, int]:
    """Train classifier on real-only vs (real + synthetic) in a *step-matched* setup.

    Returns:
        teR, teM, total_steps, n_syn, n_mix
    """

    syn_ds = SyntheticClsDataset(syn_df, tf=train_tf) if len(syn_df) > 0 else None
    mix_ds = ConcatDataset([train_real_ds, syn_ds]) if syn_ds is not None else train_real_ds

    bs = int(cfg.batch_size)
    train_real_dl = make_loader(train_real_ds, bs, shuffle=True, num_workers=int(cfg.num_workers))
    train_mix_dl = make_loader(mix_ds, bs, shuffle=True, num_workers=int(cfg.num_workers))
    val_dl = make_loader(val_ds, bs, shuffle=False, num_workers=int(cfg.num_workers))
    test_dl = make_loader(test_ds, bs, shuffle=False, num_workers=int(cfg.num_workers))

    steps_mix_per_epoch = math.ceil(len(mix_ds) / bs)
    total_steps = int(steps_mix_per_epoch * int(cfg.epochs))
    eval_every = int(max(25, total_steps // 6))

    if verbose:
        print_fn(f"\n=== {title} (STEP-MATCHED) ===")
        print_fn(
            f"Total steps BOTH = {total_steps} (eval_every={eval_every}) | syn={len(syn_df)} | mix={len(mix_ds)}"
        )

    # REAL
    if verbose:
        print_fn("\n--- REAL ---")
    if seed is not None:
        seed_everything(int(seed))
    mR = model_ctor().to(device)
    mR = _train_steps_with_eval(
        mR,
        train_real_dl,
        val_dl,
        total_steps=total_steps,
        eval_every=eval_every,
        cfg=cfg,
        num_classes=num_classes,
        target_set=target_set,
        device=device,
        verbose=verbose,
        print_fn=print_fn,
    )
    teR = evaluate_full_and_target(mR, test_dl, num_classes=int(num_classes), target_set=target_set, device=device)
    if verbose:
        print_fn(
            f"TEST REAL | acc={teR['acc']:.3f} | f1={teR['f1_macro']:.3f} "
            f"| target_acc={teR['target_acc']:.3f} | target_f1={teR['target_f1_macro']:.3f}"
        )

    # MIX
    if verbose:
        print_fn("\n--- MIX ---")
    if seed is not None:
        seed_everything(int(seed))
    mM = model_ctor().to(device)
    mM = _train_steps_with_eval(
        mM,
        train_mix_dl,
        val_dl,
        total_steps=total_steps,
        eval_every=eval_every,
        cfg=cfg,
        num_classes=num_classes,
        target_set=target_set,
        device=device,
        verbose=verbose,
        print_fn=print_fn,
    )
    teM = evaluate_full_and_target(mM, test_dl, num_classes=int(num_classes), target_set=target_set, device=device)
    if verbose:
        print_fn(
            f"TEST MIX  | acc={teM['acc']:.3f} | f1={teM['f1_macro']:.3f} "
            f"| target_acc={teM['target_acc']:.3f} | target_f1={teM['target_f1_macro']:.3f}"
        )

    return teR, teM, total_steps, int(len(syn_df)), int(len(mix_ds))
