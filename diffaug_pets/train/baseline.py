from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, f1_score


def train_one_epoch(model, dl, opt, loss_fn, device: str):
    model.train()
    total = 0.0
    for x, y in tqdm(dl, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += float(loss.item()) * x.size(0)
    return total / len(dl.dataset)


@torch.no_grad()
def evaluate_with_cm(model, dl, num_classes: int, device: str):
    model.eval()
    all_y, all_p = [], []
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    for x, y in tqdm(dl, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total += float(loss_fn(logits, y).item()) * x.size(0)
        pred = logits.argmax(1)
        all_y.append(y.detach().cpu().numpy())
        all_p.append(pred.detach().cpu().numpy())

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    cm = confusion_matrix(y, p, labels=list(range(int(num_classes))))
    acc = float((p == y).mean())
    f1m = float(f1_score(y, p, average="macro", labels=list(range(int(num_classes)))))
    return {"loss": total / len(dl.dataset), "acc": acc, "f1_macro": f1m, "cm": cm}


def select_target_classes(cm: np.ndarray, top_n: int = 10):
    cm = np.asarray(cm)
    totals = cm.sum(axis=1)
    correct = np.diag(cm)
    errors = totals - correct
    err_rate = errors / np.maximum(totals, 1)
    idx = np.argsort(-errors)[: int(top_n)]
    idx = [int(i) for i in idx]
    return idx, errors, err_rate


def build_conf_map(cm: np.ndarray, per_class: int = 3) -> Dict[int, List[int]]:
    cm = np.asarray(cm).copy()
    np.fill_diagonal(cm, 0)
    out: Dict[int, List[int]] = {}
    for i in range(cm.shape[0]):
        row = cm[i]
        jj = row.argsort()[::-1][: int(per_class)]
        out[int(i)] = [int(j) for j in jj if row[j] > 0]
    return out
