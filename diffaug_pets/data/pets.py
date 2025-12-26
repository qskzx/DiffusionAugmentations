from __future__ import annotations
import re
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet


def load_raw_pets(data_root: Path):
    data_root.mkdir(parents=True, exist_ok=True)
    ds_train = OxfordIIITPet(
        root=str(data_root), split="trainval",
        target_types=["category", "segmentation"], download=True
    )
    ds_test = OxfordIIITPet(
        root=str(data_root), split="test",
        target_types=["category", "segmentation"], download=True
    )
    return ds_train, ds_test

def norm_breed(name: str) -> str:
    # "American Bulldog" / "american_bulldog" / "American-Bulldog" -> "american_bulldog"
    s = name.strip()
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)   # пробелы -> _
    s = re.sub(r"_+", "_", s)    # схлопнуть __
    return s.casefold()          # lower + устойчивее

def build_species_by_breed(dataset_root) -> Dict[str, str]:
    ann_dir = dataset_root / "oxford-iiit-pet" / "annotations"
    tv = ann_dir / "trainval.txt"
    if not tv.exists():
        return {}

    breed2species: Dict[str, str] = {}
    with open(tv, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_id, _class_id, species, _breed_id = line.split()
            breed = re.sub(r"_\d+$", "", image_id)        # e.g. "american_bulldog" / "Abyssinian"
            sp = "cat" if int(species) == 1 else "dog"
            breed2species[norm_breed(breed)] = sp         # <-- нормализуем ключ
    return breed2species

def species_token_for_class(class_name: str, breed2species: Dict[str, str]) -> str:
    return breed2species.get(norm_breed(class_name), "pet")  # <-- нормализуем запрос

def stratified_split(labels: np.ndarray, seed: int, train_frac: float = 0.85):
    idxs = np.arange(len(labels))
    rng = np.random.RandomState(int(seed))
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        c_idx = idxs[labels == c]
        rng.shuffle(c_idx)
        cut = int(float(train_frac) * len(c_idx))
        train_idx.extend(c_idx[:cut])
        val_idx.extend(c_idx[cut:])
    return np.array(train_idx), np.array(val_idx)


def build_transforms(img_size: int):
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return train_tf, val_tf


class PetsSubset(Dataset):
    def __init__(self, base, idxs, tf=None):
        self.base = base
        self.idxs = list(map(int, idxs))
        self.tf = tf

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        x, (y, _trimap) = self.base[self.idxs[i]]
        x = self.tf(x) if self.tf else x
        return x, int(y)


class PetsMattingSubset(Dataset):
    """Returns PIL RGB image, trimap PIL, class_idx, uid (= index into base train split)."""
    def __init__(self, base, idxs):
        self.base = base
        self.idxs = list(map(int, idxs))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        x, (y, trimap) = self.base[self.idxs[i]]
        uid = int(self.idxs[i])
        return x.convert("RGB"), trimap, int(y), uid


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(
        ds, batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=int(num_workers), pin_memory=True,
        persistent_workers=False
    )


def build_mat_indices_by_class(mat_ds, num_classes: int):
    byc = {i: [] for i in range(int(num_classes))}
    for idx in range(len(mat_ds)):
        _img, _tri, y, _uid = mat_ds[idx]
        byc[int(y)].append(int(idx))
    return byc
