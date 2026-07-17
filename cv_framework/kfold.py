"""Módulo de geração de Folds Estratificados (Refatorado - Vetorizado e sem alocações inúteis)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset, Dataset
from torchvision import datasets
from torchvision.transforms import v2

from cv_framework.transforms import build_eval_transform, build_train_transform


class FoldDataset(Dataset):
    """Wrapper funcional que aplica transformações dinamicamente a índices de um ConcatDataset ou ImageFolder."""

    def __init__(
            self,
            base_dataset: Dataset,
            indices: np.ndarray | List[int],
            transform: v2.Compose,
            targets_pool: np.ndarray | List[int],
    ):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

        # Otimização: Fatiamento vetorizado instantâneo sem loops Python
        targets_array = np.asarray(targets_pool)
        self.targets = targets_array[self.indices].tolist()

    def __getitem__(self, idx: int):
        img, label = self.base_dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.indices)


def get_stratified_kfold_datasets(
        root_dir: str | Path,
        k: int = 5,
        seed: int = 42,
        image_size: int = 224,
        merge_val: bool = True,
) -> List[Tuple[FoldDataset, FoldDataset, List[str]]]:
    """Carrega o dataset da estrutura pré-dividida e gera Folds Estratificados com alta performance de memória."""
    root = Path(root_dir)

    ds_train_raw = datasets.ImageFolder(root / "train", transform=None)
    class_names = ds_train_raw.classes

    if merge_val and (root / "val").exists():
        ds_val_raw = datasets.ImageFolder(root / "val", transform=None)
        base_ds = ConcatDataset([ds_train_raw, ds_val_raw])
        targets = np.array(
            ds_train_raw.targets + ds_val_raw.targets, dtype=np.int64
            )
        print(
            f"  📦 Pool K-Fold formado: {len(base_ds)} imagens (train + val mesclados)."
            )
    else:
        base_ds = ds_train_raw
        targets = np.array(ds_train_raw.targets, dtype=np.int64)
        print(
            f"  📦 Pool K-Fold formado: {len(base_ds)} imagens (apenas train)."
            )

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    train_tf = build_train_transform(image_size)
    eval_tf = build_eval_transform(image_size)

    folds_data = []

    # Otimização: Usamos np.empty em vez de np.zeros para evitar custo de inicialização na RAM
    dummy_x = np.empty(len(targets), dtype=np.int8)

    for train_idx, val_idx in skf.split(dummy_x, targets):
        ds_train = FoldDataset(
            base_ds, train_idx, transform=train_tf, targets_pool=targets
            )
        ds_val = FoldDataset(
            base_ds, val_idx, transform=eval_tf, targets_pool=targets
            )
        folds_data.append((ds_train, ds_val, class_names))

    return folds_data