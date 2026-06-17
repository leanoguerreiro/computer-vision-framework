"""Funções para carregamento de datasets e dataloaders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .transforms import build_eval_transform, build_train_transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_imagefolder_datasets(
    root_dir: Union[str, Path],
    train_transform=None,
    eval_transform=None,
):
    """Cria os datasets ImageFolder para train/val/test."""
    root = Path(root_dir)
    train_ds = datasets.ImageFolder(root / "train", transform=train_transform or build_train_transform())
    eval_ds = datasets.ImageFolder(root / "val", transform=eval_transform or build_eval_transform())
    test_ds = datasets.ImageFolder(root / "test", transform=eval_transform or build_eval_transform())
    return train_ds, eval_ds, test_ds


def build_dataloaders(
    root_dir: Union[str, Path],
    batch_size: int,
    train_transform=None,
    eval_transform=None,
    train_workers: int = 8,
    eval_workers: int = 4,
    shuffle_train: bool = True,
):
    """Cria os dataloaders de train/val/test com a mesma convenção do projeto."""
    train_ds, val_ds, test_ds = build_imagefolder_datasets(root_dir, train_transform, eval_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=train_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=eval_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=eval_workers)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def get_class_names(root_dir: Union[str, Path], split: str = "train"):
    """Retorna os nomes das classes de um split específico."""
    dataset = datasets.ImageFolder(Path(root_dir) / split)
    return dataset.classes


def build_class_weights(targets: Iterable[int], num_classes: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Cria pesos de classe inversamente proporcionais à frequência."""
    counts = torch.tensor([sum(1 for target in targets if target == idx) for idx in range(num_classes)], dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    if device is not None:
        weights = weights.to(device)
    return weights



