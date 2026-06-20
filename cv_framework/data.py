"""Funções puras para carregamento de datasets e orquestração de dataloaders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union, Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from cv_framework.transforms import build_eval_transform, build_train_transform
from cv_framework.context import BenchmarkContext

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


def get_class_names(root_dir: Union[str, Path], split: str = "train") -> List[str]:
    """Retorna os nomes das classes de um split específico."""
    dataset = datasets.ImageFolder(Path(root_dir) / split)
    return dataset.classes


def build_class_weights(targets: Iterable[int], num_classes: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Cria pesos de classe inversamente proporcionais à frequência para lidar com desbalanceamento."""
    counts = torch.tensor([sum(1 for target in targets if target == idx) for idx in range(num_classes)], dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum()

    if device is not None:
        weights = weights.to(device)

    return weights


def setup_data_loaders(context: BenchmarkContext, batch_size: int) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], List[str]]:
    """
    Orquestrador central de dados: Configura os dataloaders garantindo que o multiprocessamento
    rode de forma eficiente no seu SO, evitando gargalos de I/O na GPU.
    """
    transform_train = build_train_transform()
    transform_eval = build_eval_transform()

    ds_train, ds_val, ds_test = build_imagefolder_datasets(
        context.root_dir,
        train_transform=transform_train,
        eval_transform=transform_eval,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=context.train_workers)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=context.eval_workers)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=context.eval_workers)

    return (train_loader, val_loader, test_loader), ds_train.classes