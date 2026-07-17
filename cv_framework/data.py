"""Funções puras para carregamento de datasets e orquestração de dataloaders
(Refatorado para Alta Performance)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from cv_framework.context import BenchmarkContext
from cv_framework.transforms import build_eval_transform, build_train_transform


def build_imagefolder_datasets(
        root_dir: Union[str, Path],
        train_transform=None,
        eval_transform=None,
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    """Cria os datasets ImageFolder para train/val/test."""
    root = Path(root_dir)
    train_ds = datasets.ImageFolder(
        root / "train", transform=train_transform or build_train_transform(),
    )
    eval_ds = datasets.ImageFolder(
        root / "val", transform=eval_transform or build_eval_transform(),
    )
    test_ds = datasets.ImageFolder(
        root / "test", transform=eval_transform or build_eval_transform(),
    )

    return train_ds, eval_ds, test_ds


def get_class_names(root_dir: Union[str, Path], split: str = "train") -> List[
    str]:
    """Lê os nomes das classes diretamente pela estrutura de pastas sem
    varrer os arquivos de imagem no disco."""
    split_dir = Path(root_dir) / split
    return sorted([d.name for d in split_dir.iterdir() if d.is_dir()])


def build_class_weights(
        targets: Iterable[int], num_classes: int,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Cria pesos de classe vetorizados (O(N) em C++) com proteção contra
    divisão por zero."""
    # Converte para tensor se for lista/iterável e executa bincount em uma
    # única passagem
    if not isinstance(targets, torch.Tensor):
        targets_tensor = torch.tensor(list(targets), dtype=torch.long)
    else:
        targets_tensor = targets.long()

    counts = torch.bincount(targets_tensor, minlength=num_classes).float()

    # Proteção: substitui contagens zero por 1.0 para evitar Infinity ou NaN
    # na divisão
    counts = torch.where(counts == 0, torch.tensor(1.0), counts)

    weights = 1.0 / counts
    weights = weights / weights.sum()  # Normaliza para que a soma seja 1.0

    if device is not None:
        weights = weights.to(device, non_blocking=True)

    return weights


def setup_data_loaders(
        context: BenchmarkContext, batch_size: int,
) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], List[str]]:
    """Orquestrador central com pin_memory e persistent_workers para acelerar
    a alimentação da GPU."""
    transform_train = build_train_transform()
    transform_eval = build_eval_transform()

    ds_train, ds_val, ds_test = build_imagefolder_datasets(
        context.root_dir,
        train_transform=transform_train,
        eval_transform=transform_eval,
    )

    use_pin_memory = torch.cuda.is_available()

    # Persistent workers só deve ser ativado se num_workers > 0
    train_persistent = context.train_workers > 0
    eval_persistent = context.eval_workers > 0

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=context.train_workers,
        pin_memory=use_pin_memory,
        persistent_workers=train_persistent,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=context.eval_workers,
        pin_memory=use_pin_memory,
        persistent_workers=eval_persistent,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=context.eval_workers,
        pin_memory=use_pin_memory,
        persistent_workers=eval_persistent,
    )

    return (train_loader, val_loader, test_loader), ds_train.classes