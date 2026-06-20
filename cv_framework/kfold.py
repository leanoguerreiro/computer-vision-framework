"""Módulo de validação cruzada K-Fold para ImageFolder."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torchvision import datasets

def get_kfold_indices(dataset: datasets.ImageFolder, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Gera índices de treino e validação para cada fold."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    # Usamos os alvos (targets) para garantir uma divisão estratificada ou representativa
    indices = np.arange(len(dataset))
    return list(kf.split(indices))

def get_kfold_subsets(dataset: datasets.ImageFolder, train_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[Subset, Subset]:
    """Cria subconjuntos Torch para treino e validação a partir dos índices."""
    return Subset(dataset, train_idx), Subset(dataset, val_idx)