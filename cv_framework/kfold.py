from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets
from sklearn.model_selection import StratifiedKFold
from torchvision.transforms import v2

from cv_framework.transforms import build_train_transform, build_eval_transform


class FoldDataset(Dataset):
    """Wrapper funcional que aplica transformações dinamicamente a índices de um ConcatDataset ou ImageFolder."""

    def __init__(
            self,
            base_dataset: Dataset,
            indices: np.ndarray | List[int],
            transform: v2.Compose,
            targets_pool: List[int],
    ):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        # Armazena os targets corretos deste fold específico para o cálculo de pesos
        self.targets = [targets_pool[i] for i in indices]

    def __getitem__(self, idx: int):
        # Acessa a imagem bruta no índice mapeado
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
    """
    Carrega o dataset da estrutura pré-dividida e gera Folds Estratificados.

    Se merge_val=True, unifica as pastas /train e /val em memória para formar o pool do K-Fold.
    """
    root = Path(root_dir)

    # 1. Carrega os ImageFolders BRUTOS (sem transformações)
    ds_train_raw = datasets.ImageFolder(root / "train", transform=None)
    class_names = ds_train_raw.classes

    if merge_val and (root / "val").exists():
        ds_val_raw = datasets.ImageFolder(root / "val", transform=None)
        # Mescla Train (14.000) + Val (4.000) = 18.000 imagens em memória
        base_ds = ConcatDataset([ds_train_raw, ds_val_raw])
        # Concatena a lista de targets de todos os sub-datasets mesclados
        targets = ds_train_raw.targets + ds_val_raw.targets
        print(
            f"  📦 Pool K-Fold formado: {len(base_ds)} imagens (train + val mesclados)."
            )
    else:
        base_ds = ds_train_raw
        targets = ds_train_raw.targets
        print(
            f"  📦 Pool K-Fold formado: {len(base_ds)} imagens (apenas train)."
            )

    # 2. Configura o gerador de K-Fold Estratificado
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    # 3. Inicializa os pipelines limpos e de treino
    train_tf = build_train_transform(image_size)
    eval_tf = build_eval_transform(image_size)

    folds_data = []
    for train_idx, val_idx in skf.split(np.zeros(len(targets)), targets):
        # Injeta o transform de TREINO no fold de treino
        ds_train = FoldDataset(
            base_ds, train_idx, transform=train_tf, targets_pool=targets
            )
        # Injeta o transform de VALIDAÇÃO LIMPO no fold de validação (OOF)
        ds_val = FoldDataset(
            base_ds, val_idx, transform=eval_tf, targets_pool=targets
            )

        folds_data.append((ds_train, ds_val, class_names))

    return folds_data