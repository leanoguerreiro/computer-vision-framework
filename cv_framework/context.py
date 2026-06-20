"""Agrupa os recursos de configuração injetáveis para o framework."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class BenchmarkContext:
    # Parâmetros obrigatórios (quem chama o framework tem que fornecer)
    root_dir: Path
    plot_dir: Path
    results_dir: Path

    # Parâmetros opcionais com padrões seguros da biblioteca
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 1e-3
    patience: int = 2
    min_delta: float = 0.001
    train_workers: int = 8
    eval_workers: int = 4
    seed: int = 42

    # Injeção de configurações específicas do modelo
    radimagenet_weights_url: str = (
        "https://huggingface.co/BMEII/RadImageNet/resolve/main/RadImageNet-ResNet50_notop.pth"
    )
    batch_size_overrides: Dict[str, int] = field(default_factory=dict)