"""Componentes de treino compartilhados."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class EarlyStopping:
    """Para o treino quando uma métrica monitorada não melhora."""

    patience: int
    min_delta: float
    path: str
    counter: int = 0
    best_score: Optional[float] = None
    triggered: bool = False
    best_data: dict = field(default_factory=dict)

    def step(self, score: float, model: nn.Module, epoch_data: dict) -> bool:
        improved = self.best_score is None or score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_data = epoch_data
            torch.save(model.state_dict(), self.path)
            print(f"   ✅ Novo melhor F1 Val: {score:.4f} — modelo salvo.")
        else:
            self.counter += 1
            print(f"   ⏳ Sem melhora ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.triggered = True

        return self.triggered


