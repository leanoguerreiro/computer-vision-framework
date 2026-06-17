"""Componentes de treino compartilhados."""

from typing import TypedDict, Optional
import torch
import torch.nn as nn

class EarlyStoppingState(TypedDict):
    patience: int
    min_delta: float
    path: str
    counter: int
    best_score: Optional[float]
    triggered: bool
    best_data: dict

def init_early_stopping(patience: int, min_delta: float, path: str) -> EarlyStoppingState:
    """Inicializa o estado do early stopping."""
    return {
        "patience": patience,
        "min_delta": min_delta,
        "path": path,
        "counter": 0,
        "best_score": None,
        "triggered": False,
        "best_data": {}
    }

def step_early_stopping(
    state: EarlyStoppingState,
    score: float,
    model: nn.Module,
    epoch_data: dict
) -> EarlyStoppingState:
    """
    Avalia a métrica e retorna um NOVO estado (imutabilidade).
    Salva o modelo como efeito colateral se houver melhora.
    """
    improved = state["best_score"] is None or score > state["best_score"] + state["min_delta"]

    new_state = dict(state)

    if improved:
        new_state["best_score"] = score
        new_state["counter"] = 0
        new_state["best_data"] = epoch_data
        torch.save(model.state_dict(), state["path"])
        print(f"   ✅ Novo melhor F1 Val: {score:.4f} — modelo salvo.")
    else:
        new_state["counter"] += 1
        print(f"   ⏳ Sem melhora ({new_state['counter']}/{state['patience']})")
        if new_state["counter"] >= state["patience"]:
            new_state["triggered"] = True

    return new_state