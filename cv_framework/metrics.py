"""Métricas de avaliação e inferência (Paradigma Funcional - Alta
Performance)."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm


# =====================================================================
# 1. FUNÇÕES MATEMÁTICAS E DE MÉTRICAS
# =====================================================================

def calculate_auc(
        labels: np.ndarray, probs_matrix: np.ndarray, num_classes: int,
) -> float:
    """Calcula a métrica AUC-ROC baseada no número de classes."""
    if num_classes == 2:
        return float(roc_auc_score(labels, probs_matrix[:, 1]))
    return float(
        roc_auc_score(labels, probs_matrix, multi_class="ovr", average="macro"),
    )


def calculate_specificity(
        labels: np.ndarray, preds: np.ndarray, num_classes: int,
) -> float:
    """Calcula a especificidade macro vetorizada (sem loops Python)."""
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # Álgebra vetorial: calcula TP, FP, FN e TN para todas as classes
    # simultaneamente
    total_samples = np.sum(cm)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = total_samples - (tp + fp + fn)

    # Evita divisão por zero usando np.divide com máscara onde denominador != 0
    denom = tn + fp
    specificities = np.divide(
        tn, denom, out=np.zeros_like(tn, dtype=float), where=(denom != 0),
    )

    return float(np.mean(specificities))


def calculate_all_metrics(
        labels: np.ndarray, preds: np.ndarray, probs_matrix: np.ndarray,
        num_classes: int,
) -> dict:
    """Calcula um dicionário completo e padronizado de métricas com
    performance otimizada."""
    acc = float(accuracy_score(labels, preds))
    mcc = float(matthews_corrcoef(labels, preds))

    # Otimização: calcula precisão, recall e f1 em uma única passada de CPU
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0,
    )

    auc_m = calculate_auc(labels, probs_matrix, num_classes)
    spec = calculate_specificity(labels, preds, num_classes)

    # Binarização robusta para cálculo do MSE
    y_true_bin = label_binarize(labels, classes=list(range(num_classes)))
    if num_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    mse = float(mean_squared_error(y_true_bin, probs_matrix))

    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1_macro": float(f1),
        "specificity": spec,
        "mcc": mcc,
        "auc_macro": auc_m,
        "mse": mse,
    }


# =====================================================================
# 2. MOTORES DE INFERÊNCIA E AVALIAÇÃO
# =====================================================================

def run_inference(
        model: nn.Module, loader: DataLoader, device: torch.device, desc: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Executa inferência com gerenciamento de memória otimizado e
    torch.inference_mode."""
    model.eval()
    preds_list, labels_list, probs_list = [], [], []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()  # perf_counter é mais preciso para
    # benchmarks que time.time()

    # inference_mode desativa rastreamento de versão no motor C++ (muito mais
    # rápido que no_grad)
    with torch.inference_mode():
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Acumula os tensores na CPU em vez de chamar .numpy() e realocar
            # listas em cada iteração
            preds_list.append(preds.cpu())
            labels_list.append(labels.cpu())
            probs_list.append(probs.cpu())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    # Apenas UMA alocação de memória contígua e conversão para NumPy no final
    # de cada loop
    final_preds = torch.cat(preds_list).numpy()
    final_labels = torch.cat(labels_list).numpy()
    final_probs = torch.cat(probs_list).numpy()

    return final_preds, final_labels, final_probs, total_time


def evaluate_split(
        model: nn.Module, loader: DataLoader, device: torch.device,
        num_classes: int, desc: str,
) -> dict:
    """Extrai todas as métricas a partir da inferência."""
    preds, labels, probs_matrix, total_time = run_inference(
        model, loader, device, desc,
    )
    dataset_size = len(loader.dataset)

    # Chama o hub de métricas otimizado
    metrics = calculate_all_metrics(labels, preds, probs_matrix, num_classes)

    return {
        "preds": preds,
        "labels": labels,
        "probs_matrix": probs_matrix,
        **metrics,
        "time_s": total_time,
        "time_ms_per_img": (total_time / dataset_size) * 1000,
    }