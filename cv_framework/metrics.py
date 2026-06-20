"""Métricas de avaliação e inferência (Paradigma Funcional)."""

from __future__ import annotations

import time
from typing import Tuple, List

import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm

def calculate_auc(labels: List[int], probs_matrix: np.ndarray, num_classes: int) -> float:
    """Calcula a métrica AUC-ROC baseada no número de classes."""
    if num_classes == 2:
        return roc_auc_score(labels, probs_matrix[:, 1])
    return roc_auc_score(labels, probs_matrix, multi_class="ovr", average="macro")

def calculate_specificity(labels: List[int], preds: List[int], num_classes: int) -> float:
    """Calcula a especificidade macro (True Negative Rate)."""
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    specificities = []

    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        if (tn + fp) == 0:
            specificities.append(0.0)
        else:
            specificities.append(tn / (tn + fp))

    return float(np.mean(specificities))

def calculate_all_metrics(labels: List[int], preds: List[int], probs_matrix: np.ndarray, num_classes: int) -> dict:
    """Calcula um dicionário completo e padronizado de métricas de classificação."""
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    auc_m = calculate_auc(labels, probs_matrix, num_classes)
    spec = calculate_specificity(labels, preds, num_classes)

    # Cálculo do MSE entre as probabilidades contínuas e o rótulo real (One-Hot)
    y_true_bin = label_binarize(labels, classes=list(range(num_classes)))
    if num_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    mse = mean_squared_error(y_true_bin, probs_matrix)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1,
        "specificity": spec,
        "mcc": mcc,
        "auc_macro": auc_m,
        "mse": mse
    }

def run_inference(model: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> Tuple[List[int], List[int], np.ndarray, float]:
    """Executa inferência e retorna predições limpas."""
    preds_list, labels_list, probs_list = [], [], []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            probs_list.append(probs.cpu().numpy())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start
    return preds_list, labels_list, np.vstack(probs_list), total_time

def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int, desc: str) -> dict:
    """Extrai todas as métricas a partir da inferência."""
    preds, labels, probs_matrix, total_time = run_inference(model, loader, device, desc)
    dataset_size = len(loader.dataset)

    # Chama o novo hub de métricas
    metrics = calculate_all_metrics(labels, preds, probs_matrix, num_classes)

    return {
        "preds": preds,
        "labels": labels,
        "probs_matrix": probs_matrix,
        **metrics, # Desempacota o dicionário com todas as métricas (acc, mcc, etc)
        "time_s": total_time,
        "time_ms_per_img": (total_time / dataset_size) * 1000,
    }