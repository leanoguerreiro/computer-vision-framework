from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import TypedDict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from cv_framework.data import build_class_weights
from cv_framework.models import build_model
from cv_framework.context import BenchmarkContext
from cv_framework.metrics import evaluate_split
from cv_framework.plots import (
    plot_confidence_distribution,
    generate_model_plots,
)
from cv_framework.explainability import (
    plot_latent_space,
    generate_gradcam_samples,
    generate_transformer_samples,
)


# --- 1. Estruturas de Estado ---

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
    """Avalia a métrica e retorna um NOVO estado. Salva o modelo como efeito colateral."""
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


# --- 2. Funções Isoladas de Época ---

def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler.LRScheduler,
        device: torch.device,
        epoch_desc: str
) -> Tuple[float, float, float]:
    """Executa uma época inteira de treinamento."""
    model.train()
    running_loss, correct = 0.0, 0
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time = time.time()

    for inputs, labels in tqdm(loader, desc=epoch_desc, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    if torch.cuda.is_available(): torch.cuda.synchronize()
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)

    return epoch_loss, epoch_acc, epoch_time


def validate_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> Tuple[float, float, List[int], List[int], np.ndarray]:
    """Executa uma época de validação sem alterar pesos."""
    model.eval()
    running_loss, correct = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            running_loss += criterion(outputs, labels).item() * inputs.size(0)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)

    return epoch_loss, epoch_acc, all_preds, all_labels, np.vstack(all_probs)


def generate_post_training_visualizations(
        model: nn.Module,
        model_name: str,
        test_loader: DataLoader,
        device: torch.device,
        plot_dir: Path,
        class_names: List[str],
        seed: int
) -> None:
    """Roteia e gera as visualizações de interpretabilidade e espaço latente."""
    plot_latent_space(model, model_name, test_loader, device, plot_dir, class_names, seed)

    if hasattr(model, 'transformer'):
        print(f"  🧠 Arquitetura Transformer detectada para {model_name}. Gerando mapas de atenção...")
        generate_transformer_samples(
            model=model, model_name=model_name, test_loader=test_loader,
            device=device, plot_dir=plot_dir, class_names=class_names, samples_per_class=5
        )
    else:
        print(f"  📷 Arquitetura CNN detectada para {model_name}. Gerando Grad-CAM...")
        original_grad_states = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad = True

        generate_gradcam_samples(
            model=model, model_name=model_name, test_loader=test_loader,
            device=device, plot_dir=plot_dir, class_names=class_names, samples_per_class=5
        )

        for param, requires_grad in zip(model.parameters(), original_grad_states):
            param.requires_grad = requires_grad


# --- 3. Orquestrador Principal ---

def train_model_pipeline(
        model_name: str,
        context: BenchmarkContext,
        loaders: Tuple[DataLoader, DataLoader, DataLoader],
        class_names: List[str]
) -> dict:
    """Orquestrador declarativo do ciclo de vida de treinamento do modelo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = loaders

    batch_size = context.batch_size_overrides.get(model_name, context.batch_size)
    print(f"\n{'=' * 55}\n  PROCESSANDO: {model_name.upper()}\n{'=' * 55}")
    if batch_size != context.batch_size:
        print(f"  ⚠️  Batch reduzido para {batch_size} (limite de VRAM)")

    num_classes = len(class_names)

    # 3.1 Setup de Arquitetura e Otimização
    model = build_model(
        model_name, num_classes, pretrained=False,
        results_dir=str(context.results_dir), radimagenet_weights_url=context.radimagenet_weights_url
    ).to(device)

    weights = build_class_weights([t[1] for t in train_loader.dataset.samples], num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=context.learning_rate)

    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=context.learning_rate, steps_per_epoch=len(train_loader),
        epochs=context.num_epochs, pct_start=0.3,
    )

    # 3.2 Setup de Estado
    model_dir = context.results_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = model_dir / "best.pth"

    es_state = init_early_stopping(context.patience, context.min_delta, str(best_weights_path))
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "f1_per_class": []}
    train_time_acumulado = 0.0

    # 3.3 Loop de Épocas
    for epoch in range(context.num_epochs):
        # Treino
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, f"Treino E{epoch + 1:02}"
        )
        train_time_acumulado += epoch_time

        # Validação
        val_loss, val_acc, val_preds, val_labels, val_probs = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Métricas Temporárias de Validação
        epoch_f1_classes = f1_score(val_labels, val_preds, average=None, zero_division=0)
        epoch_f1_macro = epoch_f1_classes.mean()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["f1_per_class"].append(epoch_f1_classes.tolist())

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"E{epoch + 1:02} | LR: {current_lr:.6f} | Val Loss: {val_loss:.4f} | Val F1-Macro: {epoch_f1_macro:.4f}")

        # Early Stopping continuará respeitando apenas o F1-Macro
        epoch_data = {"y_true": val_labels, "y_probs": val_probs, "y_preds": val_preds}
        es_state = step_early_stopping(es_state, epoch_f1_macro, model, epoch_data)

        if es_state["triggered"]:
            print(f"\n  Early stopping na época {epoch + 1} — melhor F1 Val: {es_state['best_score']:.4f}")
            break

    # 3.4 Avaliação Final e Gráficos
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best_val_data = es_state["best_data"]
    generate_model_plots(
        model_name, context.plot_dir, class_names, history,
        best_val_data["y_true"], best_val_data["y_probs"], best_val_data["y_preds"]
    )

    print("\n  🔍 Extraindo métricas finais do conjunto de TESTE...")
    test_eval = evaluate_split(model, test_loader, device, num_classes, f"Teste {model_name}")
    plot_confidence_distribution(model_name, context.plot_dir, test_eval["probs_matrix"], test_eval["preds"],
                                 test_eval["labels"])

    print("  🔍 Extraindo métricas finais do conjunto de TREINO...")
    train_eval = evaluate_split(model, train_loader, device, num_classes, f"Treino Final {model_name}")

    # Log básico no terminal para não inundar a tela
    print(
        f"  🏆 TESTE  | Acc: {test_eval['accuracy']:.4f} | Prec: {test_eval['precision']:.4f} | Rec: {test_eval['recall']:.4f} | F1: {test_eval['f1_macro']:.4f} | Spec: {test_eval['specificity']:.4f} | MCC: {test_eval['mcc']:.4f} | AUC: {test_eval['auc_macro']:.4f} | MSE: {test_eval['mse']:.4f}")

    generate_post_training_visualizations(
        model, model_name, test_loader, device, context.plot_dir, class_names, context.seed
    )

    # 3.5 Limpeza de Memória
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Retorna o dicionário expandido para o Polars salvar no CSV
    return {
        "Modelo": model_name,

        # Métricas de Treino
        "Train_Accuracy": train_eval["accuracy"],
        "Train_Precision": train_eval["precision"],
        "Train_Recall": train_eval["recall"],
        "Train_F1-Macro": train_eval["f1_macro"],
        "Train_Specificity": train_eval["specificity"],
        "Train_MCC": train_eval["mcc"],
        "Train_AUC-Macro": train_eval["auc_macro"],
        "Train_MSE": train_eval["mse"],

        # Métrica de Validação Base
        "Val_F1-Macro": es_state["best_score"],

        # Métricas de Teste
        "Test_Accuracy": test_eval["accuracy"],
        "Test_Precision": test_eval["precision"],
        "Test_Recall": test_eval["recall"],
        "Test_F1-Macro": test_eval["f1_macro"],
        "Test_Specificity": test_eval["specificity"],
        "Test_MCC": test_eval["mcc"],
        "Test_AUC-Macro": test_eval["auc_macro"],
        "Test_MSE": test_eval["mse"],

        # Métricas Computacionais
        "Train_Time_s": train_time_acumulado,
        "Inference_Time_s": test_eval["time_s"],
        "Inference_ms_per_img": test_eval["time_ms_per_img"],
        "Parâmetros": num_params,
        "Batch": batch_size,

        # Dados Brutos
        "test_preds": test_eval["preds"],
        "test_labels": test_eval["labels"],
    }

def train_kfold_fold(
    model_name: str,
    context: BenchmarkContext,
    loaders: Tuple[DataLoader, DataLoader],  # <--- Assinatura estrita: apenas Treino e Validação
    class_names: List[str],
    fold_idx: int,
) -> dict:
    """Orquestrador especializado e ultraleve para execução de um único Fold no K-Fold.

    Focado em performance pura: sem geração de gráficos, sem UMAP/Grad-CAM e sem reavaliação de treino.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = loaders
    num_classes = len(class_names)

    batch_size = context.batch_size_overrides.get(model_name, context.batch_size)
    print(f"\n{'=' * 20} FOLD {fold_idx} : {model_name.upper()} {'=' * 20}")

    # 1. Setup de Arquitetura e Otimização
    model = build_model(
        model_name,
        num_classes,
        pretrained=False,
        results_dir=str(context.results_dir),
        radimagenet_weights_url=context.radimagenet_weights_url,
    ).to(device)

    targets = train_loader.dataset.targets
    weights = build_class_weights(targets, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=context.learning_rate)

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=context.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=context.num_epochs,
        pct_start=0.3,
    )

    # 2. Setup de Estado (Isolado por Fold)
    model_dir = context.results_dir / model_name / f"fold_{fold_idx}"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = model_dir / f"best_{model_name}_f{fold_idx}.pth"

    es_state = init_early_stopping(
        context.patience, context.min_delta, str(best_weights_path)
    )
    train_time_acumulado = 0.0

    # 3. Loop de Épocas
    for epoch in range(context.num_epochs):
        train_loss, train_acc, epoch_time = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            f"F{fold_idx} Treino E{epoch + 1:02}",
        )
        train_time_acumulado += epoch_time

        val_loss, val_acc, val_preds, val_labels, val_probs = (
            validate_one_epoch(model, val_loader, criterion, device)
        )

        epoch_f1_classes = f1_score(
            val_labels, val_preds, average=None, zero_division=0
        )
        epoch_f1_macro = epoch_f1_classes.mean()

        print(
            f"E{epoch + 1:02} | Loss: {val_loss:.4f} | Val F1-Macro: {epoch_f1_macro:.4f}"
        )

        epoch_data = {
            "y_true": val_labels,
            "y_probs": val_probs,
            "y_preds": val_preds,
        }
        es_state = step_early_stopping(
            es_state, epoch_f1_macro, model, epoch_data
        )

        if es_state["triggered"]:
            print(
                f"  ⏹️ Early stopping na época {epoch + 1} — melhor F1 Val: {es_state['best_score']:.4f}"
            )
            break

    # 4. Avaliação Final Out-Of-Fold (OOF)
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("  🔍 Extraindo métricas finais Out-Of-Fold (Validação)...")
    oof_eval = evaluate_split(
        model, val_loader, device, num_classes, f"OOF F{fold_idx} {model_name}"
    )

    print(
        f"  🏆 FOLD {fold_idx} OOF | Acc: {oof_eval['accuracy']:.4f} | F1: {oof_eval['f1_macro']:.4f} | AUC: {oof_eval['auc_macro']:.4f}"
    )

    # 5. Limpeza Agressiva de Memória
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Dicionário enxuto focado apenas na agregação estatística do K-Fold
    return {
        "Modelo": model_name,
        "Fold": fold_idx,
        "Test_Accuracy": oof_eval["accuracy"],
        "Test_Precision": oof_eval["precision"],
        "Test_Recall": oof_eval["recall"],
        "Test_F1-Macro": oof_eval["f1_macro"],
        "Test_Specificity": oof_eval["specificity"],
        "Test_MCC": oof_eval["mcc"],
        "Test_AUC-Macro": oof_eval["auc_macro"],
        "Test_MSE": oof_eval["mse"],
        "Train_Time_s": train_time_acumulado,
        "Inference_ms_per_img": oof_eval["time_ms_per_img"],
        "Parâmetros": num_params,
        "oof_preds": oof_eval["preds"],
        "oof_labels": oof_eval["labels"],
    }