from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import TypedDict, Tuple, List, Optional, FrozenSet

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from cv_framework.context import BenchmarkContext
from cv_framework.data import build_class_weights
from cv_framework.explainability import (
    plot_latent_space,
    generate_gradcam_samples,
    generate_transformer_samples,
)
from cv_framework.metrics import evaluate_split
from cv_framework.models import build_model
from cv_framework.plots import (
    plot_confidence_distribution,
    generate_model_plots,
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


def init_early_stopping(
        patience: int, min_delta: float, path: str,
) -> EarlyStoppingState:
    """Inicializa o estado do early stopping."""
    return {
        "patience": patience,
        "min_delta": min_delta,
        "path": path,
        "counter": 0,
        "best_score": None,
        "triggered": False,
        "best_data": {},
    }


def step_early_stopping(
        state: EarlyStoppingState,
        score: float,
        model: nn.Module,
        epoch_data: dict,
) -> EarlyStoppingState:
    """Avalia a métrica e retorna um NOVO estado. Salva o modelo como efeito
    colateral."""
    improved = state["best_score"] is None or score > state["best_score"] + \
               state["min_delta"]
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
        device: torch.device,
        epoch_desc: str,
        scheduler: Optional[lr_scheduler.LRScheduler] = None,
) -> Tuple[float, float, float]:
    """Executa uma época inteira de treinamento."""
    model.train()
    running_loss, correct = 0.0, 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    for inputs, labels in tqdm(loader, desc=epoch_desc, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)

    return epoch_loss, epoch_acc, epoch_time


def validate_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        epoch_desc: str = "Validação",  # <- Novo parâmetro adicionado aqui
) -> Tuple[float, float, List[int], List[int], np.ndarray]:
    """Executa uma época de validação sem alterar pesos."""
    model.eval()
    running_loss, correct = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        # <- tqdm adicionado aqui com leave=False para manter o terminal limpo
        for inputs, labels in tqdm(loader, desc=epoch_desc, leave=False):
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
        seed: int,
        attention_based_models: FrozenSet[str],
) -> None:
    """Roteia e gera as visualizações de interpretabilidade e espaço latente."""
    plot_latent_space(
        model, model_name, test_loader, device, plot_dir, class_names, seed,
    )

    if model_name in attention_based_models:
        print(
            f"  🧠 Arquitetura Transformer detectada para {model_name}. "
            f"Gerando mapas de atenção...",
        )
        generate_transformer_samples(
            model=model, model_name=model_name, test_loader=test_loader,
            device=device, plot_dir=plot_dir, class_names=class_names,
            samples_per_class=5,
        )
    else:
        print(
            f"  📷 Arquitetura CNN detectada para {model_name}. Gerando "
            f"Grad-CAM...",
        )
        original_grad_states = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad = True

        generate_gradcam_samples(
            model=model, model_name=model_name, test_loader=test_loader,
            device=device, plot_dir=plot_dir, class_names=class_names,
            samples_per_class=5,
        )

        for param, requires_grad in zip(
                model.parameters(), original_grad_states,
        ):
            param.requires_grad = requires_grad


def _setup_training_components(
        model_name: str,
        num_classes: int,
        train_loader: DataLoader,
        context: BenchmarkContext,
        device: torch.device,
) -> Tuple[nn.Module, nn.Module, optim.Optimizer, lr_scheduler.LRScheduler]:
    """[DRY Helper] Configura Modelo, Loss, Otimizador e Scheduler."""
    model = build_model(
        model_name, num_classes, pretrained=False,
        results_dir=str(context.results_dir),
        radimagenet_weights_url=context.radimagenet_weights_url,
    ).to(device)

    # Extração unificada de targets (suporta tanto .targets quanto .samples)
    targets = (
        train_loader.dataset.targets
        if hasattr(train_loader.dataset, "targets")
        else [t[1] for t in train_loader.dataset.samples]
    )

    weights = build_class_weights(targets, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=context.learning_rate,
        weight_decay=context.weight_decay,
        betas=(0.9, 0.999),
    )

    # Usa a paciência do contexto para ambos os fluxos
    patience_sched = getattr(context, "patience_scheduler", 3)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=context.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=context.num_epochs,
        pct_start=0.3,  # 30% do treino gastando no aquecimento agressivo
        div_factor=25.0,  # LR inicial = max_lr / 25
        final_div_factor=1000.0
        # LR final = LR inicial / 1000 (assentamento fino)
    )

    return model, criterion, optimizer, scheduler


def _execute_training_loop(
        model: nn.Module,
        loaders: Tuple[DataLoader, DataLoader],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler.LRScheduler,
        context: BenchmarkContext,
        device: torch.device,
        best_weights_path: Path,
        prefix_desc: str = "",
) -> Tuple[EarlyStoppingState, dict, float]:
    """[DRY Helper] Executa o loop de épocas completo com validação e Early Stopping."""
    train_loader, val_loader = loaders
    es_state = init_early_stopping(
        context.patience, context.min_delta, str(best_weights_path)
        )
    history = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
        "f1_per_class": []
    }
    train_time_acumulado = 0.0

    for epoch in range(context.num_epochs):
        ep_label = f"E{epoch + 1:02}"

        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            f"{prefix_desc}Treino {ep_label}".strip(),
            scheduler=scheduler
        )
        train_time_acumulado += epoch_time

        val_loss, val_acc, val_preds, val_labels, val_probs = validate_one_epoch(
            model, val_loader, criterion, device,
            f"{prefix_desc}Val {ep_label}".strip(),
        )

        epoch_f1_classes = f1_score(
            val_labels, val_preds, average=None, zero_division=0
            )
        epoch_f1_macro = epoch_f1_classes.mean()


        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["f1_per_class"].append(epoch_f1_classes.tolist())

        # Pega a taxa de aprendizado atual para exibir no log do terminal
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{ep_label} | LR: {current_lr:.6f} | Val Loss: {val_loss:.4f} | Val F1-Macro: {epoch_f1_macro:.4f}"
            )

        epoch_data = {
            "y_true": val_labels, "y_probs": val_probs, "y_preds": val_preds
        }
        es_state = step_early_stopping(
            es_state, epoch_f1_macro, model, epoch_data
            )

        if es_state["triggered"]:
            print(
                f"\n  Early stopping na época {epoch + 1} — melhor F1 Val: {es_state['best_score']:.4f}"
                )
            break

    return es_state, history, train_time_acumulado


def _load_best_and_cleanup(model: nn.Module, path: Path, device: torch.device) -> Tuple[nn.Module, int]:
    """[DRY Helper] Carrega o melhor modelo salvo, conta parâmetros e limpa a VRAM."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, num_params


def _free_gpu_memory(model: nn.Module) -> None:
    """[DRY Helper] Limpeza agressiva do Garbage Collector e PyTorch Cache."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- 3. Orquestrador Principal ---

def train_model_pipeline(
        model_name: str,
        context: BenchmarkContext,
        loaders: Tuple[DataLoader, DataLoader, DataLoader],
        class_names: List[str],
) -> dict:
    """Orquestrador declarativo do ciclo de vida de treinamento padrão."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = loaders
    num_classes = len(class_names)

    batch_size = context.batch_size_overrides.get(
        model_name, context.batch_size
        )
    print(f"\n{'=' * 55}\n  PROCESSANDO: {model_name.upper()}\n{'=' * 55}")
    if batch_size != context.batch_size:
        print(f"  ⚠️  Batch reduzido para {batch_size} (limite de VRAM)")

    # 1. Setup
    model, criterion, optimizer, scheduler = _setup_training_components(
        model_name, num_classes, train_loader, context, device
    )

    model_dir = context.results_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = model_dir / "best.pth"

    # 2. Treino e Validação
    es_state, history, train_time = _execute_training_loop(
        model, (train_loader, val_loader), criterion, optimizer, scheduler,
        context, device, best_weights_path
    )

    # 3. Avaliação Pós-Treino
    model, num_params = _load_best_and_cleanup(model, best_weights_path, device)
    best_val_data = es_state["best_data"]

    generate_model_plots(
        model_name, context.plot_dir, class_names, history,
        best_val_data["y_true"], best_val_data["y_probs"],
        best_val_data["y_preds"],
    )

    print("\n  🔍 Extraindo métricas finais do conjunto de TESTE...")
    test_eval = evaluate_split(
        model, test_loader, device, num_classes, f"Teste {model_name}"
        )
    plot_confidence_distribution(
        model_name, context.plot_dir, test_eval["probs_matrix"],
        test_eval["preds"], test_eval["labels"]
    )

    print("  🔍 Extraindo métricas finais do conjunto de TREINO...")
    train_eval = evaluate_split(
        model, train_loader, device, num_classes, f"Treino Final {model_name}"
        )

    print(
        f"  🏆 TESTE  | Acc: {test_eval['accuracy']:.4f} | Prec: {test_eval['precision']:.4f} | "
        f"Rec: {test_eval['recall']:.4f} | F1: {test_eval['f1_macro']:.4f} | Spec: {test_eval['specificity']:.4f} | "
        f"MCC: {test_eval['mcc']:.4f} | AUC: {test_eval['auc_macro']:.4f} | MSE: {test_eval['mse']:.4f}"
    )

    generate_post_training_visualizations(
        model, model_name, test_loader, device, context.plot_dir, class_names,
        context.seed, context.attention_based_models
    )

    _free_gpu_memory(model)

    return {
        "Modelo": model_name,
        "Train_Accuracy": train_eval["accuracy"],
        "Train_Precision": train_eval["precision"],
        "Train_Recall": train_eval["recall"],
        "Train_F1-Macro": train_eval["f1_macro"],
        "Train_Specificity": train_eval["specificity"],
        "Train_MCC": train_eval["mcc"],
        "Train_AUC-Macro": train_eval["auc_macro"],
        "Train_MSE": train_eval["mse"],
        "Val_F1-Macro": es_state["best_score"],
        "Test_Accuracy": test_eval["accuracy"],
        "Test_Precision": test_eval["precision"],
        "Test_Recall": test_eval["recall"],
        "Test_F1-Macro": test_eval["f1_macro"],
        "Test_Specificity": test_eval["specificity"],
        "Test_MCC": test_eval["mcc"],
        "Test_AUC-Macro": test_eval["auc_macro"],
        "Test_MSE": test_eval["mse"],
        "Train_Time_s": train_time,
        "Inference_Time_s": test_eval["time_s"],
        "Inference_ms_per_img": test_eval["time_ms_per_img"],
        "Parâmetros": num_params,
        "Batch": batch_size,
        "test_preds": test_eval["preds"],
        "test_labels": test_eval["labels"],
    }


def train_kfold_fold(
        model_name: str,
        context: BenchmarkContext,
        loaders: Tuple[DataLoader, DataLoader],
        class_names: List[str],
        fold_idx: int,
) -> dict:
    """Orquestrador especializado e ultraleve para um único Fold no K-Fold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = loaders
    num_classes = len(class_names)

    batch_size = context.batch_size_overrides.get(
        model_name, context.batch_size
        )
    print(f"\n{'=' * 20} FOLD {fold_idx} : {model_name.upper()} {'=' * 20}")

    # 1. Setup
    model, criterion, optimizer, scheduler = _setup_training_components(
        model_name, num_classes, train_loader, context, device
    )

    model_dir = context.results_dir / model_name / f"fold_{fold_idx}"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = model_dir / f"best_{model_name}_f{fold_idx}.pth"

    # 2. Treino e Validação (Com prefixo F1, F2... nos logs do tqdm)
    es_state, _, train_time = _execute_training_loop(
        model, (train_loader, val_loader), criterion, optimizer, scheduler,
        context, device, best_weights_path, prefix_desc=f"F{fold_idx} "
    )

    # 3. Avaliação Out-Of-Fold
    model, num_params = _load_best_and_cleanup(model, best_weights_path, device)

    print("  🔍 Extraindo métricas finais Out-Of-Fold (Validação)...")
    oof_eval = evaluate_split(
        model, val_loader, device, num_classes, f"OOF F{fold_idx} {model_name}",
    )

    print(
        f"  🏆 FOLD {fold_idx} OOF | Acc: {oof_eval['accuracy']:.4f} | "
        f"F1: {oof_eval['f1_macro']:.4f} | AUC: {oof_eval['auc_macro']:.4f}"
    )

    _free_gpu_memory(model)

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
        "Train_Time_s": train_time,
        "Inference_ms_per_img": oof_eval["time_ms_per_img"],
        "Parâmetros": num_params,
        "oof_preds": oof_eval["preds"],
        "oof_labels": oof_eval["labels"],
    }