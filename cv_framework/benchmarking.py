"""Pipeline modular do benchmark principal (Paradigma Funcional)."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import umap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from statsmodels.stats.contingency_tables import mcnemar
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BENCHMARK_BATCH_SIZE_OVERRIDES,
    BENCHMARK_DATASET_ROOT,
    BENCHMARK_MODELS,
    BENCHMARK_PLOT_DIR,
    BENCHMARK_RESULTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_WORKERS,
    DEFAULT_IMAGE_FOLDER_WORKERS,
    DEFAULT_LR,
    DEFAULT_SEED,
    RADIMAGENET_WEIGHTS_URL,
)
from cv_framework.data import build_class_weights, build_imagefolder_datasets
from cv_framework.models import build_model
from cv_framework.reproducibility import set_seed
from cv_framework.training import init_early_stopping, step_early_stopping
from cv_framework.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform, build_train_transform


@dataclass(frozen=True)
class BenchmarkContext:
    """Agrupa os recursos imutáveis de configuração do benchmark."""
    root_dir: Path = BENCHMARK_DATASET_ROOT
    plot_dir: Path = BENCHMARK_PLOT_DIR
    results_dir: Path = BENCHMARK_RESULTS_DIR
    batch_size: int = DEFAULT_BATCH_SIZE
    num_epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LR
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    train_workers: int = DEFAULT_IMAGE_FOLDER_WORKERS
    eval_workers: int = DEFAULT_EVAL_WORKERS
    seed: int = DEFAULT_SEED


# --- 1. Funções Puras de Preparação e Métricas ---

def setup_data_loaders(context: BenchmarkContext, batch_size: int) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], List[str]]:
    """
    Configura os dataloaders garantindo que o multiprocessamento rode de forma eficiente
    em ambientes Linux (como Ubuntu ou Manjaro), evitando gargalos no carregamento das imagens.
    """
    transform_train = build_train_transform()
    transform_eval = build_eval_transform()

    ds_train, ds_val, ds_test = build_imagefolder_datasets(
        context.root_dir,
        train_transform=transform_train,
        eval_transform=transform_eval,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=context.train_workers)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=context.eval_workers)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=context.eval_workers)

    return (train_loader, val_loader, test_loader), ds_train.classes


def calculate_auc(labels: List[int], probs_matrix: np.ndarray, num_classes: int) -> float:
    """Calcula a métrica AUC-ROC baseada no número de classes."""
    if num_classes == 2:
        return roc_auc_score(labels, probs_matrix[:, 1])
    return roc_auc_score(labels, probs_matrix, multi_class="ovr", average="macro")


# --- 2. Funções de Visualização e Relatórios ---

def generate_model_plots(
    model_name: str,
    plot_dir: Path,
    class_names: List[str],
    history: dict,
    y_true: List[int],
    y_probs_matrix: np.ndarray,
    y_preds: List[int]
) -> float:
    """Gera gráficos estáticos de treino e validação para um modelo específico."""
    out = plot_dir / model_name
    out.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    num_classes = len(class_names)

    # Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], label="Treino")
    ax1.plot(epochs, history["val_loss"], label="Validação")
    ax1.set_title(f"Loss — {model_name}")
    ax1.set_xlabel("Época")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Treino")
    ax2.plot(epochs, history["val_acc"], label="Validação")
    ax2.set_title(f"Acurácia — {model_name}")
    ax2.set_xlabel("Época")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png")
    plt.close(fig)

    # F1 Per Class
    f1_matrix = np.array(history["f1_per_class"])
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("tab10", n_colors=num_classes)
    for i, class_name in enumerate(class_names):
        ax.plot(epochs, f1_matrix[:, i], label=class_name, color=colors[i % len(colors)], linewidth=2, marker="o", markersize=4)
    ax.plot(epochs, f1_matrix.mean(axis=1), label="macro (média)", color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_title(f"F1 Validação por classe — {model_name}")
    ax.set_xlabel("Época")
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "f1_per_class.png")
    plt.close(fig)

    # ROC AUC
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_matrix[:, i])
        roc_auc_i = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc_i:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"Curvas ROC Validação (OvR) — {model_name}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_auc_curve_val.png")
    plt.close(fig)

    # Confusion Matrix
    macro_auc = calculate_auc(y_true, y_probs_matrix, num_classes)
    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 1.4), max(5, num_classes * 1.2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Matriz de Confusão Validação — {model_name}")
    ax.set_xlabel("Previsão")
    ax.set_ylabel("Rótulo real")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix_val.png")
    plt.close(fig)

    return macro_auc


def plot_latent_space(model: nn.Module, model_name: str, test_loader: DataLoader, device: torch.device, plot_dir: Path, class_names: List[str], seed: int) -> None:
    """Gera a projeção UMAP do espaço latente."""
    print(f"  🌌 Gerando projeção do Espaço Latente (UMAP) para {model_name}...")
    model.eval()
    features, labels_list = [], []
    out_dir = plot_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            try:
                feats = model.forward_features(inputs)
                if feats.ndim == 4:
                    feats = feats.mean(dim=[-2, -1])
                elif feats.ndim == 3:
                    feats = feats[:, 0]
            except Exception:
                feats = model(inputs)

            features.append(feats.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    features_np = np.vstack(features)
    labels_np = np.array(labels_list)

    reducer = umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.1)
    try:
        embedding = reducer.fit_transform(features_np)
    except Exception as exc:
        print(f"  ⚠️ Erro no UMAP para {model_name}: {exc}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels_np, cmap="coolwarm", alpha=0.7, s=50, edgecolors="k")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, class_names, title="Classes")
    ax.set_title(f"Espaço Latente (UMAP) — {model_name}")
    ax.set_xlabel("UMAP Dimensão 1")
    ax.set_ylabel("UMAP Dimensão 2")
    fig.tight_layout()
    fig.savefig(out_dir / "umap_latent_space.png", dpi=300)
    plt.close(fig)


def get_target_layer_for_cam(model: nn.Module, model_name: str) -> Optional[list]:
    if "mobilenetv3" in model_name and hasattr(model, "blocks"):
        return [model.blocks[-1]]
    if "resnet" in model_name and hasattr(model, "layer4"):
        return [model.layer4[-1]]
    if "densenet" in model_name and hasattr(model, "features") and hasattr(model.features, "norm5"):
        return [model.features.norm5]
    if "efficientnet" in model_name and hasattr(model, "blocks"):
        return [model.blocks[-1]]

    for _, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return [module]
    return None


def generate_gradcam_samples(model: nn.Module, model_name: str, test_loader: DataLoader, device: torch.device, plot_dir: Path, class_names: List[str], num_samples: int = 5) -> None:
    """Gera visualizações do Grad-CAM para amostras do conjunto de teste."""
    target_layers = get_target_layer_for_cam(model, model_name)
    if not target_layers:
        print(f"  ⚠️  Grad-CAM pulado para {model_name}: não foi possível identificar a target layer de forma automática.")
        return

    out_dir = plot_dir / model_name / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
    except Exception as exc:
        print(f"  ⚠️  Erro ao inicializar Grad-CAM para {model_name}: {exc}")
        return

    model.eval()
    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    n = min(num_samples, inputs.size(0))

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    for i in range(n):
        input_tensor = inputs[i].unsqueeze(0)
        real_label = labels[i].item()

        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()

        cam_targets = [ClassifierOutputTarget(pred_class)]
        try:
            grayscale_cam = cam(input_tensor=input_tensor, targets=cam_targets)[0, :]
        except Exception as exc:
            print(f"  ⚠️  Erro ao gerar Grad-CAM para amostra {i + 1}: {exc}")
            continue

        img_np = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img_np = np.asarray(np.clip(std * img_np + mean, 0, 1), dtype=np.float32)
        grayscale_cam = np.asarray(grayscale_cam, dtype=np.float32)

        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_np)
        ax1.set_title(f"Original (Real: {class_names[real_label]})")
        ax1.axis("off")

        color_text = "darkgreen" if real_label == pred_class else "darkred"
        ax2.imshow(cam_image)
        ax2.set_title(f"Grad-CAM (Pred: {class_names[pred_class]})", color=color_text, fontweight="bold")
        ax2.axis("off")

        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i + 1}_cam.png", bbox_inches="tight")
        plt.close(fig)


def update_results_csv(df_new: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Atualiza o arquivo CSV de forma funcional."""
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        trained_models = df_new["Modelo"].tolist()
        df_old = df_old[~df_old["Modelo"].isin(trained_models)]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final = df_final.sort_values("Test_F1-Macro", ascending=False)
    df_final.to_csv(csv_path, index=False)
    return df_final


# --- 3. Funções Puras de Inferência e Avaliação ---

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

    return {
        "preds": preds,
        "labels": labels,
        "probs_matrix": probs_matrix,
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "auc_macro": calculate_auc(labels, probs_matrix, num_classes),
        "time_s": total_time,
        "time_ms_per_img": (total_time / dataset_size) * 1000,
    }


# --- 4. Pipeline Central de Treinamento ---

def train_model_pipeline(model_name: str, context: BenchmarkContext, loaders: Tuple[DataLoader, DataLoader, DataLoader], class_names: List[str]) -> dict:
    """Orquestra o loop de treino utilizando fluxo de estado funcional e dados imutáveis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = loaders

    batch_size = BENCHMARK_BATCH_SIZE_OVERRIDES.get(model_name, context.batch_size)
    print(f"\n{'=' * 55}\n  PROCESSANDO: {model_name.upper()}\n{'=' * 55}")
    if batch_size != context.batch_size:
        print(f"  ⚠️  Batch reduzido para {batch_size} (limite de VRAM)")

    num_classes = len(class_names)
    model = build_model(
        model_name,
        num_classes,
        pretrained=False,
        results_dir=str(context.results_dir),
        radimagenet_weights_url=RADIMAGENET_WEIGHTS_URL
    ).to(device)

    weights = build_class_weights([t[1] for t in train_loader.dataset.samples], num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=context.learning_rate)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=context.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=context.num_epochs,
        pct_start=0.3,
    )

    model_dir = context.results_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = model_dir / "best.pth"

    # Inicializa estado funcional do Early Stopping
    es_state = init_early_stopping(context.patience, context.min_delta, str(best_weights_path))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "f1_per_class": []}
    train_time_acumulado = 0.0

    for epoch in range(context.num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start_train_epoch = time.time()

        for inputs, labels in tqdm(train_loader, desc=f"Treino E{epoch + 1:02}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        if torch.cuda.is_available(): torch.cuda.synchronize()
        train_time_acumulado += time.time() - start_train_epoch

        # Validação da época
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                val_correct += (preds == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_f1_classes = f1_score(all_labels, all_preds, average=None, zero_division=0)
        epoch_f1_macro = epoch_f1_classes.mean()

        history["train_loss"].append(train_loss / len(train_loader.dataset))
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(train_correct / len(train_loader.dataset))
        history["val_acc"].append(val_correct / len(val_loader.dataset))
        history["f1_per_class"].append(epoch_f1_classes.tolist())

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"E{epoch + 1:02} | LR: {current_lr:.6f} | Val Loss: {epoch_val_loss:.4f} | Val F1-Macro: {epoch_f1_macro:.4f}")

        # Atualização Funcional do Early Stopping
        epoch_data = {"y_true": all_labels, "y_probs": np.vstack(all_probs), "y_preds": all_preds}
        es_state = step_early_stopping(es_state, epoch_f1_macro, model, epoch_data)

        if es_state["triggered"]:
            print(f"\n  Early stopping na época {epoch + 1} — melhor F1 Val: {es_state['best_score']:.4f}")
            break

    # Carrega os melhores pesos para a avaliação final
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best_val_data = es_state["best_data"]
    generate_model_plots(model_name, context.plot_dir, class_names, history, best_val_data["y_true"], best_val_data["y_probs"], best_val_data["y_preds"])

    print("\n  🔍 Extraindo métricas finais do conjunto de TESTE...")
    test_eval = evaluate_split(model, test_loader, device, num_classes, f"Teste {model_name}")

    plot_confidence_distribution(model_name, context.plot_dir, test_eval["probs_matrix"], test_eval["preds"],
                                 test_eval["labels"])

    print("  🔍 Extraindo métricas finais do conjunto de TREINO...")
    train_eval = evaluate_split(model, train_loader, device, num_classes, f"Treino Final {model_name}")

    print(f"  🏆 TESTE  | F1-Macro: {test_eval['f1_macro']:.4f} | AUC-Macro: {test_eval['auc_macro']:.4f}")
    print(f"  🏆 TREINO | F1-Macro: {train_eval['f1_macro']:.4f} | AUC-Macro: {train_eval['auc_macro']:.4f}")

    # Visualizações Extras
    plot_latent_space(model, model_name, test_loader, device, context.plot_dir, class_names, context.seed)

    print("  🎨 Gerando heatmaps do Grad-CAM...")
    for param in model.parameters(): param.requires_grad = True
    generate_gradcam_samples(model, model_name, test_loader, device, context.plot_dir, class_names, num_samples=5)
    for param in model.parameters(): param.requires_grad = False

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "Modelo": model_name,
        "Train_F1-Macro": train_eval["f1_macro"],
        "Train_AUC-Macro": train_eval["auc_macro"],
        "Val_F1-Macro": es_state["best_score"],
        "Test_F1-Macro": test_eval["f1_macro"],
        "Test_AUC-Macro": test_eval["auc_macro"],
        "Train_Time_s": train_time_acumulado,
        "Inference_Time_s": test_eval["time_s"],
        "Inference_ms_per_img": test_eval["time_ms_per_img"],
        "Parâmetros": num_params,
        "Batch": batch_size,
        "test_preds": test_eval["preds"],
        "test_labels": test_eval["labels"],
    }


def analyze_inter_model_agreement(results: List[Dict], plot_dir: Path) -> None:
    """
    Realiza a análise de concordância inter-modelo, Teste de McNemar
    e análise de falhas sistêmicas usando funções puras baseadas nos resultados.
    """
    print("\n" + "=" * 85)
    print("GERANDO ANÁLISE INTER-MODELO (Acordo e Significância)")
    print("=" * 85)

    if len(results) < 2:
        print("  ⚠️ Necessário pelo menos 2 modelos treinados para gerar a análise inter-modelo.")
        return

    # Extrai os dados do dicionário de resultados
    preds_dict = {res["Modelo"]: np.array(res["test_preds"]) for res in results}
    true_labels = np.array(results[0]["test_labels"])  # Assume que todos testaram no mesmo dataset
    model_names = list(preds_dict.keys())

    # --- 1. Matriz de Acordo ---
    agreement = np.zeros((len(model_names), len(model_names)))
    for i, mod1 in enumerate(model_names):
        for j, mod2 in enumerate(model_names):
            agreement[i, j] = np.mean(preds_dict[mod1] == preds_dict[mod2])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(agreement, annot=True, fmt=".2%", cmap="Purples", xticklabels=model_names, yticklabels=model_names,
                ax=ax)
    ax.set_title("Matriz de Acordo Inter-Modelo (Predições Idênticas)")
    fig.tight_layout()
    fig.savefig(plot_dir / "inter_model_agreement.png", dpi=300)
    plt.close(fig)

    # --- 2. Teste Estatístico de McNemar ---
    # Identifica o melhor modelo com base no F1-Macro
    top_model = max(results, key=lambda x: x["Test_F1-Macro"])["Modelo"]
    print("\n📊 Teste Estatístico de McNemar (Top 1 vs Outros):")

    p1 = preds_dict[top_model]

    for mod2 in model_names:
        if mod2 == top_model:
            continue

        p2 = preds_dict[mod2]

        # Tabela de Contingência
        ambos_acertam = np.sum((p1 == true_labels) & (p2 == true_labels))
        p1_acerta_p2_erra = np.sum((p1 == true_labels) & (p2 != true_labels))
        p1_erra_p2_acerta = np.sum((p1 != true_labels) & (p2 == true_labels))
        ambos_erram = np.sum((p1 != true_labels) & (p2 != true_labels))

        contingency_table = [[ambos_acertam, p1_acerta_p2_erra],
                             [p1_erra_p2_acerta, ambos_erram]]

        result = mcnemar(contingency_table, exact=True)

        if result.pvalue < 0.05:
            print(f"  ✅ {top_model} vs {mod2}: Diferença SIGNIFICATIVA (p={result.pvalue:.4f})")
        else:
            print(f"  ⚖️  {top_model} vs {mod2}: Empate Estatístico (p={result.pvalue:.4f})")

    # --- 3. Análise de Falha Genuína ---
    all_preds = np.array([preds_dict[m] for m in model_names])
    acertos = all_preds == true_labels
    erros_totais_idx = np.where(np.sum(acertos, axis=0) == 0)[0]

    print("\n🔥 Análise de Falha Genuína:")
    print(f"  Ocorreram {len(erros_totais_idx)} imagens de teste que TODOS os modelos erraram.")
    if len(erros_totais_idx) > 0:
        print("  Índices dessas imagens no Dataset de Teste:", erros_totais_idx)


def generate_global_reports(df_final: pd.DataFrame, plot_dir: Path) -> None:
    """Gera os gráficos comparativos globais entre todos os modelos do benchmark."""
    print("  📊 Gerando relatórios globais...")

    # 1. Ranking Global (F1 Teste)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=df_final, x="Test_F1-Macro", y="Modelo", hue="Modelo", palette="viridis", legend=False, ax=ax)
    ax.set_title("Ranking Global — F1-Score Macro (CONJUNTO DE TESTE)")
    fig.tight_layout()
    fig.savefig(plot_dir / "ranking_f1_test.png", dpi=300)
    plt.close(fig)

    if "Params_M" in df_final.columns:
        # 2. Eficiência (Parâmetros vs F1)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=df_final, x="Params_M", y="Test_F1-Macro", hue="Modelo", s=200, palette="tab20",
                        legend=False, ax=ax)
        for _, row in df_final.iterrows():
            ax.text(row["Params_M"] + 0.5, row["Test_F1-Macro"], row["Modelo"], fontsize=9)
        ax.set_title("Eficiência (Teste) — Parâmetros vs F1")
        ax.set_xlabel("Parâmetros (M)")
        fig.tight_layout()
        fig.savefig(plot_dir / "eficiencia_test.png", dpi=300)
        plt.close(fig)

        # 3. Velocidade vs Performance
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=df_final, x="Inference_ms_per_img", y="Test_F1-Macro", size="Params_M",
            sizes=(50, 800), hue="Modelo", alpha=0.7, palette="tab20", legend=False, ax=ax
        )
        for _, row in df_final.iterrows():
            ax.text(row["Inference_ms_per_img"] * 1.02, row["Test_F1-Macro"], row["Modelo"], fontsize=9)
        ax.set_title("Trade-off de Produção: Velocidade vs F1 (Bolhas = Tamanho do Modelo)")
        ax.set_xlabel("Tempo de Inferência por Imagem (ms)")
        ax.set_ylabel("F1-Score Macro (Teste)")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(plot_dir / "velocidade_vs_performance.png", dpi=300)
        plt.close(fig)

    # 4. Análise de Overfitting
    df_melted = df_final.melt(
        id_vars=["Modelo"], value_vars=["Train_F1-Macro", "Test_F1-Macro"],
        var_name="Conjunto", value_name="F1-Score"
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(data=df_melted, x="F1-Score", y="Modelo", hue="Conjunto", palette="Set1", ax=ax)
    ax.set_title("Análise de Overfitting (Treino vs Teste)")
    ax.set_xlabel("F1-Score Macro")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(plot_dir / "overfitting_analysis.png", dpi=300)
    plt.close(fig)

    # 5. Tempo de Treinamento
    df_final_copy = df_final.copy()
    df_final_copy["Train_Time_min"] = df_final_copy["Train_Time_s"] / 60.0
    df_final_sorted_time = df_final_copy.sort_values("Train_Time_min", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=df_final_sorted_time, x="Train_Time_min", y="Modelo", palette="rocket", ax=ax)
    ax.set_title("Custo de Treinamento: Tempo Acumulado na GPU")
    ax.set_xlabel("Tempo de Treino (Minutos)")
    fig.tight_layout()
    fig.savefig(plot_dir / "tempo_treinamento.png", dpi=300)
    plt.close(fig)


def plot_confidence_distribution(
        model_name: str,
        plot_dir: Path,
        test_probs_matrix: np.ndarray,
        test_preds: List[int],
        test_labels: List[int]
) -> None:
    """Gera o histograma de confiança do modelo no conjunto de teste."""
    test_confidences = np.max(test_probs_matrix, axis=1)
    correct_mask = np.array(test_preds) == np.array(test_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(test_confidences[correct_mask], bins=20, color="green", label="Corretos", kde=True, alpha=0.6, ax=ax)
    sns.histplot(test_confidences[~correct_mask], bins=20, color="red", label="Incorretos", kde=True, alpha=0.6, ax=ax)
    ax.set_title(f"Distribuição de Confiança (Teste) — {model_name}")
    ax.set_xlabel("Confiança (Probabilidade da Classe Majoritária)")
    ax.set_ylabel("Frequência de Imagens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / model_name / "confidence_distribution.png")
    plt.close(fig)

def run_benchmark(models: Iterable[str] = BENCHMARK_MODELS, context: Optional[BenchmarkContext] = None) -> pd.DataFrame:
    """Entry point principal de execução do benchmark."""
    ctx = context or BenchmarkContext()
    ctx.plot_dir.mkdir(parents=True, exist_ok=True)
    ctx.results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(ctx.seed)
    loaders, class_names = setup_data_loaders(ctx, ctx.batch_size)
    print(f"Classes detectadas ({len(class_names)}): {class_names}")

    results = []
    for model_name in models:
        try:
            result = train_model_pipeline(model_name, ctx, loaders, class_names)
            results.append(result)
        except Exception as exc:
            print(f"\n❌ ERRO CRÍTICO ao treinar o modelo '{model_name}': {exc}")
            print(f"PULANDO '{model_name}' e limpando a memória para o próximo modelo...\n")
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not results:
        print("\n⚠️ Nenhum modelo foi treinado com sucesso. Encerrando o script.")
        return pd.DataFrame()

    analyze_inter_model_agreement(results, ctx.plot_dir)

    df_new = pd.DataFrame(results)
    df_new["Params_M"] = df_new["Parâmetros"] / 1e6

    # Limpeza para CSV
    df_clean = df_new.drop(columns=["test_preds", "test_labels"], errors='ignore')
    csv_path = ctx.plot_dir / "benchmark_results_test_train.csv"
    df_final = update_results_csv(df_clean, csv_path)

    print("\n" + "=" * 85)
    print("RANKING FINAL ATUALIZADO (MÉTRICAS DE TREINO, VALIDAÇÃO E TESTE)")
    print("=" * 85)
    colunas_exibicao = ["Modelo", "Train_F1-Macro", "Val_F1-Macro", "Test_F1-Macro", "Test_AUC-Macro"]
    print(df_final[colunas_exibicao].to_string(index=False))

    generate_global_reports(df_final, ctx.plot_dir)

    return df_final