"""Pipeline modular do benchmark principal."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
from cv_framework.models import build_model as build_shared_model
from cv_framework.reproducibility import set_seed
from cv_framework.training import EarlyStopping
from cv_framework.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform, build_train_transform


@dataclass(frozen=True)
class BenchmarkContext:
    """Agrupa os recursos compartilhados do benchmark."""

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


class BenchmarkRunner:
    """Executa treino, avaliação e relatórios do benchmark."""

    def __init__(self, context: BenchmarkContext | None = None):
        self.context = context or BenchmarkContext()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.context.seed)
        self.plot_dir = Path(self.context.plot_dir)
        self.results_dir = Path(self.context.results_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.transform_train = build_train_transform()
        self.transform_eval = build_eval_transform()
        self.dataset_train, self.dataset_val, self.dataset_test = build_imagefolder_datasets(
            self.context.root_dir,
            train_transform=self.transform_train,
            eval_transform=self.transform_eval,
        )
        self.class_names = self.dataset_train.classes
        self.num_classes = len(self.class_names)
        print(f"Classes detectadas ({self.num_classes}): {self.class_names}")

    def make_loaders(self, batch_size: int):
        train = DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.context.train_workers,
        )
        val = DataLoader(
            self.dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.context.eval_workers,
        )
        test = DataLoader(
            self.dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.context.eval_workers,
        )
        return train, val, test

    def build_criterion(self) -> nn.CrossEntropyLoss:
        weights = build_class_weights(self.dataset_train.targets, self.num_classes, device=self.device)
        return nn.CrossEntropyLoss(weight=weights)

    def build_model(self, model_name: str) -> nn.Module:
        return build_shared_model(
            model_name,
            self.num_classes,
            pretrained=False,
            results_dir=str(self.results_dir),
            radimagenet_weights_url=RADIMAGENET_WEIGHTS_URL,
        )

    def _score_auc(self, labels: list[int], probs_matrix: np.ndarray) -> float:
        if self.num_classes == 2:
            return roc_auc_score(labels, probs_matrix[:, 1])
        return roc_auc_score(labels, probs_matrix, multi_class="ovr", average="macro")

    def save_plots(self, model_name: str, history: dict, y_true: list, y_probs_matrix: np.ndarray, y_preds: list) -> float:
        out = self.plot_dir / model_name
        out.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)

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

        f1_matrix = np.array(history["f1_per_class"])
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette("tab10", n_colors=len(self.class_names))
        for i, class_name in enumerate(self.class_names):
            ax.plot(
                epochs,
                f1_matrix[:, i],
                label=class_name,
                color=colors[i % len(colors)],
                linewidth=2,
                marker="o",
                markersize=4,
            )
        ax.plot(
            epochs,
            f1_matrix.mean(axis=1),
            label="macro (média)",
            color="black",
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
        )
        ax.set_title(f"F1 Validação por classe — {model_name}")
        ax.set_xlabel("Época")
        ax.set_ylabel("F1-Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "f1_per_class.png")
        plt.close(fig)

        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        if self.num_classes == 2:
            y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, class_name in enumerate(self.class_names):
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

        macro_auc = self._score_auc(y_true, y_probs_matrix)

        cm = confusion_matrix(y_true, y_preds)
        fig, ax = plt.subplots(figsize=(max(6, self.num_classes * 1.4), max(5, self.num_classes * 1.2)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_title(f"Matriz de Confusão Validação — {model_name}")
        ax.set_xlabel("Previsão")
        ax.set_ylabel("Rótulo real")
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix_val.png")
        plt.close(fig)

        return macro_auc

    def plot_latent_space(self, model: nn.Module, model_name: str, test_loader: DataLoader) -> None:
        print(f"  🌌 Gerando projeção do Espaço Latente (UMAP) para {model_name}...")
        model.eval()
        features, labels_list = [], []
        out_dir = self.plot_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
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

        reducer = umap.UMAP(random_state=self.context.seed, n_neighbors=15, min_dist=0.1)
        try:
            embedding = reducer.fit_transform(features_np)
        except Exception as exc:
            print(f"  ⚠️ Erro no UMAP para {model_name}: {exc}")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels_np, cmap="coolwarm", alpha=0.7, s=50, edgecolors="k")
        handles, _ = scatter.legend_elements()
        ax.legend(handles, self.class_names, title="Classes")
        ax.set_title(f"Espaço Latente (UMAP) — {model_name}")
        ax.set_xlabel("UMAP Dimensão 1")
        ax.set_ylabel("UMAP Dimensão 2")
        fig.tight_layout()
        fig.savefig(out_dir / "umap_latent_space.png", dpi=300)
        plt.close(fig)

    def get_target_layer_for_cam(self, model: nn.Module, model_name: str):
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

    def save_gradcam_samples(self, model: nn.Module, model_name: str, test_loader: DataLoader, num_samples: int = 5) -> None:
        target_layers = self.get_target_layer_for_cam(model, model_name)
        if not target_layers:
            print(f"  ⚠️  Grad-CAM pulado para {model_name}: não foi possível identificar a target layer de forma automática.")
            return

        out_dir = self.plot_dir / model_name / "gradcam"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            cam = GradCAM(model=model, target_layers=target_layers)  # type: ignore[arg-type]
        except Exception as exc:
            print(f"  ⚠️  Erro ao inicializar Grad-CAM para {model_name}: {exc}")
            return

        model.eval()
        inputs, labels = next(iter(test_loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
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
                grayscale_cam = cam(input_tensor=input_tensor, targets=cam_targets)[0, :]  # type: ignore[arg-type]
            except Exception as exc:
                print(f"  ⚠️  Erro ao gerar Grad-CAM para amostra {i + 1}: {exc}")
                continue

            img_np = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
            img_np = np.asarray(np.clip(std * img_np + mean, 0, 1), dtype=np.float32)
            grayscale_cam = np.asarray(grayscale_cam, dtype=np.float32)

            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(img_np)
            ax1.set_title(f"Original (Real: {self.class_names[real_label]})")
            ax1.axis("off")

            color_text = "darkgreen" if real_label == pred_class else "darkred"
            ax2.imshow(cam_image)
            ax2.set_title(f"Grad-CAM (Pred: {self.class_names[pred_class]})", color=color_text, fontweight="bold")
            ax2.axis("off")

            fig.tight_layout()
            fig.savefig(out_dir / f"sample_{i + 1}_cam.png", bbox_inches="tight")
            plt.close(fig)

    def _run_inference(self, model: nn.Module, loader: DataLoader, desc: str) -> tuple[list[int], list[int], np.ndarray, float]:
        preds_list, labels_list, probs_list = [], [], []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=desc, leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                probs_list.append(probs.cpu().numpy())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - start
        probs_matrix = np.vstack(probs_list)
        return preds_list, labels_list, probs_matrix, total_time

    def _evaluate_split(self, model: nn.Module, loader: DataLoader, desc: str) -> dict:
        preds_list, labels_list, probs_matrix, total_time = self._run_inference(model, loader, desc)
        dataset_size = len(loader.dataset)  # type: ignore[arg-type]
        infer_ms_per_img = (total_time / dataset_size) * 1000
        f1_macro = f1_score(labels_list, preds_list, average="macro", zero_division=0)
        auc_macro = self._score_auc(labels_list, probs_matrix)
        return {
            "preds": preds_list,
            "labels": labels_list,
            "probs_matrix": probs_matrix,
            "f1_macro": f1_macro,
            "auc_macro": auc_macro,
            "time_s": total_time,
            "time_ms_per_img": infer_ms_per_img,
        }

    def train_model(self, model_name: str) -> dict:
        batch_size = BENCHMARK_BATCH_SIZE_OVERRIDES.get(model_name, self.context.batch_size)
        print(f"\n{'=' * 55}\n  PROCESSANDO: {model_name.upper()}\n{'=' * 55}")
        if batch_size != self.context.batch_size:
            print(f"  ⚠️  Batch reduzido para {batch_size} (limite de VRAM)")

        model_dir = self.results_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader, test_loader = self.make_loaders(batch_size)
        model = self.build_model(model_name).to(self.device)
        criterion = self.build_criterion()
        optimizer = optim.Adam(model.parameters(), lr=self.context.learning_rate)
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.context.learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=self.context.num_epochs,
            pct_start=0.3,
        )

        best_weights_path = model_dir / "best.pth"
        early_stopping = EarlyStopping(
            patience=self.context.patience,
            min_delta=self.context.min_delta,
            path=str(best_weights_path),
        )
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "f1_per_class": []}
        train_time_acumulado = 0.0

        for epoch in range(self.context.num_epochs):
            model.train()
            train_loss, train_correct = 0.0, 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_train_epoch = time.time()

            for inputs, labels in tqdm(train_loader, desc=f"Treino E{epoch + 1:02}", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            train_time_acumulado += time.time() - start_train_epoch

            model.eval()
            val_loss, val_correct = 0.0, 0
            all_preds, all_labels, all_probs = [], [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item() * inputs.size(0)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
                    val_correct += (preds == labels).sum().item()

            epoch_val_loss = val_loss / len(self.dataset_val)
            epoch_f1_classes = f1_score(all_labels, all_preds, average=None, zero_division=0)
            epoch_f1_macro = epoch_f1_classes.mean()

            history["train_loss"].append(train_loss / len(self.dataset_train))
            history["val_loss"].append(epoch_val_loss)
            history["train_acc"].append(train_correct / len(self.dataset_train))
            history["val_acc"].append(val_correct / len(self.dataset_val))
            history["f1_per_class"].append(epoch_f1_classes.tolist())

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"E{epoch + 1:02} | LR: {current_lr:.6f} | Val Loss: {epoch_val_loss:.4f} | Val F1-Macro: {epoch_f1_macro:.4f}"
            )

            epoch_data = {"y_true": all_labels, "y_probs": np.vstack(all_probs), "y_preds": all_preds}
            if early_stopping.step(epoch_f1_macro, model, epoch_data):
                print(f"\n  Early stopping na época {epoch + 1} — melhor F1 Val: {early_stopping.best_score:.4f}")
                break

        best_val = early_stopping.best_data
        self.save_plots(model_name, history, best_val["y_true"], best_val["y_probs"], best_val["y_preds"])
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.load_state_dict(torch.load(best_weights_path, map_location=self.device))
        model.eval()

        print("\n  🔍 Extraindo métricas finais do conjunto de TESTE...")
        test_eval = self._evaluate_split(model, test_loader, desc=f"Teste {model_name}")
        test_preds = test_eval["preds"]
        test_labels = test_eval["labels"]
        test_probs_matrix = test_eval["probs_matrix"]
        test_f1_macro = test_eval["f1_macro"]
        test_auc_macro = test_eval["auc_macro"]
        infer_ms_per_img = test_eval["time_ms_per_img"]
        total_infer_time = test_eval["time_s"]

        cm_test = confusion_matrix(test_labels, test_preds)
        fig, ax = plt.subplots(figsize=(max(6, self.num_classes * 1.4), max(5, self.num_classes * 1.2)))
        sns.heatmap(
            cm_test,
            annot=True,
            fmt="d",
            cmap="Greens",
            cbar=False,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_title(f"Matriz de Confusão TESTE — {model_name}")
        ax.set_xlabel("Previsão")
        ax.set_ylabel("Rótulo real")
        fig.tight_layout()
        fig.savefig(self.plot_dir / model_name / "confusion_matrix_test.png")
        plt.close(fig)

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
        fig.savefig(self.plot_dir / model_name / "confidence_distribution.png")
        plt.close(fig)

        print("  🎨 Gerando heatmaps do Grad-CAM...")
        for param in model.parameters():
            param.requires_grad = True
        self.save_gradcam_samples(model, model_name, test_loader, num_samples=5)
        for param in model.parameters():
            param.requires_grad = False

        print("  🔍 Extraindo métricas finais do conjunto de TREINO...")
        train_eval = self._evaluate_split(model, train_loader, desc=f"Treino Final {model_name}")
        train_preds = train_eval["preds"]
        train_labels = train_eval["labels"]
        train_probs_matrix = train_eval["probs_matrix"]
        train_f1_macro = train_eval["f1_macro"]
        train_auc_macro = train_eval["auc_macro"]

        print(f"  🏆 TESTE  | F1-Macro: {test_f1_macro:.4f} | AUC-Macro: {test_auc_macro:.4f}")
        print(f"  🏆 TREINO | F1-Macro: {train_f1_macro:.4f} | AUC-Macro: {train_auc_macro:.4f}")

        self.plot_latent_space(model, model_name, test_loader)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "Modelo": model_name,
            "Train_F1-Macro": train_f1_macro,
            "Train_AUC-Macro": train_auc_macro,
            "Val_F1-Macro": early_stopping.best_score,
            "Test_F1-Macro": test_f1_macro,
            "Test_AUC-Macro": test_auc_macro,
            "Train_Time_s": train_time_acumulado,
            "Inference_Time_s": total_infer_time,
            "Inference_ms_per_img": infer_ms_per_img,
            "Parâmetros": num_params,
            "Batch": batch_size,
            "test_preds": test_preds,
            "test_labels": test_labels,
        }

    def _update_results_csv(self, df_new: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
        if csv_path.exists():
            df_old = pd.read_csv(csv_path)  # type: ignore[call-arg]
            trained_models = df_new["Modelo"].tolist()
            df_old = df_old[~df_old["Modelo"].isin(trained_models)]
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new

        df_final = df_final.sort_values("Test_F1-Macro", ascending=False)
        df_final.to_csv(csv_path, index=False)
        return df_final

    def _plot_global_reports(self, df_final: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=df_final, x="Test_F1-Macro", y="Modelo", hue="Modelo", palette="viridis", legend=False, ax=ax)
        ax.set_title("Ranking Global — F1-Score Macro (CONJUNTO DE TESTE)")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "ranking_f1_test.png", dpi=300)
        plt.close(fig)

        # Garantir que Params_M exista antes de usar
        if "Params_M" not in df_final.columns and "Parâmetros" in df_final.columns:
            df_final = df_final.copy()
            df_final["Params_M"] = df_final["Parâmetros"] / 1e6

        if "Params_M" in df_final.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(data=df_final, x="Params_M", y="Test_F1-Macro", hue="Modelo", s=200, palette="tab20", legend=False, ax=ax)
            for _, row in df_final.iterrows():
                ax.text(row["Params_M"] + 0.5, row["Test_F1-Macro"], row["Modelo"], fontsize=9)
            ax.set_title("Eficiência (Teste) — Parâmetros vs F1")
            ax.set_xlabel("Parâmetros (M)")
            fig.tight_layout()
            fig.savefig(self.plot_dir / "eficiencia_test.png", dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(
                data=df_final,
                x="Inference_ms_per_img",
                y="Test_F1-Macro",
                size="Params_M",
                sizes=(50, 800),
                hue="Modelo",
                alpha=0.7,
                palette="tab20",
                legend=False,
                ax=ax,
            )
            for _, row in df_final.iterrows():
                ax.text(row["Inference_ms_per_img"] * 1.02, row["Test_F1-Macro"], row["Modelo"], fontsize=9)
            ax.set_title("Trade-off de Produção: Velocidade vs F1 (Bolhas = Tamanho do Modelo)")
            ax.set_xlabel("Tempo de Inferência por Imagem (ms)")
            ax.set_ylabel("F1-Score Macro (Teste)")
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            fig.savefig(self.plot_dir / "velocidade_vs_performance.png", dpi=300)
            plt.close(fig)

        df_melted = df_final.melt(
            id_vars=["Modelo"],
            value_vars=["Train_F1-Macro", "Test_F1-Macro"],
            var_name="Conjunto",
            value_name="F1-Score",
        )
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=df_melted, x="F1-Score", y="Modelo", hue="Conjunto", palette="Set1", ax=ax)
        ax.set_title("Análise de Overfitting (Treino vs Teste)")
        ax.set_xlabel("F1-Score Macro")
        ax.set_xlim(0, 1.05)
        fig.tight_layout()
        fig.savefig(self.plot_dir / "overfitting_analysis.png", dpi=300)
        plt.close(fig)

        df_final_copy = df_final.copy()
        df_final_copy["Train_Time_min"] = df_final_copy["Train_Time_s"] / 60.0
        df_final_sorted_time = df_final_copy.sort_values("Train_Time_min", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=df_final_sorted_time, x="Train_Time_min", y="Modelo", palette="rocket", ax=ax)
        ax.set_title("Custo de Treinamento: Tempo Acumulado na GPU")
        ax.set_xlabel("Tempo de Treino (Minutos)")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "tempo_treinamento.png", dpi=300)
        plt.close(fig)

    def _analyze_inter_model_agreement(self, df_new: pd.DataFrame, results: list[dict]) -> None:
        print("\n" + "=" * 85)
        print("GERANDO ANÁLISE INTER-MODELO (Acordo e Significância)")
        print("=" * 85)

        preds_dict = {res["Modelo"]: res["test_preds"] for res in results}
        true_labels = np.array(results[0]["test_labels"])
        model_names = list(preds_dict.keys())

        if len(model_names) < 2:
            return

        agreement = np.zeros((len(model_names), len(model_names)))
        for i, mod1 in enumerate(model_names):
            for j, mod2 in enumerate(model_names):
                agreement[i, j] = np.mean(np.array(preds_dict[mod1]) == np.array(preds_dict[mod2]))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(agreement, annot=True, fmt=".2%", cmap="Purples", xticklabels=model_names, yticklabels=model_names, ax=ax)
        ax.set_title("Matriz de Acordo Inter-Modelo (Predições Idênticas)")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "inter_model_agreement.png", dpi=300)
        plt.close(fig)

        print("\n📊 Teste Estatístico de McNemar (Top 1 vs Outros):")
        top_model = df_new.sort_values("Test_F1-Macro", ascending=False).iloc[0]["Modelo"]
        p1 = np.array(preds_dict[top_model])
        for mod2 in model_names:
            if mod2 == top_model:
                continue
            p2 = np.array(preds_dict[mod2])
            ambos_acertam = np.sum((p1 == true_labels) & (p2 == true_labels))
            p1_acerta_p2_erra = np.sum((p1 == true_labels) & (p2 != true_labels))
            p1_erra_p2_acerta = np.sum((p1 != true_labels) & (p2 == true_labels))
            ambos_erram = np.sum((p1 != true_labels) & (p2 != true_labels))
            result = mcnemar([[ambos_acertam, p1_acerta_p2_erra], [p1_erra_p2_acerta, ambos_erram]], exact=True)
            if result.pvalue < 0.05:
                print(f"  ✅ {top_model} vs {mod2}: Diferença SIGNIFICATIVA (p={result.pvalue:.4f})")
            else:
                print(f"  ⚖️  {top_model} vs {mod2}: Empate Estatístico (p={result.pvalue:.4f})")

        all_preds = np.array([preds_dict[m] for m in model_names])
        acertos = all_preds == true_labels
        erros_totais_idx = np.where(np.sum(acertos, axis=0) == 0)[0]
        print("\n🔥 Análise de Falha Genuína:")
        print(f"  Ocorreram {len(erros_totais_idx)} imagens de teste que TODOS os modelos erraram.")
        if len(erros_totais_idx) > 0:
            print("  Índices dessas imagens no Dataset de Teste:", erros_totais_idx)

    def run(self, models: Iterable[str] = BENCHMARK_MODELS) -> pd.DataFrame:
        results = []
        for model_name in models:
            try:
                result = self.train_model(model_name)
                results.append(result)
            except Exception as exc:
                print(f"\n❌ ERRO CRÍTICO ao treinar o modelo '{model_name}': {exc}")
                print(f"PULANDO '{model_name}' e limpando a memória para o próximo modelo...\n")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not results:
            print("\n⚠️ Nenhum modelo foi treinado com sucesso. Encerrando o script.")
            return pd.DataFrame()

        df_new = pd.DataFrame(results)
        df_new["Params_M"] = df_new["Parâmetros"] / 1e6
        self._analyze_inter_model_agreement(df_new, results)

        df_clean = df_new.drop(columns=["test_preds", "test_labels"])
        csv_path = self.plot_dir / "benchmark_results_test_train.csv"
        df_final = self._update_results_csv(df_clean, csv_path)
        print("\n" + "=" * 85)
        print("RANKING FINAL ATUALIZADO (MÉTRICAS DE TREINO, VALIDAÇÃO E TESTE)")
        print("=" * 85)
        colunas_exibicao = ["Modelo", "Train_F1-Macro", "Val_F1-Macro", "Test_F1-Macro", "Test_AUC-Macro"]
        print(df_final[colunas_exibicao].to_string(index=False))

        self._plot_global_reports(df_final)
        print(f"\nTreino e atualização concluídos. Relatórios atualizados salvos em: {self.plot_dir}/")
        return df_final


def run_benchmark(models: Iterable[str] = BENCHMARK_MODELS) -> pd.DataFrame:
    """Atalho para executar o benchmark principal."""
    runner = BenchmarkRunner()
    return runner.run(models)


