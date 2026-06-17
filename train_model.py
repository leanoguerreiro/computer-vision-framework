import gc
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

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
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MODEL_NAME,
    DEFAULT_SEED,
    RADIMAGENET_WEIGHTS_URL,
    TRAINING_CSV_PATH,
    TRAINING_DATASET_ROOT,
    TRAINING_PLOT_DIR,
    TRAINING_RESULTS_DIR,
)
from cv_framework.data import build_class_weights, build_imagefolder_datasets
from cv_framework.models import build_model as build_shared_model
from cv_framework.reproducibility import set_seed
from cv_framework.training import EarlyStopping
from cv_framework.transforms import build_eval_transform, build_train_transform

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TrainConfig:
    """Configurações centralizadas para o treino atual."""
    model: str = DEFAULT_MODEL_NAME
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    seed: int = DEFAULT_SEED


class SingleModelTrainer:
    """Orquestra o ciclo completo de treino, avaliação e explicabilidade para um único modelo."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(self.config.seed)

        # Definição e criação de caminhos estruturados
        self.root_dir = Path(TRAINING_DATASET_ROOT)
        self.plot_dir = Path(TRAINING_PLOT_DIR) / self.config.model
        self.results_dir = Path(TRAINING_RESULTS_DIR) / self.config.model
        self.csv_path = Path(TRAINING_CSV_PATH)

        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Preparação de Pipelines de Dados
        self.transform_train = build_train_transform()
        self.transform_eval = build_eval_transform()

        self.dataset_train, self.dataset_val, self.dataset_test = build_imagefolder_datasets(
            self.root_dir,
            train_transform=self.transform_train,
            eval_transform=self.transform_eval,
        )

        self.class_names = self.dataset_train.classes
        self.num_classes = len(self.class_names)
        print(f"Classes detectadas ({self.num_classes}): {self.class_names}")

        self.train_loader, self.val_loader, self.test_loader = self._make_loaders()

    def _make_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train = DataLoader(self.dataset_train, batch_size=self.config.batch_size, shuffle=True, num_workers=8)
        val = DataLoader(self.dataset_val, batch_size=self.config.batch_size, shuffle=False, num_workers=8)
        test = DataLoader(self.dataset_test, batch_size=self.config.batch_size, shuffle=False, num_workers=8)
        return train, val, test

    def _build_criterion(self) -> nn.CrossEntropyLoss:
        weights = build_class_weights(self.dataset_train.targets, self.num_classes, device=self.device)
        return nn.CrossEntropyLoss(weight=weights)

    def _get_target_layer_for_cam(self, model: nn.Module) -> list | None:
        model_name_lower = self.config.model.lower()
        if "mobilenetv3" in model_name_lower: return [model.blocks[-1]]
        if "resnet" in model_name_lower: return [model.layer4[-1]]
        if "densenet" in model_name_lower: return [model.features.norm5]
        if "efficientnet" in model_name_lower: return [model.blocks[-1]]
        if "convnext" in model_name_lower: return [model.stages[-1].blocks[-1]]
        if "convformer" in model_name_lower: return [model.stages[-1].blocks[-1]]

        for _, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return [module]
        return None

    def save_plots(self, history: dict, y_true: list, y_probs_matrix: np.ndarray, y_preds: list) -> float:
        epochs = range(1, len(history["train_loss"]) + 1)

        # 1. Curvas de Loss e Acurácia
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(epochs, history["train_loss"], label="Treino")
        ax1.plot(epochs, history["val_loss"], label="Validação")
        ax1.set_title(f"Loss — {self.config.model}")
        ax1.set_xlabel("Época")
        ax1.legend()

        ax2.plot(epochs, history["train_acc"], label="Treino")
        ax2.plot(epochs, history["val_acc"], label="Validação")
        ax2.set_title(f"Acurácia — {self.config.model}")
        ax2.set_xlabel("Época")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(self.plot_dir / "learning_curves.png")
        plt.close(fig)

        # 2. F1 por classe
        f1_matrix = np.array(history["f1_per_class"])
        colors = plt.cm.tab10.colors
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, class_name in enumerate(self.class_names):
            ax.plot(epochs, f1_matrix[:, i], label=class_name, color=colors[i % len(colors)], linewidth=2, marker="o",
                    markersize=4)
        f1_macro_curve = f1_matrix.mean(axis=1)
        ax.plot(epochs, f1_macro_curve, label="macro (média)", color="black", linewidth=1.5, linestyle="--", alpha=0.6)
        ax.set_title(f"F1 Validação por classe — {self.config.model}")
        ax.set_xlabel("Época")
        ax.set_ylabel("F1-Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plot_dir / "f1_per_class.png")
        plt.close(fig)

        # 3. Curvas ROC multiclasse
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        if self.num_classes == 2:
            y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_matrix[:, i])
            roc_auc_i = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc_i:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"Curvas ROC Validação (OvR) — {self.config.model}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plot_dir / "roc_auc_curve_val.png")
        plt.close(fig)

        if self.num_classes == 2:
            macro_auc = roc_auc_score(y_true, y_probs_matrix[:, 1])
        else:
            macro_auc = roc_auc_score(y_true, y_probs_matrix, multi_class="ovr", average="macro")

        # 4. Matriz de Confusão da Validação
        cm = confusion_matrix(y_true, y_preds)
        fig, ax = plt.subplots(figsize=(max(6, self.num_classes * 1.4), max(5, self.num_classes * 1.2)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=self.class_names,
                    yticklabels=self.class_names, ax=ax)
        ax.set_title(f"Matriz de Confusão Validação — {self.config.model}")
        ax.set_xlabel("Previsão")
        ax.set_ylabel("Rótulo real")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "confusion_matrix_val.png")
        plt.close(fig)

        return macro_auc

    def plot_latent_space(self, model: nn.Module):
        print(f"  🌌 Gerando projeção do Espaço Latente (UMAP) para {self.config.model}...")
        model.eval()
        features, labels_list = [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
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

        reducer = umap.UMAP(random_state=self.config.seed, n_neighbors=15, min_dist=0.1)
        try:
            embedding = reducer.fit_transform(features_np)
        except Exception as e:
            print(f"  ⚠️ Erro no UMAP para {self.config.model}: {e}")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels_np, cmap="coolwarm", alpha=0.7, s=50,
                             edgecolors='k')
        handles, _ = scatter.legend_elements()
        ax.legend(handles, self.class_names, title="Classes")
        ax.set_title(f"Espaço Latente (UMAP) — {self.config.model}")
        ax.set_xlabel("UMAP Dimensão 1")
        ax.set_ylabel("UMAP Dimensão 2")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "umap_latent_space.png", dpi=300)
        plt.close(fig)

    def save_gradcam_samples(self, model: nn.Module, num_samples: int = 5):
        target_layers = self._get_target_layer_for_cam(model)
        if not target_layers:
            print(f"  ⚠️ Grad-CAM pulado para {self.config.model}: target layer não identificada.")
            return

        cam_dir = self.plot_dir / "gradcam"
        cam_dir.mkdir(parents=True, exist_ok=True)

        try:
            cam = GradCAM(model=model, target_layers=target_layers)
        except Exception as e:
            print(f"  ⚠️ Erro ao inicializar Grad-CAM para {self.config.model}: {e}")
            return

        model.eval()
        inputs, labels = next(iter(self.test_loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        n = min(num_samples, inputs.size(0))

        for i in range(n):
            input_tensor = inputs[i].unsqueeze(0)
            real_label = labels[i].item()

            with torch.no_grad():
                output = model(input_tensor)
                pred_class = output.argmax(dim=1).item()

            targets = [ClassifierOutputTarget(pred_class)]

            try:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            except Exception:
                continue

            img_np = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            img_np = np.clip(std * img_np + mean, 0, 1)

            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(img_np)
            ax1.set_title(f"Original (Real: {self.class_names[real_label]})")
            ax1.axis('off')

            cor_texto = "darkgreen" if real_label == pred_class else "darkred"
            ax2.imshow(cam_image)
            ax2.set_title(f"Grad-CAM (Pred: {self.class_names[pred_class]})", color=cor_texto, fontweight="bold")
            ax2.axis('off')

            fig.tight_layout()
            fig.savefig(cam_dir / f"sample_{i + 1}_cam.png", bbox_inches='tight')
            plt.close(fig)

    def _run_inference(self, model: nn.Module, loader: DataLoader, desc: str) -> tuple[
        list[int], list[int], np.ndarray, float]:
        preds_list, labels_list, probs_list = [], [], []
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=desc, leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                probs_list.append(probs.cpu().numpy())

        if torch.cuda.is_available(): torch.cuda.synchronize()
        total_time = time.time() - start_time
        return preds_list, labels_list, np.vstack(probs_list), total_time

    def train(self) -> dict:
        print(f"\n{'=' * 55}\n  INICIANDO TREINAMENTO: {self.config.model.upper()}\n{'=' * 55}")

        model = build_shared_model(
            self.config.model,
            self.num_classes,
            pretrained=True,
            results_dir=str(TRAINING_RESULTS_DIR),
            radimagenet_weights_url=RADIMAGENET_WEIGHTS_URL,
        ).to(self.device)

        criterion = self._build_criterion()
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        best_weights_path = self.results_dir / "best.pth"
        early_stopping = EarlyStopping(patience=self.config.patience, min_delta=self.config.min_delta,
                                       path=str(best_weights_path))

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "f1_per_class": []}
        train_time_acumulado = 0.0

        for epoch in range(self.config.epochs):
            # ── Ciclo de Treino ───────────────────────────────────────────────
            model.train()
            train_loss, train_correct = 0.0, 0

            if torch.cuda.is_available(): torch.cuda.synchronize()
            start_train_epoch = time.time()

            for inputs, labels in tqdm(self.train_loader, desc=f"Treino E{epoch + 1:02}", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()

            if torch.cuda.is_available(): torch.cuda.synchronize()
            train_time_acumulado += (time.time() - start_train_epoch)

            # ── Ciclo de Validação ────────────────────────────────────────────
            model.eval()
            val_loss, val_correct = 0.0, 0
            all_preds, all_labels, all_probs = [], [], []

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item() * inputs.size(0)

                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
                    val_correct += (preds == labels).sum().item()

            # ── Extração de Métricas da Época ─────────────────────────────────
            epoch_val_loss = val_loss / len(self.dataset_val)
            epoch_f1_classes = f1_score(all_labels, all_preds, average=None, zero_division=0)
            epoch_f1_macro = epoch_f1_classes.mean()

            history["train_loss"].append(train_loss / len(self.dataset_train))
            history["val_loss"].append(epoch_val_loss)
            history["train_acc"].append(train_correct / len(self.dataset_train))
            history["val_acc"].append(val_correct / len(self.dataset_val))
            history["f1_per_class"].append(epoch_f1_classes.tolist())

            scheduler.step(epoch_val_loss)

            print(f"E{epoch + 1:02} | Val Loss: {epoch_val_loss:.4f} | Val F1-Macro: {epoch_f1_macro:.4f}")

            epoch_data = {"y_true": all_labels, "y_probs": np.vstack(all_probs), "y_preds": all_preds}
            if early_stopping.step(epoch_f1_macro, model, epoch_data):
                print(f"\n  Early stopping na época {epoch + 1} — melhor F1 Val: {early_stopping.best_score:.4f}")
                break

        # Geração dos gráficos baseados na melhor época de validação
        best_val = early_stopping.best_data
        self.save_plots(history, best_val["y_true"], best_val["y_probs"], best_val["y_preds"])
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Carregar os melhores pesos salvos para a avaliação final de Teste/Treino
        model.load_state_dict(torch.load(best_weights_path, map_location=self.device))
        model.eval()

        # ── Avaliação Final no Conjunto de TESTE ──────────────────────────────
        print(f"\n  🔍 Extraindo métricas finais do conjunto de TESTE...")
        test_preds, test_labels, test_probs_matrix, total_infer_time = self._run_inference(model, self.test_loader,
                                                                                           "Avaliando Teste")

        test_f1_macro = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        test_auc_macro = roc_auc_score(test_labels,
                                       test_probs_matrix[:, 1] if self.num_classes == 2 else test_probs_matrix,
                                       multi_class="ovr", average="macro")

        # Matriz de Confusão do Teste
        cm_test = confusion_matrix(test_labels, test_preds)
        fig, ax = plt.subplots(figsize=(max(6, self.num_classes * 1.4), max(5, self.num_classes * 1.2)))
        sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", cbar=False, xticklabels=self.class_names,
                    yticklabels=self.class_names, ax=ax)
        ax.set_title(f"Matriz de Confusão TESTE — {self.config.model}")
        ax.set_xlabel("Previsão")
        ax.set_ylabel("Rótulo real")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "confusion_matrix_test.png")
        plt.close(fig)

        # Distribuição de Confiança
        test_confidences = np.max(test_probs_matrix, axis=1)
        correct_mask = np.array(test_preds) == np.array(test_labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(test_confidences[correct_mask], bins=20, color='green', label='Corretos', kde=True, alpha=0.6,
                     ax=ax)
        sns.histplot(test_confidences[~correct_mask], bins=20, color='red', label='Incorretos', kde=True, alpha=0.6,
                     ax=ax)
        ax.set_title(f"Distribuição de Confiança (Teste) — {self.config.model}")
        ax.set_xlabel("Confiança (Probabilidade da Classe Majoritária)")
        ax.set_ylabel("Frequência de Imagens")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.plot_dir / "confidence_distribution.png")
        plt.close(fig)

        # Ativação de gradientes temporária para extração do Grad-CAM
        print("  🎨 Gerando heatmaps do Grad-CAM...")
        for param in model.parameters(): param.requires_grad = True
        self.save_gradcam_samples(model, num_samples=5)
        for param in model.parameters(): param.requires_grad = False

        # ── Avaliação Final no Conjunto de TREINO ─────────────────────────────
        print(f"  🔍 Extraindo métricas finais do conjunto de TREINO...")
        train_preds, train_labels, train_probs_matrix, _ = self._run_inference(model, self.train_loader,
                                                                               "Avaliando Treino")
        train_f1_macro = f1_score(train_labels, train_preds, average="macro", zero_division=0)
        train_auc_macro = roc_auc_score(train_labels,
                                        train_probs_matrix[:, 1] if self.num_classes == 2 else train_probs_matrix,
                                        multi_class="ovr", average="macro")

        print(f"\n  🏆 TESTE  | F1-Macro: {test_f1_macro:.4f} | AUC-Macro: {test_auc_macro:.4f}")
        print(f"  🏆 TREINO | F1-Macro: {train_f1_macro:.4f} | AUC-Macro: {train_auc_macro:.4f}")

        # Projeção Espaço Latente UMAP
        self.plot_latent_space(model)

        # Limpeza agressiva da memória da GPU
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        return {
            "Modelo": self.config.model,
            "Train_F1-Macro": train_f1_macro,
            "Train_AUC-Macro": train_auc_macro,
            "Val_F1-Macro": early_stopping.best_score,
            "Test_F1-Macro": test_f1_macro,
            "Test_AUC-Macro": test_auc_macro,
            "Train_Time_s": train_time_acumulado,
            "Parâmetros": f"{num_params / 1e6:.2f} M",
        }

    def save_metrics_csv(self, summary_data: dict):
        df_novo = pd.DataFrame([summary_data])

        if self.csv_path.exists():
            df_existente = pd.read_csv(self.csv_path)
            df_existente = df_existente[df_existente["Modelo"] != self.config.model]
            df_final = pd.concat([df_existente, df_novo], ignore_index=True)
        else:
            df_final = df_novo

        df_final.to_csv(self.csv_path, index=False)
        print("\n" + "=" * 55)
        print(f"  RESULTADOS ATUALIZADOS EM: {self.csv_path}")
        print("=" * 55)
        print(df_novo.to_string(index=False))


if __name__ == "__main__":
    config = TrainConfig(
        model="resnet18",
        batch_size=32,
        epochs=50,
        lr=1e-3
    )

    try:
        trainer = SingleModelTrainer(config)
        resultado_treino = trainer.train()
        trainer.save_metrics_csv(resultado_treino)
        print(f"\n✅ Processo executado com sucesso para a arquitetura: {config.model}")
    except Exception as e:
        print(f"\n❌ Falha crítica encontrada durante o processo: {e}")