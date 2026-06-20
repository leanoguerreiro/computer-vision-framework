"""Módulo focado puramente em gerar gráficos estáticos (Matplotlib/Seaborn)."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

# --- Funções Atômicas de Plotagem ---

def plot_learning_curves(history: dict, out_dir: Path, model_name: str) -> None:
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
    fig.savefig(out_dir / "learning_curves.png")
    plt.close(fig)


def plot_f1_per_class(history: dict, class_names: List[str], out_dir: Path, model_name: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    f1_matrix = np.array(history["f1_per_class"])
    num_classes = len(class_names)

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
    fig.savefig(out_dir / "f1_per_class.png")
    plt.close(fig)


def plot_roc_auc_curve(y_true: List[int], y_probs_matrix: np.ndarray, class_names: List[str], out_dir: Path, model_name: str) -> None:
    num_classes = len(class_names)
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
    fig.savefig(out_dir / "roc_auc_curve_val.png")
    plt.close(fig)


def plot_confusion_matrix(y_true: List[int], y_preds: List[int], class_names: List[str], out_dir: Path, model_name: str) -> None:
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 1.4), max(5, num_classes * 1.2)))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Matriz de Confusão Validação — {model_name}")
    ax.set_xlabel("Previsão")
    ax.set_ylabel("Rótulo real")

    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_val.png")
    plt.close(fig)


def plot_confidence_distribution(model_name: str, plot_dir: Path, test_probs_matrix: np.ndarray, test_preds: List[int], test_labels: List[int]) -> None:
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
    out_dir = plot_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confidence_distribution.png")
    plt.close(fig)


# --- Orquestradores Globais ---

def generate_model_plots(model_name: str, plot_dir: Path, class_names: List[str], history: dict, y_true: List[int], y_probs_matrix: np.ndarray, y_preds: List[int]) -> None:
    """Orquestra a geração de todos os gráficos pós-treinamento de um único modelo."""
    out = plot_dir / model_name
    out.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(history, out, model_name)
    plot_f1_per_class(history, class_names, out, model_name)
    plot_roc_auc_curve(y_true, y_probs_matrix, class_names, out, model_name)
    plot_confusion_matrix(y_true, y_preds, class_names, out, model_name)


def generate_global_reports(df_final: pl.DataFrame, plot_dir: Path) -> None:
    """Gera os relatórios comparativos do benchmark entre múltiplos modelos usando Polars."""
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

        # Traduzido de .iterrows() para .iter_rows(named=True) do Polars
        for row in df_final.iter_rows(named=True):
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
        for row in df_final.iter_rows(named=True):
            ax.text(row["Inference_ms_per_img"] * 1.02, row["Test_F1-Macro"], row["Modelo"], fontsize=9)

        ax.set_title("Trade-off de Produção: Velocidade vs F1 (Bolhas = Tamanho do Modelo)")
        ax.set_xlabel("Tempo de Inferência por Imagem (ms)")
        ax.set_ylabel("F1-Score Macro (Teste)")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(plot_dir / "velocidade_vs_performance.png", dpi=300)
        plt.close(fig)

    # 4. Análise de Overfitting (Substituído .melt do Pandas por .unpivot do Polars)
    df_melted = df_final.unpivot(
        index=["Modelo"],
        on=["Train_F1-Macro", "Test_F1-Macro"],
        variable_name="Conjunto",
        value_name="F1-Score"
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(data=df_melted, x="F1-Score", y="Modelo", hue="Conjunto", palette="Set1", ax=ax)
    ax.set_title("Análise de Overfitting (Treino vs Teste)")
    ax.set_xlabel("F1-Score Macro")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(plot_dir / "overfitting_analysis.png", dpi=300)
    plt.close(fig)

    # 5. Tempo de Treinamento (Tratamento sem mutação direta de colunas)
    df_final_sorted_time = df_final.with_columns(
        (pl.col("Train_Time_s") / 60.0).alias("Train_Time_min")
    ).sort("Train_Time_min", descending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=df_final_sorted_time, x="Train_Time_min", y="Modelo", palette="rocket", ax=ax)
    ax.set_title("Custo de Treinamento: Tempo Acumulado na GPU")
    ax.set_xlabel("Tempo de Treino (Minutos)")
    fig.tight_layout()
    fig.savefig(plot_dir / "tempo_treinamento.png", dpi=300)
    plt.close(fig)


def plot_inter_model_agreement(agreement_matrix: np.ndarray, model_names: List[str], plot_dir: Path) -> None:
    """Gera e salva o heatmap da matriz de acordo inter-modelo."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        agreement_matrix,
        annot=True,
        fmt=".2%",
        cmap="Purples",
        xticklabels=model_names,
        yticklabels=model_names,
        ax=ax
    )

    ax.set_title("Matriz de Acordo Inter-Modelo (Predições Idênticas)")
    fig.tight_layout()

    # Garante que o diretório base exista antes de salvar
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(plot_dir / "inter_model_agreement.png", dpi=300)
    plt.close(fig)