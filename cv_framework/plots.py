"""Módulo focado puramente em gerar gráficos estáticos (Matplotlib/Seaborn -
DRY)."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


# =====================================================================
# 1. HELPER DE CICLO DE VIDA DO MATPLOTLIB (DRY)
# =====================================================================

def _save_and_close(
        fig: plt.Figure, out_dir: Path, filename: str, dpi: int = 300,
) -> None:
    """Automatiza o salvamento, fechamento da figura e criação de diretórios
    no disco."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=dpi)
    plt.close(fig)


# =====================================================================
# 2. FUNÇÕES ATÔMICAS DE PLOTAGEM DE MODELO ÚNICO
# =====================================================================

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

    _save_and_close(fig, out_dir, "learning_curves.png")


def plot_f1_per_class(
        history: dict, class_names: List[str], out_dir: Path, model_name: str,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    f1_matrix = np.array(history["f1_per_class"])
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("tab10", n_colors=len(class_names))

    for i, class_name in enumerate(class_names):
        ax.plot(
            epochs, f1_matrix[:, i], label=class_name,
            color=colors[i % len(colors)], lw=2, marker="o", ms=4,
        )

    ax.plot(
        epochs, f1_matrix.mean(axis=1), label="macro (média)", color="k",
        lw=1.5, ls="--", alpha=0.6,
    )
    ax.set(
        title=f"F1 Validação por classe — {model_name}", xlabel="Época",
        ylabel="F1-Score", ylim=(0, 1.05),
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    _save_and_close(fig, out_dir, "f1_per_class.png")


def plot_roc_auc_curve(
        y_true: List[int], y_probs_matrix: np.ndarray, class_names: List[str],
        out_dir: Path, model_name: str,
) -> None:
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_matrix[:, i])
        ax.plot(
            fpr, tpr, lw=2, label=f"{class_name} (AUC = {auc(fpr, tpr):.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(
        xlim=(0.0, 1.0), ylim=(0.0, 1.05), xlabel="FPR", ylabel="TPR",
        title=f"Curvas ROC Validação (OvR) — {model_name}",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    _save_and_close(fig, out_dir, "roc_auc_curve_val.png")


def plot_confusion_matrix(
        y_true: List[int], y_preds: List[int], class_names: List[str],
        out_dir: Path, model_name: str,
) -> None:
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(
        figsize=(max(6, num_classes * 1.4), max(5, num_classes * 1.2)),
    )

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set(
        title=f"Matriz de Confusão Validação — {model_name}", xlabel="Previsão",
        ylabel="Rótulo real",
    )

    _save_and_close(fig, out_dir, "confusion_matrix_val.png")


def plot_confidence_distribution(
        model_name: str, plot_dir: Path, test_probs_matrix: np.ndarray,
        test_preds: List[int], test_labels: List[int],
) -> None:
    test_confidences = np.max(test_probs_matrix, axis=1)
    correct_mask = np.array(test_preds) == np.array(test_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(
        test_confidences[correct_mask], bins=20, color="green",
        label="Corretos", kde=True, alpha=0.6, ax=ax,
    )
    sns.histplot(
        test_confidences[~correct_mask], bins=20, color="red",
        label="Incorretos", kde=True, alpha=0.6, ax=ax,
    )
    ax.set(
        title=f"Distribuição de Confiança (Teste) — {model_name}",
        xlabel="Confiança (Probabilidade)", ylabel="Frequência",
    )
    ax.legend()

    _save_and_close(fig, plot_dir / model_name, "confidence_distribution.png")


# =====================================================================
# 3. ORQUESTRADORES GLOBAIS E RELATÓRIOS
# =====================================================================

def generate_model_plots(
        model_name: str, plot_dir: Path, class_names: List[str], history: dict,
        y_true: List[int], y_probs_matrix: np.ndarray, y_preds: List[int],
) -> None:
    out = plot_dir / model_name
    plot_learning_curves(history, out, model_name)
    plot_f1_per_class(history, class_names, out, model_name)
    plot_roc_auc_curve(y_true, y_probs_matrix, class_names, out, model_name)
    plot_confusion_matrix(y_true, y_preds, class_names, out, model_name)


def generate_global_reports(df_final: pl.DataFrame, plot_dir: Path) -> None:
    print("  📊 Gerando relatórios globais...")
    df_pd = df_final.to_pandas()  # Blindagem contra incompatibilidade no
    # Seaborn < 0.13

    # 1. Ranking Global
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=df_pd, x="Test_F1-Macro", y="Modelo", hue="Modelo",
        palette="viridis", legend=False, ax=ax,
    )
    ax.set_title("Ranking Global — F1-Score Macro (CONJUNTO DE TESTE)")
    _save_and_close(fig, plot_dir, "ranking_f1_test.png")

    if "Params_M" in df_pd.columns:
        # 2. Eficiência (Parâmetros vs F1)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=df_pd, x="Params_M", y="Test_F1-Macro", hue="Modelo", s=200,
            palette="tab20", legend=False, ax=ax,
        )
        for _, row in df_pd.iterrows():
            ax.text(
                row["Params_M"] + 0.5, row["Test_F1-Macro"], row["Modelo"],
                fontsize=9,
            )
        ax.set(
            title="Eficiência (Teste) — Parâmetros vs F1",
            xlabel="Parâmetros (M)",
        )
        _save_and_close(fig, plot_dir, "eficiencia_test.png")

        # 3. Velocidade vs Performance
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=df_pd, x="Inference_ms_per_img", y="Test_F1-Macro",
            size="Params_M", sizes=(50, 800), hue="Modelo", alpha=0.7,
            palette="tab20", legend=False, ax=ax,
        )
        for _, row in df_pd.iterrows():
            ax.text(
                row["Inference_ms_per_img"] * 1.02, row["Test_F1-Macro"],
                row["Modelo"], fontsize=9,
            )
        ax.set(
            title="Trade-off de Produção: Velocidade vs F1",
            xlabel="Tempo de Inferência (ms)", ylabel="F1-Score Macro",
        )
        ax.grid(True, ls="--", alpha=0.5)
        _save_and_close(fig, plot_dir, "velocidade_vs_performance.png")

    # 4. Análise de Overfitting
    df_melted = df_final.unpivot(
        index=["Modelo"], on=["Train_F1-Macro", "Test_F1-Macro"],
        variable_name="Conjunto", value_name="F1-Score",
    ).to_pandas()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(
        data=df_melted, x="F1-Score", y="Modelo", hue="Conjunto",
        palette="Set1", ax=ax,
    )
    ax.set(
        title="Análise de Overfitting (Treino vs Teste)",
        xlabel="F1-Score Macro", xlim=(0, 1.05),
    )
    _save_and_close(fig, plot_dir, "overfitting_analysis.png")

    # 5. Tempo de Treinamento
    df_time = df_final.with_columns(
        (pl.col("Train_Time_s") / 60.0).alias("Train_Time_min"),
    ).sort("Train_Time_min", descending=True).to_pandas()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=df_time, x="Train_Time_min", y="Modelo", palette="rocket", ax=ax,
    )
    ax.set(
        title="Custo de Treinamento: Tempo Acumulado na GPU",
        xlabel="Tempo de Treino (Minutos)",
    )
    _save_and_close(fig, plot_dir, "tempo_treinamento.png")


def plot_inter_model_agreement(
        agreement_matrix: np.ndarray, model_names: List[str], plot_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        agreement_matrix, annot=True, fmt=".2%", cmap="Purples",
        xticklabels=model_names, yticklabels=model_names, ax=ax,
    )
    ax.set_title("Matriz de Acordo Inter-Modelo (Predições Idênticas)")
    _save_and_close(fig, plot_dir, "inter_model_agreement.png")