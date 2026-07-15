"""Pipeline comparativo de auditoria de robustez utilizando Ensembles K-Fold."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

# Importações da raiz do projeto
from config import (
    BENCHMARK_BATCH_SIZE_OVERRIDES,
    BENCHMARK_DATASET_ROOT,
    BENCHMARK_MODELS,
    BENCHMARK_PLOT_DIR,
    BENCHMARK_RESULTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_WORKERS,
    DEFAULT_SEED,
    IMAGE_SIZE,
    RADIMAGENET_WEIGHTS_URL,
)

# Importações do framework agnóstico
from cv_framework.context import BenchmarkContext
from cv_framework.metrics import calculate_all_metrics
from cv_framework.models import build_model
from cv_framework.reproducibility import set_seed
from cv_framework.transforms import build_perturbation_transforms


def load_kfold_ensemble(
    model_name: str,
    k: int,
    num_classes: int,
    context: BenchmarkContext,
    device: torch.device,
) -> List[nn.Module]:
    """Carrega para a VRAM os K modelos de uma arquitetura treinados na validação cruzada."""
    models = []
    print(f"\n⏳ Carregando Ensemble de {k} Folds para a arquitetura: {model_name.upper()}...")

    for fold_idx in range(1, k + 1):
        model = build_model(
            model_name,
            num_classes,
            pretrained=False,
            results_dir=str(context.results_dir),
            radimagenet_weights_url=context.radimagenet_weights_url,
        )

        # Procura o arquivo salvo no padrão do motor train_kfold_fold
        weights_path = (
            context.results_dir
            / model_name
            / f"fold_{fold_idx}"
            / f"best_{model_name}_f{fold_idx}.pth"
        )

        # Fallback de busca de caminho
        if not weights_path.exists():
            weights_path = context.results_dir / model_name / f"best_{model_name}_fold{fold_idx}.pth"

        if not weights_path.exists():
            raise FileNotFoundError(f"Pesos do fold não encontrados em: {weights_path}")

        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)
        print(f"   ✅ {model_name} [Fold {fold_idx}/{k}] carregado com sucesso.")

    return models


def run_ensemble_inference(
    models: List[nn.Module],
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> Tuple[List[int], List[int], np.ndarray, float]:
    """Executa inferência em consenso: calcula a média das probabilidades Softmax dos K modelos."""
    preds_list, labels_list, probs_list = [], [], []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device)

            batch_probs_per_model = []
            for model in models:
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                batch_probs_per_model.append(probs)

            # Média aritmética vetorial das probabilidades dos K modelos
            stacked_probs = torch.stack(batch_probs_per_model, dim=0)
            ensemble_probs = torch.mean(stacked_probs, dim=0)

            _, preds = torch.max(ensemble_probs, 1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.numpy())
            probs_list.append(ensemble_probs.cpu().numpy())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    return preds_list, labels_list, np.vstack(probs_list), total_time


def plot_comparative_degradation(df_results: pl.DataFrame, plot_dir: Path, k: int) -> None:
    """Gera gráficos comparativos mostrando a resiliência das diferentes arquiteturas ao ruído."""
    print("  📈 Gerando gráficos comparativos de degradação de robustez...")
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_pd = df_results.to_pandas()

    # 1. Gráfico Geral de Barras Agrupadas (Modelo vs Perturbação)
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(
        data=df_pd,
        x="Perturbação",
        y="Test_F1-Macro",
        hue="Modelo",
        palette="tab10",
        edgecolor="black",
        alpha=0.85,
        ax=ax,
    )

    ax.set_title(f"Auditoria de Robustez sob Estresse (Ensembles {k}-Fold) — F1-Score Macro", fontsize=15, pad=15)
    ax.set_xlabel("Cenário de Avaliação / Perturbação Aplicada", fontsize=12)
    ax.set_ylabel("F1-Score Macro (Teste)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Arquitetura (Ensemble)", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(plot_dir / f"kfold_{k}f_comparative_robustness_f1.png", dpi=300)
    plt.close(fig)

    # 2. Curvas de Queda de Performance (Lineplot para visualização de tendência)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(
        data=df_pd,
        x="Perturbação",
        y="Test_F1-Macro",
        hue="Modelo",
        style="Modelo",
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=9,
        palette="Dark2",
        ax=ax,
    )

    ax.set_title("Curva de Degradação de Performance por Arquitetura", fontsize=15, pad=15)
    ax.set_xlabel("Tipo e Intensidade do Ruído / Perturbação", fontsize=12)
    ax.set_ylabel("F1-Score Macro", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Arquitetura", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(plot_dir / f"kfold_{k}f_degradation_curves.png", dpi=300)
    plt.close(fig)


def run_kfold_robustness_benchmark(
    models: Iterable[str] = BENCHMARK_MODELS,
    k: int = 5,
    context: Optional[BenchmarkContext] = None,
) -> pl.DataFrame:
    """Orquestrador central para comparar a robustez de múltiplos Ensembles K-Fold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup de Contexto apontando para os diretórios gerados pelo benchmark_kfold
    ctx = context or BenchmarkContext(
        root_dir=BENCHMARK_DATASET_ROOT,
        plot_dir=BENCHMARK_PLOT_DIR / f"kfold_{k}_splits" / "robustness",
        results_dir=BENCHMARK_RESULTS_DIR / f"kfold_{k}_splits",
        batch_size=DEFAULT_BATCH_SIZE,
        eval_workers=DEFAULT_EVAL_WORKERS,
        seed=DEFAULT_SEED,
        radimagenet_weights_url=RADIMAGENET_WEIGHTS_URL,
        batch_size_overrides=BENCHMARK_BATCH_SIZE_OVERRIDES,
    )
    ctx.plot_dir.mkdir(parents=True, exist_ok=True)
    set_seed(ctx.seed)

    test_dir = ctx.root_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Diretório de teste não encontrado em: {test_dir}")

    # Descobre o número de classes via dataset temporário
    temp_ds = datasets.ImageFolder(test_dir, transform=None)
    num_classes = len(temp_ds.classes)
    del temp_ds

    perturbations = build_perturbation_transforms(IMAGE_SIZE)
    all_results = []

    print(f"\n{'=' * 80}")
    print(f"  INICIANDO AUDITORIA DE ROBUSTEZ MULTI-MODELO ({len(perturbations)} CENÁRIOS PER TURBAÇÃO)")
    print(f"{'=' * 80}")

    # 2. Loop Principal por Modelo
    for model_name in models:
        try:
            # Carrega o ensemble de K modelos treinados
            ensemble_models = load_kfold_ensemble(model_name, k, num_classes, ctx, device)
        except Exception as exc:
            print(f"\n❌ Erro ao carregar o ensemble de '{model_name}': {exc}")
            print("Pulando para o próximo modelo...\n")
            continue

        batch_size = ctx.batch_size_overrides.get(model_name, ctx.batch_size)

        # 3. Loop de Perturbações para a Arquitetura Atual
        for pert_name, transform_pipeline in perturbations.items():
            test_ds = datasets.ImageFolder(test_dir, transform=transform_pipeline)
            use_pin = torch.cuda.is_available()

            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=ctx.eval_workers,
                pin_memory=use_pin,
            )

            # Inferência consensual do Ensemble
            preds, labels, probs_matrix, total_time = run_ensemble_inference(
                ensemble_models, test_loader, device, desc=f"[{model_name}] -> {pert_name}"
            )

            metrics = calculate_all_metrics(labels, preds, probs_matrix, num_classes)
            dataset_size = len(test_loader.dataset)

            all_results.append({
                "Modelo": f"{model_name}_ens{k}f",
                "Perturbação": pert_name,
                "Test_Accuracy": metrics["accuracy"],
                "Test_Precision": metrics["precision"],
                "Test_Recall": metrics["recall"],
                "Test_F1-Macro": metrics["f1_macro"],
                "Test_Specificity": metrics["specificity"],
                "Test_MCC": metrics["mcc"],
                "Test_AUC-Macro": metrics["auc_macro"],
                "Test_MSE": metrics["mse"],
                "Inference_Time_s": total_time,
                "Inference_ms_per_img": (total_time / dataset_size) * 1000,
            })

            print(
                f"  [{model_name:18s} | {pert_name:16s}] -> F1-Macro: {metrics['f1_macro']:.4f} | Acc: {metrics['accuracy']:.4f}"
            )

            del test_loader, test_ds
            gc.collect()

        # Descarrega os K modelos da VRAM antes de carregar a próxima arquitetura
        for model in ensemble_models:
            del model
        del ensemble_models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_results:
        print("\n⚠️ Nenhum resultado foi gerado. Verifique os caminhos dos modelos.")
        return pl.DataFrame()

    # 4. Agregação Geral, Exportação e Plots
    df_results = pl.DataFrame(all_results)
    csv_path = ctx.plot_dir / f"benchmark_kfold_{k}f_robustness_summary.csv"
    df_results.write_csv(csv_path)

    print("\n" + "=" * 90)
    print("🏆 TABELA RESUMO DE ROBUSTEZ NO ESTRESSE (AMOSTRA: CLEAN vs EXTREMA)")
    print("=" * 90)

    # Filtra as perturbações principais para exibição rápida no terminal
    df_summary_view = df_results.filter(
        pl.col("Perturbação").is_in(["Clean", "Noise_Extrema", "Blur_Extrema", "Contrast_Extrema"])
    ).select(["Modelo", "Perturbação", "Test_F1-Macro", "Test_AUC-Macro", "Test_Accuracy"])

    with pl.Config(tbl_rows=len(df_summary_view)):
        print(df_summary_view)

    print(f"\n📁 Tabela completa exportada para: {csv_path}")

    # Gera os gráficos comparativos multi-modelo
    plot_comparative_degradation(df_results, ctx.plot_dir, k)

    return df_results


if __name__ == "__main__":
    run_kfold_robustness_benchmark(k=5)