"""Pipeline modular do benchmark principal (Orquestrador de Aplicação)."""

from __future__ import annotations

from typing import Iterable, Optional

import polars as pl
import torch

# Importações da raiz do projeto
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

# Importações do framework agnóstico
from cv_framework.context import BenchmarkContext
from cv_framework.data import setup_data_loaders
from cv_framework.inter_model import calculate_inter_model_metrics
from cv_framework.io import update_results_csv
from cv_framework.plots import generate_global_reports, plot_inter_model_agreement
from cv_framework.reproducibility import set_seed
from cv_framework.training import train_model_pipeline


def run_benchmark(models: Iterable[str] = BENCHMARK_MODELS, context: Optional[BenchmarkContext] = None) -> pl.DataFrame:
    """Entry point principal de execução do benchmark com injeção de dependências."""

    # Construção do Contexto injetando as configurações específicas desta aplicação
    ctx = context or BenchmarkContext(
        root_dir=BENCHMARK_DATASET_ROOT,
        plot_dir=BENCHMARK_PLOT_DIR,
        results_dir=BENCHMARK_RESULTS_DIR,
        batch_size=DEFAULT_BATCH_SIZE,
        num_epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LR,
        patience=DEFAULT_EARLY_STOPPING_PATIENCE,
        min_delta=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        train_workers=DEFAULT_IMAGE_FOLDER_WORKERS,
        eval_workers=DEFAULT_EVAL_WORKERS,
        seed=DEFAULT_SEED,
        radimagenet_weights_url=RADIMAGENET_WEIGHTS_URL,
        batch_size_overrides=BENCHMARK_BATCH_SIZE_OVERRIDES
    )

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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results:
        print("\n⚠️ Nenhum modelo foi treinado com sucesso. Encerrando o script.")
        return pl.DataFrame()

    # Executa a análise estatística inter-modelo (Matemática Pura)
    agreement, model_names_res, mcnemar_results, erros_totais = calculate_inter_model_metrics(results)

    # Plota o heatmap de acordo se houver dados válidos
    if agreement.size > 0:
        plot_inter_model_agreement(agreement, model_names_res, ctx.plot_dir)

    # Criação e tratamento do DataFrame com Polars
    df_new = pl.DataFrame(results)
    df_new = df_new.with_columns((pl.col("Parâmetros") / 1e6).alias("Params_M"))

    # Limpeza de colunas temporárias/pesadas para persistência no CSV
    df_clean = df_new.drop(["test_preds", "test_labels"], strict=False)

    csv_path = ctx.plot_dir / "benchmark_results_test_train.csv"
    df_final = update_results_csv(df_clean, csv_path)

    print("\n" + "=" * 85)
    print("RANKING FINAL ATUALIZADO (MÉTRICAS DE TREINO, VALIDAÇÃO E TESTE)")
    print("=" * 85)

    colunas_exibicao = [
        "Modelo",
        "Val_F1-Macro",
        "Train_F1-Macro",
        "Test_F1-Macro",
        "Train_Accuracy",
        "Test_Accuracy",
        "Train_MCC",
        "Test_MCC",
        "Train_AUC-Macro",
        "Test_AUC-Macro",
        "Test_Specificity",
        "Test_MSE"
    ]
    with pl.Config(tbl_rows=len(df_final), tbl_cols=len(colunas_exibicao)):
        print(df_final.select(colunas_exibicao))

    # Geração dos relatórios gráficos globais agregados
    generate_global_reports(df_final, ctx.plot_dir)

    return df_final


if __name__ == "__main__":
    run_benchmark()