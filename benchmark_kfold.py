"""Pipeline modular de benchmark comparativo utilizando Validação Cruzada (K-Fold)."""

from __future__ import annotations

import gc
from typing import Iterable, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

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
from cv_framework.kfold import get_stratified_kfold_datasets
from cv_framework.inter_model import calculate_inter_model_metrics
from cv_framework.io import update_results_csv
from cv_framework.plots import generate_global_reports, \
    plot_inter_model_agreement
from cv_framework.reproducibility import set_seed
from cv_framework.training import train_kfold_fold


def run_kfold_benchmark(
        models: Iterable[str] = BENCHMARK_MODELS,
        k: int = 5,
        context: Optional[BenchmarkContext] = None,
) -> pl.DataFrame:
    """Entry point principal para execução de benchmarks comparativos sob K-Fold."""

    # 1. Construção do Contexto
    ctx = context or BenchmarkContext(
        root_dir=BENCHMARK_DATASET_ROOT,
        plot_dir=BENCHMARK_PLOT_DIR / f"kfold_{k}_splits",
        results_dir=BENCHMARK_RESULTS_DIR / f"kfold_{k}_splits",
        batch_size=DEFAULT_BATCH_SIZE,
        num_epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LR,
        patience=DEFAULT_EARLY_STOPPING_PATIENCE,
        min_delta=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        train_workers=DEFAULT_IMAGE_FOLDER_WORKERS,
        eval_workers=DEFAULT_EVAL_WORKERS,
        seed=DEFAULT_SEED,
        radimagenet_weights_url=RADIMAGENET_WEIGHTS_URL,
        batch_size_overrides=BENCHMARK_BATCH_SIZE_OVERRIDES,
    )

    ctx.plot_dir.mkdir(parents=True, exist_ok=True)
    ctx.results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(ctx.seed)

    # 2. Configuração do Pool e Divisão dos Folds (Mescla Train + Val por padrão)
    print(f"⏳ Gerando {k} Folds Estratificados a partir de: {ctx.root_dir}")
    folds_data = get_stratified_kfold_datasets(
        root_dir=ctx.root_dir,
        k=k,
        seed=ctx.seed,
        merge_val=True,
    )

    models_summary = []
    inter_model_raw_data = []

    # 3. Loop Exclusivo por Arquitetura de Modelo
    for model_name in models:
        print(
            f"\n{'=' * 70}\n  AVALIANO ARQUITETURA NO K-FOLD: {model_name.upper()}\n{'=' * 70}"
            )
        fold_results = []

        try:
            for fold_idx, (ds_train, ds_val, class_names) in enumerate(
                    folds_data
                    ):
                use_pin = torch.cuda.is_available()

                loader_train = DataLoader(
                    ds_train,
                    batch_size=ctx.batch_size,
                    shuffle=True,
                    num_workers=ctx.train_workers,
                    pin_memory=use_pin,
                )
                loader_val = DataLoader(
                    ds_val,
                    batch_size=ctx.batch_size,
                    shuffle=False,
                    num_workers=ctx.eval_workers,
                    pin_memory=use_pin,
                )

                # Aciona o motor enxuto focado exclusivamente em métricas OOF
                res = train_kfold_fold(
                    model_name=model_name,
                    context=ctx,
                    loaders=(loader_train, loader_val),
                    class_names=class_names,
                    fold_idx=fold_idx + 1,
                )
                fold_results.append(res)

                # Limpeza entre folds
                del loader_train, loader_val
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as exc:
            print(
                f"\n❌ ERRO CRÍTICO ao processar o modelo '{model_name}': {exc}"
                )
            print(f"PULANDO '{model_name}' e limpando a memória do GPU...\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if not fold_results:
            continue

        # 4. Agregação Estatística das Métricas Out-Of-Fold via Polars
        df_folds = pl.DataFrame(fold_results)

        # Vetoriza a união das predições de todos os folds para análise de concordância
        all_oof_preds = np.concatenate([r["oof_preds"] for r in fold_results])
        all_oof_labels = np.concatenate([r["oof_labels"] for r in fold_results])

        inter_model_raw_data.append(
            {
                "Modelo": model_name,
                "test_preds": all_oof_preds,
                "test_labels": all_oof_labels,
                "Test_F1-Macro": df_folds["Test_F1-Macro"].mean(),
            }
        )

        # Calcula Média e Desvio Padrão para as métricas do modelo atual
        summary_row = {
            "Modelo": model_name,
            "Params_M": df_folds["Parâmetros"].mean() / 1e6,
            "Inference_ms_per_img": df_folds["Inference_ms_per_img"].mean(),
            "Train_Time_s": df_folds["Train_Time_s"].sum(),

            # Médias de Desempenho (OOF)
            "Test_F1-Macro": df_folds["Test_F1-Macro"].mean(),
            "Test_F1-Std": df_folds["Test_F1-Macro"].std(),
            "Test_Accuracy": df_folds["Test_Accuracy"].mean(),
            "Test_Accuracy_Std": df_folds["Test_Accuracy"].std(),
            "Test_AUC-Macro": df_folds["Test_AUC-Macro"].mean(),
            "Test_AUC_Std": df_folds["Test_AUC-Macro"].std(),
            "Test_Specificity": df_folds["Test_Specificity"].mean(),
            "Test_MCC": df_folds["Test_MCC"].mean(),
            "Test_MSE": df_folds["Test_MSE"].mean(),

            # Espelha nas métricas de treino/val para compatibilidade com os plots globais
            "Train_F1-Macro": df_folds["Test_F1-Macro"].mean(),
            "Val_F1-Macro": df_folds["Test_F1-Macro"].mean(),
            "Train_Accuracy": df_folds["Test_Accuracy"].mean(),
            "Train_AUC-Macro": df_folds["Test_AUC-Macro"].mean(),
            "Train_MCC": df_folds["Test_MCC"].mean(),
        }
        models_summary.append(summary_row)

        # Salva o log granular de todos os folds do modelo no disco
        df_folds.drop(["oof_preds", "oof_labels"], strict=False).write_csv(
            ctx.results_dir / f"{model_name}_granular_folds.csv"
        )

    if not models_summary:
        print("\n⚠️ Nenhum modelo foi validado com sucesso. Encerrando.")
        return pl.DataFrame()

    # 5. Análise Estatística Inter-Modelo sobre as predições Out-Of-Fold
    print(
        "\n🔬 Executando análise estatística (McNemar e Concordância Inter-Modelo)..."
        )
    agreement, model_names_res, mcnemar_res, _ = calculate_inter_model_metrics(
        inter_model_raw_data
        )

    if agreement.size > 0:
        plot_inter_model_agreement(agreement, model_names_res, ctx.plot_dir)

    # 6. Atualização da Tabela Central de Resultados
    df_final = pl.DataFrame(models_summary)
    csv_path = ctx.plot_dir / f"benchmark_kfold_{k}f_results.csv"
    df_final = update_results_csv(df_final, csv_path)

    print("\n" + "=" * 90)
    print(
        f"🏆 RANKING COMPARATIVO FINAL — {k}-FOLD CROSS VALIDATION (OOF METRICS)"
        )
    print("=" * 90)

    colunas_exibicao = [
        "Modelo",
        "Test_F1-Macro",
        "Test_F1-Std",
        "Test_AUC-Macro",
        "Test_AUC_Std",
        "Test_Accuracy",
        "Test_MCC",
        "Params_M",
        "Inference_ms_per_img",
    ]
    with pl.Config(tbl_rows=len(df_final), tbl_cols=len(colunas_exibicao)):
        print(df_final.select(colunas_exibicao))

    # 7. Geração de Gráficos e Relatórios Agregados via Seaborn/Matplotlib
    generate_global_reports(df_final, ctx.plot_dir)

    return df_final


if __name__ == "__main__":
    run_kfold_benchmark(k=5)