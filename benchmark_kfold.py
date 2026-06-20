"""Pipeline de Treinamento com Validação Cruzada (K-Fold)."""

from __future__ import annotations

import gc
import torch
import polars as pl
from pathlib import Path

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED,
    TRAINING_DATASET_ROOT,
    TRAINING_PLOT_DIR,
    TRAINING_RESULTS_DIR,
    RADIMAGENET_WEIGHTS_URL,
)

from cv_framework import (
    BenchmarkContext,
    set_seed,
    setup_data_loaders,
    train_model_pipeline,
    build_imagefolder_datasets
)
from cv_framework.kfold import get_kfold_indices, get_kfold_subsets


def run_kfold_benchmark(model_name: str, k: int = 5):
    # 1. Configuração do Contexto
    ctx = BenchmarkContext(
        root_dir=Path(TRAINING_DATASET_ROOT),
        plot_dir=Path(TRAINING_PLOT_DIR) / f"{model_name}_kfold",
        results_dir=Path(TRAINING_RESULTS_DIR) / f"{model_name}_kfold",
        batch_size=DEFAULT_BATCH_SIZE,
        num_epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LR,
        seed=DEFAULT_SEED,
        radimagenet_weights_url=RADIMAGENET_WEIGHTS_URL
    )

    ctx.plot_dir.mkdir(parents=True, exist_ok=True)
    ctx.results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(ctx.seed)

    # 2. Carrega dataset completo (ImageFolder)
    # Nota: usamos o mesmo setup de transforms do framework
    ds_train, _, _ = build_imagefolder_datasets(ctx.root_dir)
    class_names = ds_train.classes

    # 3. Gera os índices do K-Fold
    folds = get_kfold_indices(ds_train, k=k, seed=ctx.seed)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'=' * 20} FOLD {fold_idx + 1}/{k} {'=' * 20}")

        # Cria subconjuntos
        ds_fold_train, ds_fold_val = get_kfold_subsets(ds_train, train_idx, val_idx)

        # Cria Loaders temporários para este Fold
        loader_train = torch.utils.data.DataLoader(ds_fold_train, batch_size=ctx.batch_size, shuffle=True,
                                                   num_workers=ctx.train_workers)
        loader_val = torch.utils.data.DataLoader(ds_fold_val, batch_size=ctx.batch_size, shuffle=False,
                                                 num_workers=ctx.eval_workers)

        # Executa o pipeline de treino
        fold_res = train_model_pipeline(
            model_name=f"{model_name}_fold{fold_idx + 1}",
            context=ctx,
            loaders=(loader_train, loader_val, loader_val),  # Usa loader_val como teste para o fold
            class_names=class_names
        )
        fold_results.append(fold_res)

        # Limpeza de memória
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 4. Agregação Final com Polars
    df_kfold = pl.DataFrame(fold_results)

    print("\n" + "=" * 50)
    print(f"RESUMO DO K-FOLD ({k} folds)")
    print("=" * 50)

    # Média das métricas principais
    summary = df_kfold.select([
        pl.col("Test_F1-Macro").mean().alias("Avg_F1-Macro"),
        pl.col("Test_Accuracy").mean().alias("Avg_Accuracy"),
        pl.col("Test_AUC-Macro").mean().alias("Avg_AUC")
    ])

    print(summary)
    df_kfold.write_csv(ctx.results_dir / "kfold_results.csv")


if __name__ == "__main__":
    run_kfold_benchmark("resnet18", k=5)