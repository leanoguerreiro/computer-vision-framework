"""Orquestrador de Treinamento Individual (Refatorado para usar o cv_framework)"""

import warnings
from pathlib import Path

import polars as pl
import torch

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MODEL_NAME,
    TRAINING_CSV_PATH,
    TRAINING_DATASET_ROOT,
    TRAINING_PLOT_DIR,
    TRAINING_RESULTS_DIR,
)
from cv_framework import (
    BenchmarkContext,
    setup_data_loaders,
    train_model_pipeline,
    update_results_csv,
    set_seed
)

warnings.filterwarnings("ignore", category=UserWarning)


def train_single_model(model_name: str, batch_size: int, epochs: int, lr: float):
    """Executa o pipeline modular do cv_framework para um único modelo."""

    # 1. Adaptando os caminhos da configuração de treino individual ao Contexto do Framework
    context = BenchmarkContext(
        root_dir=Path(TRAINING_DATASET_ROOT),
        plot_dir=Path(TRAINING_PLOT_DIR),
        results_dir=Path(TRAINING_RESULTS_DIR),
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=lr
    )

    context.plot_dir.mkdir(parents=True, exist_ok=True)
    context.results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(TRAINING_CSV_PATH)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Inicialização Segura
    set_seed(context.seed)
    loaders, class_names = setup_data_loaders(context, context.batch_size)

    # 3. Execução do Treinamento Delegado
    try:
        resultado = train_model_pipeline(model_name, context, loaders, class_names)

        # 4. Tratamento Polars e Exportação
        df_new = pl.DataFrame([resultado])
        df_new = df_new.with_columns((pl.col("Parâmetros") / 1e6).alias("Params_M"))

        # O strict=False no drop ignora o erro caso o pipeline já não retorne colunas descartáveis
        df_clean = df_new.drop(["test_preds", "test_labels"], strict=False)

        df_final = update_results_csv(df_clean, csv_path)

        print(f"\n✅ Treinamento finalizado. Resumo atualizado em: {csv_path}")
        with pl.Config(tbl_rows=1, tbl_cols=6):
            print(df_new.select(["Modelo", "Train_F1-Macro", "Val_F1-Macro", "Test_F1-Macro", "Test_AUC-Macro"]))

    except Exception as e:
        print(f"\n❌ Falha crítica encontrada durante o processo: {e}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    train_single_model(
        model_name=DEFAULT_MODEL_NAME,
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        lr=DEFAULT_LR
    )