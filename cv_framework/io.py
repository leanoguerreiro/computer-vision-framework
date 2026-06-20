"""Manipulação de arquivos de resultados."""

from __future__ import annotations

from pathlib import Path
import polars as pl


def update_results_csv(df_new: pl.DataFrame, csv_path: Path) -> pl.DataFrame:
    """Atualiza o arquivo CSV de forma funcional utilizando Polars."""
    if csv_path.exists():
        df_old = pl.read_csv(csv_path)

        # Extrai a lista de modelos como uma lista nativa do Python
        trained_models = df_new["Modelo"].to_list()

        # Filtra removendo os modelos que já foram treinados (upsert)
        df_old = df_old.filter(~pl.col("Modelo").is_in(trained_models))

        # Concatena os DataFrames verticalmente
        df_final = pl.concat([df_old, df_new])
    else:
        df_final = df_new

    # Ordena pelo F1-Macro decrescente e salva no disco
    df_final = df_final.sort("Test_F1-Macro", descending=True)
    df_final.write_csv(csv_path)

    return df_final