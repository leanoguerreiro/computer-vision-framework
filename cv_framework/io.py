"""Manipulação de arquivos de resultados (Refatorado - Upsert nativo com Polars)."""

from __future__ import annotations

from pathlib import Path
import polars as pl


def update_results_csv(df_new: pl.DataFrame, csv_path: Path) -> pl.DataFrame:
    """Atualiza o arquivo CSV utilizando operação de Upsert nativa em multithread no Polars."""
    csv_path = Path(csv_path)

    if csv_path.exists():
        df_old = pl.read_csv(csv_path)
        df_final = pl.concat([df_old, df_new]).unique(subset=["Modelo"], keep="last")
    else:
        df_final = df_new

    df_final = df_final.sort("Test_F1-Macro", descending=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.write_csv(csv_path)

    return df_final