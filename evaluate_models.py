"""Avaliação global de robustez de múltiplos modelos (Polars + CV Framework)."""

import gc
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    ROBUSTNESS_DATASET_ROOT,
    ROBUSTNESS_MODELS,
    ROBUSTNESS_PLOT_DIR,
    ROBUSTNESS_RESULTS_DIR,
)
from cv_framework import (
    build_model,
    build_perturbation_transforms,
    calculate_auc,
    run_inference,
    set_seed,
)

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================
PASTA_RAIZ = Path(ROBUSTNESS_DATASET_ROOT)
RESULTS_DIR = Path(ROBUSTNESS_RESULTS_DIR)
ROBUSTNESS_DIR = Path(ROBUSTNESS_PLOT_DIR)
BATCH_SIZE = DEFAULT_BATCH_SIZE
MODELOS = list(ROBUSTNESS_MODELS)

ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(DEFAULT_SEED)

# =============================================================================
# PREPARAÇÃO DOS DATALOADERS
# =============================================================================
print("Carregando datasets de teste perturbados...")
test_dir = PASTA_RAIZ / "test"

# Apenas para extrair as classes
dummy_dataset = datasets.ImageFolder(str(test_dir))
NOME_CLASSES = dummy_dataset.classes
NUM_CLASSES = len(NOME_CLASSES)
print(f"Classes detectadas ({NUM_CLASSES}): {NOME_CLASSES}")

perturbation_transforms = build_perturbation_transforms()
test_loaders = {}

for pert_name, trans in perturbation_transforms.items():
    ds_test = datasets.ImageFolder(str(test_dir), transform=trans)
    test_loaders[pert_name] = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# =============================================================================
# AVALIAÇÃO DO MODELO
# =============================================================================
def evaluate_model(model_name: str) -> Optional[dict]:
    weights_path = RESULTS_DIR / model_name / "best.pth"

    if not weights_path.exists():
        print(f"  ⚠️ Pesos não encontrados para {model_name}. Pulando...")
        return None

    print(f"\n{'=' * 60}\n  AVALIANDO ROBUSTEZ: {model_name.upper()}\n{'=' * 60}")

    model = build_model(model_name, NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    resultados_modelo = {"Modelo": model_name}

    for pert_name, loader in test_loaders.items():
        # Delegação total para a função pura do framework
        preds_list, labels_list, probs_matrix, total_infer_time = run_inference(
            model, loader, device, f"Inferência [{pert_name.ljust(15)}]"
        )

        infer_ms_per_img = (total_infer_time / len(loader.dataset)) * 1000
        f1_macro = f1_score(labels_list, preds_list, average="macro", zero_division=0)
        auc_macro = calculate_auc(labels_list, probs_matrix, NUM_CLASSES)

        resultados_modelo[f"F1_{pert_name}"] = f1_macro
        resultados_modelo[f"AUC_{pert_name}"] = auc_macro
        resultados_modelo[f"Time_ms_img_{pert_name}"] = infer_ms_per_img

        print(f"  ➔ {pert_name.ljust(17)}: F1-Macro = {f1_macro:.4f} | AUC-Macro = {auc_macro:.4f} | Tempo/img = {infer_ms_per_img:.2f} ms")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return resultados_modelo


# =============================================================================
# FUNÇÃO AUXILIAR PARA PLOTAGEM (API Polars)
# =============================================================================
def plot_robustness_group(df: pl.DataFrame, base_col: str, pert_prefix: str, title: str, filename: str):
    """Gera um gráfico comparando a base limpa com os 3 níveis de uma perturbação."""
    cols_to_plot = [base_col] + [col for col in df.columns if col.startswith(f"F1_{pert_prefix}")]

    # Substitui df.melt do Pandas pelo unpivot do Polars
    df_unpivoted = df.select(["Modelo"] + cols_to_plot).unpivot(
        index="Modelo",
        on=cols_to_plot,
        variable_name="Nível",
        value_name="F1-Macro"
    )

    # Limpeza elegante de strings usando a API de expressões do Polars
    df_unpivoted = df_unpivoted.with_columns(
        pl.col("Nível").str.replace("F1_", "").str.replace(f"{pert_prefix}_", "Nível ")
    )

    # Seaborn funciona melhor nativamente com Pandas, então convertemos apenas no momento do plot
    df_plot = df_unpivoted.to_pandas()

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(data=df_plot, x="Modelo", y="F1-Macro", hue="Nível", palette="rocket_r", ax=ax)

    ax.set_title(title, fontsize=16, pad=15, fontweight='bold')
    ax.set_ylabel("F1-Score Macro", fontsize=12)
    ax.set_xlabel("", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    ax.legend(title="Condição", bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plot_path = ROBUSTNESS_DIR / filename
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    resultados_gerais = []

    for nome in MODELOS:
        res = evaluate_model(nome)
        if res is not None:
            resultados_gerais.append(res)

    if not resultados_gerais:
        print("\nNenhum modelo foi avaliado. Verifique se a pasta 'results' contém os pesos treinados.")
        exit()

    # Criação do DataFrame Polars e ordenação nativa
    df = pl.DataFrame(resultados_gerais)
    if "F1_Clean" in df.columns:
        df = df.sort("F1_Clean", descending=True)

    csv_path = ROBUSTNESS_DIR / "robustness_metrics_levels.csv"
    df.write_csv(csv_path)

    print("\n" + "=" * 80)
    print("RESUMO EXECUTIVO SALVO EM CSV")
    print("=" * 80)

    colunas_exibicao = ["Modelo", "F1_Clean", "AUC_Clean", "Time_ms_img_Clean"]
    colunas_existentes = [col for col in colunas_exibicao if col in df.columns]

    if colunas_existentes:
        with pl.Config(tbl_rows=len(df), tbl_cols=len(colunas_existentes)):
            print(df.select(colunas_existentes))

    plot_robustness_group(df, "F1_Clean", "Noise", "Queda de Performance por Adição de Ruído", "robustness_noise.png")
    plot_robustness_group(df, "F1_Clean", "Blur", "Queda de Performance por Desfoque (Blur)", "robustness_blur.png")
    plot_robustness_group(df, "F1_Clean", "Contrast", "Queda de Performance por Redução de Contraste", "robustness_contrast.png")

    print(f"\n✅ Análise de robustez finalizada com sucesso!")
    print(f"📊 3 novos gráficos gerados na pasta: {ROBUSTNESS_DIR}/")