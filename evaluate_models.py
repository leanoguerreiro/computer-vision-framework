import os
import gc
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from dotenv import load_dotenv
from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    ROBUSTNESS_CSV_PATH,
    ROBUSTNESS_DATASET_ROOT,
    ROBUSTNESS_MODELS,
    ROBUSTNESS_PLOT_DIR,
    ROBUSTNESS_RESULTS_DIR,
)
from cv_framework.models import build_model as build_shared_model
from cv_framework.reproducibility import set_seed
from cv_framework.transforms import build_perturbation_transforms

load_dotenv()

# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================
PASTA_RAIZ = str(ROBUSTNESS_DATASET_ROOT)
RESULTS_DIR = str(ROBUSTNESS_RESULTS_DIR)
ROBUSTNESS_DIR = str(ROBUSTNESS_PLOT_DIR)
BATCH_SIZE = DEFAULT_BATCH_SIZE
MODELOS = list(ROBUSTNESS_MODELS)

Path(ROBUSTNESS_DIR).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


set_seed(DEFAULT_SEED)


perturbation_transforms = build_perturbation_transforms()

# =============================================================================
# PREPARAÇÃO DOS DATALOADERS
# =============================================================================

print("Carregando datasets de teste perturbados...")
test_dir = os.path.join(PASTA_RAIZ, "test")

# Apenas para extrair as classes
dummy_dataset = datasets.ImageFolder(test_dir)
NOME_CLASSES = dummy_dataset.classes
NUM_CLASSES = len(NOME_CLASSES)
print(f"Classes detectadas ({NUM_CLASSES}): {NOME_CLASSES}")

test_loaders = {}
for pert_name, trans in perturbation_transforms.items():
    ds_test = datasets.ImageFolder(test_dir, transform=trans)
    test_loaders[pert_name] = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# =============================================================================
# AVALIAÇÃO DO MODELO
# =============================================================================

def build_model(model_name: str) -> nn.Module:
    """Cria a arquitetura base do modelo sem pesos pré-treinados iniciais."""
    return build_shared_model(model_name, NUM_CLASSES, pretrained=False)


def evaluate_model(model_name: str) -> Optional[dict]:
    weights_path = os.path.join(RESULTS_DIR, model_name, "best.pth")

    if not os.path.exists(weights_path):
        print(f"  ⚠️ Pesos não encontrados para {model_name}. Pulando...")
        return None

    print(f"\n{'=' * 60}\n  AVALIANDO ROBUSTEZ: {model_name.upper()}\n{'=' * 60}")

    model = build_model(model_name).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    resultados_modelo = {"Modelo": model_name}

    for pert_name, loader in test_loaders.items():
        preds_list: list[int] = []
        labels_list: list[int] = []
        probs_list: list[np.ndarray] = []

        if torch.cuda.is_available(): torch.cuda.synchronize()
        start_infer = time.time()

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f"Inferência [{pert_name.ljust(15)}]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.detach().cpu().tolist())  # type: ignore[arg-type]
                probs_list.append(probs.detach().cpu().tolist())  # type: ignore[arg-type]

        if torch.cuda.is_available(): torch.cuda.synchronize()
        total_infer_time = time.time() - start_infer

        infer_ms_per_img = (total_infer_time / len(loader.dataset)) * 1000

        probs_matrix = np.vstack(probs_list)
        f1_macro = f1_score(labels_list, preds_list, average="macro", zero_division=0)

        if NUM_CLASSES == 2:
            auc_macro = roc_auc_score(labels_list, probs_matrix[:, 1])
        else:
            true_bin = label_binarize(labels_list, classes=list(range(NUM_CLASSES)))
            auc_macro = roc_auc_score(true_bin, probs_matrix, multi_class="ovr", average="macro")

        resultados_modelo[f"F1_{pert_name}"] = f1_macro
        resultados_modelo[f"AUC_{pert_name}"] = auc_macro
        resultados_modelo[f"Time_ms_img_{pert_name}"] = infer_ms_per_img

        print(
            f"  ➔ {pert_name.ljust(17)}: F1-Macro = {f1_macro:.4f} | AUC-Macro = {auc_macro:.4f} | Tempo/img = {infer_ms_per_img:.2f} ms")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return resultados_modelo


# =============================================================================
# FUNÇÃO AUXILIAR PARA PLOTAGEM
# =============================================================================

def plot_robustness_group(df, base_col, pert_prefix, title, filename):
    """Gera um gráfico comparando a base limpa com os 3 níveis de uma perturbação."""
    cols_to_plot = [base_col] + [col for col in df.columns if col.startswith(f"F1_{pert_prefix}")]

    df_melted = df.melt(  # type: ignore[call-arg]
        id_vars=["Modelo"],
        value_vars=cols_to_plot,
        var_name="Nível",
        value_name="F1_Macro"
    )
    df_melted = df_melted.rename(columns={"F1_Macro": "F1-Macro"})

    # Limpar os nomes para a legenda ficar elegante
    niveis = [str(valor).replace("F1_", "").replace(f"{pert_prefix}_", "Nível ") for valor in df_melted["Nível"].tolist()]
    df_melted["Nível"] = niveis

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(data=df_melted, x="Modelo", y="F1-Macro", hue="Nível", palette="rocket_r", ax=ax)

    ax.set_title(title, fontsize=16, pad=15, fontweight='bold')
    ax.set_ylabel("F1-Score Macro", fontsize=12)
    ax.set_xlabel("", fontsize=12)  # Esconde o texto do eixo X para focar nos nomes
    plt.xticks(rotation=45, ha="right", fontsize=10)
    ax.legend(title="Condição", bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plot_path = os.path.join(ROBUSTNESS_DIR, filename)
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

    # Criação do DataFrame e ordenação
    df = pd.DataFrame(resultados_gerais)
    if "F1_Clean" in df.columns:
        df = df.sort_values("F1_Clean", ascending=False)

    csv_path = os.path.join(ROBUSTNESS_DIR, "robustness_metrics_levels.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("RESUMO EXECUTIVO SALVO EM CSV")
    print("=" * 80)

    colunas_exibicao = ["Modelo", "F1_Clean", "AUC_Clean", "Time_ms_img_Clean"]
    if all(col in df.columns for col in colunas_exibicao):
        print(df[colunas_exibicao].to_string(index=False))


    plot_robustness_group(df, "F1_Clean", "Noise", "Queda de Performance por Adição de Ruído", "robustness_noise.png")
    plot_robustness_group(df, "F1_Clean", "Blur", "Queda de Performance por Desfoque (Blur)", "robustness_blur.png")
    plot_robustness_group(df, "F1_Clean", "Contrast", "Queda de Performance por Redução de Contraste",
                          "robustness_contrast.png")

    print(f"\n✅ Análise de robustez finalizada com sucesso!")
    print(f"📊 3 novos gráficos gerados na pasta: {ROBUSTNESS_DIR}/")