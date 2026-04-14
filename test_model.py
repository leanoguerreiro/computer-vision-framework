import os
import gc
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================
INPUT_DIR = "datasets"
PASTA_RAIZ = f"{INPUT_DIR}/mri_split_70_20_10"
RESULTS_DIR = f"results_standart/{os.path.basename(PASTA_RAIZ)}"
ROBUSTNESS_DIR = f"robustness_analysis_standart/{os.path.basename(PASTA_RAIZ)}"
CSV_PATH = os.path.join(ROBUSTNESS_DIR, "robustness_metrics_levels.csv")
BATCH_SIZE = 32

# Configuração do Modelo Único
MODEL_NAME = "resnet18"  # Escolha o modelo aqui

os.makedirs(ROBUSTNESS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)


# =============================================================================
# CLASSES DE TRANSFORMAÇÃO E PERTURBAÇÃO
# =============================================================================

class AdjustContrast:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return TF.adjust_contrast(img, self.factor)


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) // 2)
        vp = int((max_wh - h) // 2)
        padding = [hp, vp, int(max_wh - w - hp), int(max_wh - h - vp)]
        return TF.pad(image, padding, 0, "constant")


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)


base_transforms = [
    SquarePad(),
    transforms.Resize((224, 224)),
]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

perturbation_transforms = {
    "Clean": transforms.Compose(base_transforms + [transforms.ToTensor(), normalize]),

    "Noise_Leve": transforms.Compose(base_transforms + [transforms.ToTensor(), AddGaussianNoise(std=0.05), normalize]),
    "Noise_Moderada": transforms.Compose(
        base_transforms + [transforms.ToTensor(), AddGaussianNoise(std=0.15), normalize]),
    "Noise_Extrema": transforms.Compose(
        base_transforms + [transforms.ToTensor(), AddGaussianNoise(std=0.30), normalize]),

    "Blur_Leve": transforms.Compose(
        base_transforms + [transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ToTensor(), normalize]),
    "Blur_Moderada": transforms.Compose(
        base_transforms + [transforms.GaussianBlur(kernel_size=5, sigma=2.0), transforms.ToTensor(), normalize]),
    "Blur_Extrema": transforms.Compose(
        base_transforms + [transforms.GaussianBlur(kernel_size=9, sigma=4.0), transforms.ToTensor(), normalize]),

    "Contrast_Leve": transforms.Compose(base_transforms + [AdjustContrast(0.6), transforms.ToTensor(), normalize]),
    "Contrast_Moderada": transforms.Compose(base_transforms + [AdjustContrast(0.3), transforms.ToTensor(), normalize]),
    "Contrast_Extrema": transforms.Compose(base_transforms + [AdjustContrast(0.1), transforms.ToTensor(), normalize])
}

# =============================================================================
# PREPARAÇÃO DOS DATALOADERS
# =============================================================================

print("Carregando datasets de teste perturbados...")
test_dir = os.path.join(PASTA_RAIZ, "test")

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

def evaluate_single_model(model_name: str) -> dict:
    weights_path = os.path.join(RESULTS_DIR, model_name, "best.pth")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pesos não encontrados para {model_name} em: {weights_path}")

    print(f"\n{'=' * 60}\n  AVALIANDO ROBUSTEZ: {model_name.upper()}\n{'=' * 60}")

    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    resultados_modelo = {
        "Data_Hora": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Modelo": model_name
    }

    for pert_name, loader in test_loaders.items():
        preds_list, labels_list, probs_list = [], [], []

        if torch.cuda.is_available(): torch.cuda.synchronize()
        start_infer = time.time()

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f"Inferência [{pert_name.ljust(15)}]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                probs_list.append(probs.cpu().numpy())

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
# GERAÇÃO DE GRÁFICO INDIVIDUAL
# =============================================================================

def plot_single_model_degradation(resultados: dict, model_name: str):
    """Gera um gráfico focado na degradação de um único modelo frente a diferentes ruídos."""

    # Estruturando os dados para o gráfico
    data = []
    baseline = resultados["F1_Clean"]

    tipos_pertubacao = ["Noise", "Blur", "Contrast"]
    niveis = ["Leve", "Moderada", "Extrema"]

    # Adicionando o baseline para todos os tipos para visualização
    for tipo in tipos_pertubacao:
        data.append({"Perturbação": tipo, "Severidade": "Limpo (Baseline)", "F1-Macro": baseline})
        for nivel in niveis:
            chave = f"F1_{tipo}_{nivel}"
            if chave in resultados:
                data.append({"Perturbação": tipo, "Severidade": nivel, "F1-Macro": resultados[chave]})

    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_plot,
        x="Perturbação",
        y="F1-Macro",
        hue="Severidade",
        palette="viridis",
        ax=ax
    )

    ax.set_title(f"Análise de Robustez (Degradação) — {model_name.upper()}", fontsize=14, pad=15, fontweight='bold')
    ax.set_ylabel("F1-Score Macro", fontsize=12)
    ax.set_xlabel("Tipo de Alteração", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Nível de Severidade", bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plot_path = os.path.join(ROBUSTNESS_DIR, f"robustness_degradation_{model_name}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    try:
        resultado = evaluate_single_model(MODEL_NAME)
        df_novo = pd.DataFrame([resultado])

        # Lógica de salvamento em CSV (Smart Append)
        if os.path.exists(CSV_PATH):
            df_existente = pd.read_csv(CSV_PATH)
            # Remove entrada anterior do mesmo modelo para não duplicar no histórico
            df_existente = df_existente[df_existente["Modelo"] != MODEL_NAME]
            df_final = pd.concat([df_existente, df_novo], ignore_index=True)
        else:
            df_final = df_novo

        # Reordena para manter os melhores no topo baseado na imagem limpa
        df_final = df_final.sort_values("F1_Clean", ascending=False)
        df_final.to_csv(CSV_PATH, index=False)

        print("\n" + "=" * 80)
        print(f"RESUMO SALVO EM: {CSV_PATH}")
        print("=" * 80)

        colunas_exibicao = ["Modelo", "F1_Clean", "AUC_Clean", "Time_ms_img_Clean"]
        print(df_novo[colunas_exibicao].to_string(index=False))

        # Gera o gráfico de degradação
        plot_single_model_degradation(resultado, MODEL_NAME)

        print(f"\n✅ Análise finalizada!")
        print(f"📊 Gráfico de degradação salvo em: {ROBUSTNESS_DIR}/robustness_degradation_{MODEL_NAME}.png")

    except Exception as e:
        print(f"\n❌ Falha durante a avaliação do modelo: {e}")