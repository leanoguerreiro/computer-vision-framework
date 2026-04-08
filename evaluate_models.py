import os
import gc
import random
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

PASTA_RAIZ = "mri_split_70_20_10"
RESULTS_DIR = "results"
ROBUSTNESS_DIR = "robustness_analysis"
BATCH_SIZE = 32

MODELOS = [
    # ── originais ──────────────────────────────────────────────────────────────
    "mobilenetv3_large_100", "efficientnet_b0", "resnet18", "resnet50",
    "efficientnet_b3", "convnext_small", "mobilevit_s", "fastvit_t8",
    "tiny_vit_11m_224", "vit_small_patch16_224", "swin_tiny_patch4_window7_224",
    "vit_base_patch16_224",
    # ── leves ──────────────────────────────────────────────────────────────────
    "efficientnet_b1", "efficientnet_b2", "mobilenetv3_small_100", "ghostnet_100",
    # ── médios ─────────────────────────────────────────────────────────────────
    "resnet34", "densenet121", "convnext_tiny", "swin_s3_tiny_224",
    "xcit_small_12_p16_224", "vit_base_patch32_224",
    # ── médico ─────────────────────────────────────────────────────────────────
    "densenet169"
]

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
    """Classe serializável para ajustar o contraste (substitui o lambda)."""
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return TF.adjust_contrast(img, self.factor)

class SquarePad:
    """Adiciona padding para tornar a imagem quadrada antes do resize."""

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) // 2)
        vp = int((max_wh - h) // 2)
        padding = [hp, vp, int(max_wh - w - hp), int(max_wh - h - vp)]
        return TF.pad(image, padding, 0, "constant")


class AddGaussianNoise(object):
    """Adiciona ruído gaussiano determinístico."""

    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # Semente fixada via dataloader (worker_init_fn ou estado global)
        # garante reprodutibilidade no teste.
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)


# Transformações base para todas as imagens
base_transforms = [
    SquarePad(),
    transforms.Resize((224, 224)),
]

# Normalização padrão da ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Dicionário com 3 níveis para cada tipo de perturbação
perturbation_transforms = {
    "Clean": transforms.Compose(base_transforms + [
        transforms.ToTensor(), normalize
    ]),

    # ── RUÍDO (Simula artefatos de sensor/aquisição) ─────────────────────────
    "Noise_Leve": transforms.Compose(base_transforms + [
        transforms.ToTensor(), AddGaussianNoise(std=0.05), normalize
    ]),
    "Noise_Moderada": transforms.Compose(base_transforms + [
        transforms.ToTensor(), AddGaussianNoise(std=0.15), normalize
    ]),
    "Noise_Extrema": transforms.Compose(base_transforms + [
        transforms.ToTensor(), AddGaussianNoise(std=0.30), normalize
    ]),

    # ── DESFOQUE (Simula movimento do paciente/baixa resolução) ──────────────
    "Blur_Leve": transforms.Compose(base_transforms + [
        transforms.GaussianBlur(kernel_size=3, sigma=1.0),
        transforms.ToTensor(), normalize
    ]),
    "Blur_Moderada": transforms.Compose(base_transforms + [
        transforms.GaussianBlur(kernel_size=5, sigma=2.0),
        transforms.ToTensor(), normalize
    ]),
    "Blur_Extrema": transforms.Compose(base_transforms + [
        transforms.GaussianBlur(kernel_size=9, sigma=4.0),
        transforms.ToTensor(), normalize
    ]),

    # ── CONTRASTE (Simula calibração deficiente da máquina MRI) ──────────────
    "Contrast_Leve": transforms.Compose(base_transforms + [
        AdjustContrast(0.6),  # Substituiu o lambda!
        transforms.ToTensor(), normalize
    ]),
    "Contrast_Moderada": transforms.Compose(base_transforms + [
        AdjustContrast(0.3),  # Substituiu o lambda!
        transforms.ToTensor(), normalize
    ]),
    "Contrast_Extrema": transforms.Compose(base_transforms + [
        AdjustContrast(0.1),  # Substituiu o lambda!
        transforms.ToTensor(), normalize
    ])
}

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
    return timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)


def evaluate_model(model_name: str) -> dict:
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
        preds_list, labels_list, probs_list = [], [], []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f"Inferência [{pert_name.ljust(15)}]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                probs_list.append(probs.cpu().numpy())

        probs_matrix = np.vstack(probs_list)
        f1_macro = f1_score(labels_list, preds_list, average="macro", zero_division=0)

        true_bin = label_binarize(labels_list, classes=list(range(NUM_CLASSES)))
        auc_macro = roc_auc_score(true_bin, probs_matrix, multi_class="ovr", average="macro")

        resultados_modelo[f"F1_{pert_name}"] = f1_macro
        resultados_modelo[f"AUC_{pert_name}"] = auc_macro

        print(f"  ➔ {pert_name.ljust(17)}: F1-Macro = {f1_macro:.4f} | AUC-Macro = {auc_macro:.4f}")

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

    df_melted = df.melt(
        id_vars=["Modelo"],
        value_vars=cols_to_plot,
        var_name="Nível",
        value_name="F1-Macro"
    )

    # Limpar os nomes para a legenda ficar elegante
    df_melted["Nível"] = df_melted["Nível"].str.replace("F1_", "").str.replace(f"{pert_prefix}_", "Nível ")

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

    # Gerando os 3 gráficos separados para legibilidade
    plot_robustness_group(df, "F1_Clean", "Noise", "Queda de Performance por Adição de Ruído", "robustness_noise.png")
    plot_robustness_group(df, "F1_Clean", "Blur", "Queda de Performance por Desfoque (Blur)", "robustness_blur.png")
    plot_robustness_group(df, "F1_Clean", "Contrast", "Queda de Performance por Redução de Contraste",
                          "robustness_contrast.png")

    print(f"\n✅ Análise de robustez finalizada com sucesso!")
    print(f"📊 3 novos gráficos gerados na pasta: {ROBUSTNESS_DIR}/")