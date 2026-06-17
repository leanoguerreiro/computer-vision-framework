import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets
from config import DEFAULT_SEED, TRAINING_DATASET_ROOT, VISUALIZATION_ROBUSTNESS_DIR
from cv_framework.transforms import build_visualization_perturbation_transforms

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
PASTA_RAIZ = str(TRAINING_DATASET_ROOT)
ROBUSTNESS_DIR = str(VISUALIZATION_ROBUSTNESS_DIR)

os.makedirs(ROBUSTNESS_DIR, exist_ok=True)


perturbation_transforms = build_visualization_perturbation_transforms()


# =============================================================================
# CARREGAMENTO E PLOTAGEM
# =============================================================================

def plot_perturbation_samples():
    test_dir = os.path.join(PASTA_RAIZ, "test")

    # Carregamos usando dataset genérico apenas para pegar os caminhos das imagens
    dataset = datasets.ImageFolder(test_dir)
    if len(dataset) == 0:
        print("Nenhuma imagem encontrada na pasta de teste.")
        return

    # Escolhe uma imagem aleatória
    idx = random.randint(0, len(dataset) - 1)
    img_path, class_idx = dataset.samples[idx]
    class_name = dataset.classes[class_idx]

    # Abre a imagem original
    raw_image = Image.open(img_path).convert("RGB")
    print(f"Plotando amostras para a classe '{class_name}'...")

    # Configura o plot (2 linhas x 5 colunas)
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for ax, (pert_name, transform) in zip(axes, perturbation_transforms.items()):
        # Aplica a transformação
        img_tensor = transform(raw_image)

        # Converte de [C, H, W] para [H, W, C] para o Matplotlib
        img_np = img_tensor.permute(1, 2, 0).numpy()

        ax.imshow(img_np)
        ax.set_title(pert_name, fontsize=12, pad=10)
        ax.axis("off")

    plt.suptitle(f"Amostras de Perturbação - Classe: {class_name}", fontsize=18, y=1.02)
    plt.tight_layout()

    # Salva o gráfico
    out_path = os.path.join(ROBUSTNESS_DIR, "perturbation_samples.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Grid de imagens salvo em: {out_path}")


if __name__ == "__main__":
    # Fixar seed para sempre gerar a mesma imagem, se quiser (opcional)
    random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)

    plot_perturbation_samples()