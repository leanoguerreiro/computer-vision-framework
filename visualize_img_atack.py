import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PASTA_RAIZ = "mri_split_70_20_10"
ROBUSTNESS_DIR = "robustness_analysis"

os.makedirs(ROBUSTNESS_DIR, exist_ok=True)


# =============================================================================
# CLASSES DE TRANSFORMAÇÃO
# =============================================================================

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
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)


# Transformações base (pad e resize)
base_transforms = [
    SquarePad(),
    transforms.Resize((224, 224)),
]

# NOTA: Removemos o Normalize() para podermos visualizar as cores corretamente.
perturbation_transforms = {
    "Original / Clean": transforms.Compose(base_transforms + [
        transforms.ToTensor()
    ]),

    # ── RUÍDO ────────────────────────────────────────────────────────────────
    "Noise Leve\n(std=0.05)": transforms.Compose(base_transforms + [
        transforms.ToTensor(), AddGaussianNoise(std=0.05)
    ]),
    "Noise Moderado\n(std=0.15)": transforms.Compose(base_transforms + [
        transforms.ToTensor(), AddGaussianNoise(std=0.15)
    ]),
    "Noise Extremo\n(std=0.30)": transforms.Compose(base_transforms + [
        transforms.ToTensor(), AddGaussianNoise(std=0.30)
    ]),

    # ── DESFOQUE ─────────────────────────────────────────────────────────────
    "Blur Leve\n(k=3, s=1.0)": transforms.Compose(base_transforms + [
        transforms.GaussianBlur(kernel_size=3, sigma=1.0),
        transforms.ToTensor()
    ]),
    "Blur Moderado\n(k=5, s=2.0)": transforms.Compose(base_transforms + [
        transforms.GaussianBlur(kernel_size=5, sigma=2.0),
        transforms.ToTensor()
    ]),
    "Blur Extremo\n(k=9, s=4.0)": transforms.Compose(base_transforms + [
        transforms.GaussianBlur(kernel_size=9, sigma=4.0),
        transforms.ToTensor()
    ]),

    # ── CONTRASTE ────────────────────────────────────────────────────────────
    "Contrast Leve\n(60%)": transforms.Compose(base_transforms + [
        transforms.Lambda(lambda img: TF.adjust_contrast(img, 0.6)),
        transforms.ToTensor()
    ]),
    "Contrast Moderado\n(30%)": transforms.Compose(base_transforms + [
        transforms.Lambda(lambda img: TF.adjust_contrast(img, 0.3)),
        transforms.ToTensor()
    ]),
    "Contrast Extremo\n(10%)": transforms.Compose(base_transforms + [
        transforms.Lambda(lambda img: TF.adjust_contrast(img, 0.1)),
        transforms.ToTensor()
    ])
}


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
    random.seed(42)
    torch.manual_seed(42)

    plot_perturbation_samples()