"""Módulo de geração de amostras visuais de robustez (Refatorado - DRY e Pathlib)."""

from pathlib import Path
import random
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets

from config import DEFAULT_SEED, TRAINING_DATASET_ROOT, \
    VISUALIZATION_ROBUSTNESS_DIR
from cv_framework.transforms import build_visualization_perturbation_transforms

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
PASTA_RAIZ = Path(TRAINING_DATASET_ROOT)
ROBUSTNESS_DIR = Path(VISUALIZATION_ROBUSTNESS_DIR)
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

perturbation_transforms = build_visualization_perturbation_transforms()


# =============================================================================
# HELPERS ATÔMICOS (DRY)
# =============================================================================

def sanitizar_nome_arquivo(nome: str) -> str:
    """Remove quebras de linha e caracteres especiais para nomear arquivos no disco."""
    nome_limpo = re.sub(
        r'\s*\(.*?\)\s*|\s*\-\s*', ' ', nome.lower().replace('\n', '')
        ).strip()
    nome_limpo = re.sub(r'\s+', '_', nome_limpo)
    return re.sub(r'[^a-z0-9_]', '', nome_limpo)


def _save_1x4_grid(
        raw_image: Image.Image,
        perturbations: List[Tuple[str, object]],
        suptitle: str,
        out_path: Path
) -> None:
    """Isola a lógica de plotagem de um grid 1x4 (Original + 3 intensidades do ruído)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Coluna 0: Baseline (Original)
    axes[0].imshow(raw_image)
    axes[0].set_title(
        "Original (Clean)", fontsize=13, fontweight='bold', pad=10
        )
    axes[0].axis("off")

    # Colunas 1 a 3: As intensidades (Leve, Moderado, Extremo)
    for j, (pert_name, transform) in enumerate(perturbations, start=1):
        img_tensor = transform(raw_image)
        img_np = img_tensor.permute(1, 2, 0).numpy()

        axes[j].imshow(img_np)
        axes[j].set_title(pert_name, fontsize=12, pad=10)
        axes[j].axis("off")

    plt.suptitle(suptitle, fontsize=16, y=1.02, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# ORQUESTRADOR PRINCIPAL
# =============================================================================

def plot_perturbation_samples() -> None:
    test_dir = PASTA_RAIZ / "test"
    dataset = datasets.ImageFolder(str(test_dir))

    if len(dataset) == 0:
        print("⚠️ Nenhuma imagem encontrada na pasta de teste.")
        return

    # Seleciona uma imagem aleatória do teste
    idx = random.randint(0, len(dataset) - 1)
    img_path, class_idx = dataset.samples[idx]
    class_name = dataset.classes[class_idx]
    img_filename = Path(img_path).stem

    sample_out_dir = ROBUSTNESS_DIR / f"{class_name}_{img_filename}"
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    raw_image = Image.open(img_path).convert("RGB")
    print(
        f"🔬 Gerando análise de robustez para '{class_name}/{img_filename}'..."
        )

    # Pula a chave 'Original / Clean' e pega as 9 perturbações reais
    todos_itens = list(perturbation_transforms.items())
    perturbações = todos_itens[1:] if "original" in todos_itens[0][
        0].lower() else todos_itens

    # Agrupa de 3 em 3 (Leve, Moderado, Extremo para cada categoria)
    categorias = ["Noise", "Blur", "Contrast"]

    for i, nome_ruido_base in enumerate(categorias):
        grupo = perturbações[i * 3: (i + 1) * 3]

        titulo_geral = f"Análise de Robustez: {nome_ruido_base} (Classe: {class_name})"
        clean_name = sanitizar_nome_arquivo(nome_ruido_base)
        out_path = sample_out_dir / f"analise_{i + 1:02d}_{clean_name}.png"

        _save_1x4_grid(raw_image, grupo, titulo_geral, out_path)
        print(
            f" └─ Gerado ({i + 1}/{len(categorias)}): {nome_ruido_base} -> {out_path.name}"
            )

    print(f"\n✅ Concluído! As imagens foram salvas em:\n   {sample_out_dir}")


if __name__ == "__main__":
    random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)
    plot_perturbation_samples()