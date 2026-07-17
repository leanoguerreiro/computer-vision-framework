"""Transforms compartilhadas usando funções puras e a API v2 do PyTorch (Paradigma Funcional - DRY)."""

from typing import Dict, List, Tuple
import torch
from torchvision.transforms import v2
import torchvision.transforms.functional as TF

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =====================================================================
# 1. FUNÇÕES PURAS DE TRANSFORMAÇÃO
# =====================================================================

def square_pad(image):
    """Adiciona padding para tornar a imagem quadrada antes do resize."""
    w, h = image.size
    max_wh = max(w, h)
    hp, vp = int((max_wh - w) // 2), int((max_wh - h) // 2)
    return TF.pad(image, [hp, vp, int(max_wh - w - hp), int(max_wh - h - vp)], 0, "constant")


def apply_contrast(factor: float):
    return lambda img: TF.adjust_contrast(img, factor)


def add_gaussian_noise(mean: float = 0.0, std: float = 0.1):
    def _add_noise(tensor):
        noise = torch.randn(tensor.size()) * std + mean
        return torch.clamp(tensor + noise, 0.0, 1.0)
    return _add_noise


# =====================================================================
# 2. PIPELINES BASE (DRY)
# =====================================================================

def _base_pipeline(image_size: int = IMAGE_SIZE) -> list:
    return [v2.Lambda(square_pad), v2.Resize((image_size, image_size), antialias=True)]


def _to_tensor_pipeline() -> list:
    return [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]


# =====================================================================
# 3. TRANSFORMS DE TREINO E AVALIAÇÃO
# =====================================================================

def build_train_transform(image_size: int = IMAGE_SIZE) -> v2.Compose:
    return v2.Compose(
        _base_pipeline(image_size) + [
            v2.RandomRotation(degrees=360),
            v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.0)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
            v2.RandomGrayscale(p=0.8),
            *_to_tensor_pipeline(),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            v2.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random')
        ]
    )


def build_eval_transform(image_size: int = IMAGE_SIZE) -> v2.Compose:
    return v2.Compose(_base_pipeline(image_size) + _to_tensor_pipeline() + [v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


# =====================================================================
# 4. FÁBRICA UNIFICADA DE ROBUSTEZ (DRY)
# =====================================================================

# Receita: (Nome Técnico, Nome Visualização, Operação, Exige_Tensor_Antes)
_PERTURBATION_RECIPES: List[Tuple[str, str, object, bool]] = [
    ("Noise_Leve", "Noise Leve\n(std=0.05)", v2.Lambda(add_gaussian_noise(std=0.05)), True),
    ("Noise_Moderada", "Noise Moderado\n(std=0.15)", v2.Lambda(add_gaussian_noise(std=0.15)), True),
    ("Noise_Extrema", "Noise Extremo\n(std=0.30)", v2.Lambda(add_gaussian_noise(std=0.30)), True),
    ("Blur_Leve", "Blur Leve\n(k=3, s=1.0)", v2.GaussianBlur(kernel_size=3, sigma=1.0), False),
    ("Blur_Moderada", "Blur Moderado\n(k=5, s=2.0)", v2.GaussianBlur(kernel_size=5, sigma=2.0), False),
    ("Blur_Extrema", "Blur Extremo\n(k=9, s=4.0)", v2.GaussianBlur(kernel_size=9, sigma=4.0), False),
    ("Contrast_Leve", "Contrast Leve\n(60%)", v2.Lambda(apply_contrast(0.6)), False),
    ("Contrast_Moderada", "Contrast Moderado\n(30%)", v2.Lambda(apply_contrast(0.3)), False),
    ("Contrast_Extrema", "Contrast Extremo\n(10%)", v2.Lambda(apply_contrast(0.1)), False),
]

def _build_perturbation_dict(visualize: bool = False, image_size: int = IMAGE_SIZE) -> Dict[str, v2.Compose]:
    """Constrói dinamicamente os dicionários de robustez evitando duplicar as 9 operações."""
    base = _base_pipeline(image_size)
    tensor_conv = _to_tensor_pipeline()
    norm = [] if visualize else [v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    clean_key = "Original / Clean" if visualize else "Clean"
    transforms_dict = {clean_key: v2.Compose(base + tensor_conv + norm)}

    for tech_name, vis_name, op, requires_tensor in _PERTURBATION_RECIPES:
        key = vis_name if visualize else tech_name
        if requires_tensor:
            pipeline = base + tensor_conv + [op] + norm
        else:
            pipeline = base + [op] + tensor_conv + norm

        transforms_dict[key] = v2.Compose(pipeline)

    return transforms_dict


def build_perturbation_transforms(image_size: int = IMAGE_SIZE) -> Dict[str, v2.Compose]:
    """Transforms para avaliação métrica de robustez (com normalização ImageNet)."""
    return _build_perturbation_dict(visualize=False, image_size=image_size)


def build_visualization_perturbation_transforms(image_size: int = IMAGE_SIZE) -> Dict[str, v2.Compose]:
    """Transforms para visualização humana (sem normalização ImageNet, com quebra de linha)."""
    return _build_perturbation_dict(visualize=True, image_size=image_size)