"""Transforms compartilhadas usando funções puras e a nova API v2 do PyTorch (Paradigma Funcional)."""

import torch
from torchvision.transforms import v2
import torchvision.transforms.functional as TF

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- 1. Funções Puras de Transformação ---

def square_pad(image):
    """Adiciona padding para tornar a imagem quadrada antes do resize."""
    w, h = image.size
    max_wh = max(w, h)
    hp = int((max_wh - w) // 2)
    vp = int((max_wh - h) // 2)
    padding = [hp, vp, int(max_wh - w - hp), int(max_wh - h - vp)]
    return TF.pad(image, padding, 0, "constant")


def apply_contrast(factor: float):
    """Retorna uma função que ajusta o contraste."""
    return lambda img: TF.adjust_contrast(img, factor)


def add_gaussian_noise(mean: float = 0.0, std: float = 0.1):
    """Retorna uma função que adiciona ruído gaussiano ao tensor."""
    def _add_noise(tensor):
        noise = torch.randn(tensor.size()) * std + mean
        return torch.clamp(tensor + noise, 0.0, 1.0)
    return _add_noise


# --- 2. Helpers e Pipelines Base ---

def _base_pipeline(image_size: int = IMAGE_SIZE):
    """Pipeline inicial compartilhado por todos os modos."""
    return [
        v2.Lambda(square_pad),
        v2.Resize((image_size, image_size), antialias=True)
    ]


def _to_tensor_pipeline():
    """Substitui o antigo transforms.ToTensor() pelo padrão seguro da v2."""
    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]


# --- 3. Transforms de Treinamento e Avaliação ---

def build_train_transform(image_size: int = IMAGE_SIZE):
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


def build_eval_transform(image_size: int = IMAGE_SIZE):
    """Transform limpo para validação e teste."""
    return v2.Compose(
        _base_pipeline(image_size) +
        _to_tensor_pipeline() + [
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# --- 4. Pipelines de Perturbação (Robustez) ---

def build_perturbation_transforms(image_size: int = IMAGE_SIZE):
    base = _base_pipeline(image_size)
    tensor_conversion = _to_tensor_pipeline()
    normalize = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    return {
        "Clean": v2.Compose(base + tensor_conversion + [normalize]),

        "Noise_Leve": v2.Compose(base + tensor_conversion + [
            v2.Lambda(add_gaussian_noise(std=0.05)),
            normalize
        ]),

        "Noise_Moderada": v2.Compose(base + tensor_conversion + [
            v2.Lambda(add_gaussian_noise(std=0.15)),
            normalize
        ]),

        "Noise_Extrema": v2.Compose(base + tensor_conversion + [
            v2.Lambda(add_gaussian_noise(std=0.30)),
            normalize
        ]),

        "Blur_Leve": v2.Compose(base + [
            v2.GaussianBlur(kernel_size=3, sigma=1.0)
        ] + tensor_conversion + [normalize]),

        "Blur_Moderada": v2.Compose(base + [
            v2.GaussianBlur(kernel_size=5, sigma=2.0)
        ] + tensor_conversion + [normalize]),

        "Blur_Extrema": v2.Compose(base + [
            v2.GaussianBlur(kernel_size=9, sigma=4.0)
        ] + tensor_conversion + [normalize]),

        "Contrast_Leve": v2.Compose(base + [
            v2.Lambda(apply_contrast(0.6))
        ] + tensor_conversion + [normalize]),

        "Contrast_Moderada": v2.Compose(base + [
            v2.Lambda(apply_contrast(0.3))
        ] + tensor_conversion + [normalize]),

        "Contrast_Extrema": v2.Compose(base + [
            v2.Lambda(apply_contrast(0.1))
        ] + tensor_conversion + [normalize]),
    }


def build_visualization_perturbation_transforms(image_size: int = IMAGE_SIZE):
    """Transforms para visualização de exemplos com nomes amigáveis (sem normalização matemática)."""
    base = _base_pipeline(image_size)
    tensor_conversion = _to_tensor_pipeline()

    return {
        "Original / Clean": v2.Compose(base + tensor_conversion),

        "Noise Leve\n(std=0.05)": v2.Compose(base + tensor_conversion + [
            v2.Lambda(add_gaussian_noise(std=0.05))
        ]),

        "Noise Moderado\n(std=0.15)": v2.Compose(base + tensor_conversion + [
            v2.Lambda(add_gaussian_noise(std=0.15))
        ]),

        "Noise Extremo\n(std=0.30)": v2.Compose(base + tensor_conversion + [
            v2.Lambda(add_gaussian_noise(std=0.30))
        ]),

        "Blur Leve\n(k=3, s=1.0)": v2.Compose(base + [
            v2.GaussianBlur(kernel_size=3, sigma=1.0)
        ] + tensor_conversion),

        "Blur Moderado\n(k=5, s=2.0)": v2.Compose(base + [
            v2.GaussianBlur(kernel_size=5, sigma=2.0)
        ] + tensor_conversion),

        "Blur Extremo\n(k=9, s=4.0)": v2.Compose(base + [
            v2.GaussianBlur(kernel_size=9, sigma=4.0)
        ] + tensor_conversion),

        "Contrast Leve\n(60%)": v2.Compose(base + [
            v2.Lambda(apply_contrast(0.6))
        ] + tensor_conversion),

        "Contrast Moderado\n(30%)": v2.Compose(base + [
            v2.Lambda(apply_contrast(0.3))
        ] + tensor_conversion),

        "Contrast Extremo\n(10%)": v2.Compose(base + [
            v2.Lambda(apply_contrast(0.1))
        ] + tensor_conversion),
    }