"""Transforms compartilhadas usando funções puras (Paradigma Funcional)."""

import torch
from torchvision import transforms
from torchvision.transforms import v2
import torchvision.transforms.functional as TF

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Funções puras de transformação
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

def _base_pipeline(image_size: int = IMAGE_SIZE):
    return [transforms.Lambda(square_pad), transforms.Resize((image_size, image_size))]

def build_train_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        _base_pipeline(image_size)
        + [
            v2.RandomRotation(degrees=360),
            v2.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.0)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
            v2.RandomGrayscale(p=0.8),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            v2.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random')
        ]
    )


def build_eval_transform(image_size: int = IMAGE_SIZE):
    """Transform limpo para validação e teste."""
    return transforms.Compose(
        _base_pipeline(image_size)
        + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_perturbation_transforms(image_size: int = IMAGE_SIZE):
    base = _base_pipeline(image_size)
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    return {
        "Clean": transforms.Compose(base + [
            transforms.ToTensor(),
            normalize
        ]),

        "Noise_Leve": transforms.Compose(base + [
            transforms.ToTensor(),
            transforms.Lambda(add_gaussian_noise(std=0.05)),
            normalize
        ]),

        "Noise_Moderada": transforms.Compose(base + [
            transforms.ToTensor(),
            transforms.Lambda(add_gaussian_noise(std=0.15)),
            normalize
        ]),

        "Noise_Extrema": transforms.Compose(base + [
            transforms.ToTensor(),
            transforms.Lambda(add_gaussian_noise(std=0.30)),
            normalize
        ]),

        "Blur_Leve": transforms.Compose(base + [
            transforms.GaussianBlur(kernel_size=3, sigma=1.0),
            transforms.ToTensor(),
            normalize
        ]),

        "Blur_Moderada": transforms.Compose(base + [
            transforms.GaussianBlur(kernel_size=5, sigma=2.0),
            transforms.ToTensor(),
            normalize
        ]),

        "Blur_Extrema": transforms.Compose(base + [
            transforms.GaussianBlur(kernel_size=9, sigma=4.0),
            transforms.ToTensor(),
            normalize
        ]),

        "Contrast_Leve": transforms.Compose(base + [
            transforms.Lambda(apply_contrast(0.6)),
            transforms.ToTensor(),
            normalize
        ]),

        "Contrast_Moderada": transforms.Compose(base + [
            transforms.Lambda(apply_contrast(0.3)),
            transforms.ToTensor(),
            normalize
        ]),

        "Contrast_Extrema": transforms.Compose(base + [
            transforms.Lambda(apply_contrast(0.1)),
            transforms.ToTensor(),
            normalize
        ]),
    }


def build_visualization_perturbation_transforms(image_size: int = IMAGE_SIZE):
    """Transforms para visualização de exemplos com nomes amigáveis (sem normalização)."""
    base = _base_pipeline(image_size)

    return {
        "Original / Clean": transforms.Compose(base + [
            transforms.ToTensor()
        ]),

        "Noise Leve\n(std=0.05)": transforms.Compose(base + [
            transforms.ToTensor(),
            transforms.Lambda(add_gaussian_noise(std=0.05))
        ]),

        "Noise Moderado\n(std=0.15)": transforms.Compose(base + [
            transforms.ToTensor(),
            transforms.Lambda(add_gaussian_noise(std=0.15))
        ]),

        "Noise Extremo\n(std=0.30)": transforms.Compose(base + [
            transforms.ToTensor(),
            transforms.Lambda(add_gaussian_noise(std=0.30))
        ]),

        "Blur Leve\n(k=3, s=1.0)": transforms.Compose(base + [
            transforms.GaussianBlur(kernel_size=3, sigma=1.0),
            transforms.ToTensor()
        ]),

        "Blur Moderado\n(k=5, s=2.0)": transforms.Compose(base + [
            transforms.GaussianBlur(kernel_size=5, sigma=2.0),
            transforms.ToTensor()
        ]),

        "Blur Extremo\n(k=9, s=4.0)": transforms.Compose(base + [
            transforms.GaussianBlur(kernel_size=9, sigma=4.0),
            transforms.ToTensor()
        ]),

        "Contrast Leve\n(60%)": transforms.Compose(base + [
            transforms.Lambda(apply_contrast(0.6)),
            transforms.ToTensor()
        ]),

        "Contrast Moderado\n(30%)": transforms.Compose(base + [
            transforms.Lambda(apply_contrast(0.3)),
            transforms.ToTensor()
        ]),

        "Contrast Extremo\n(10%)": transforms.Compose(base + [
            transforms.Lambda(apply_contrast(0.1)),
            transforms.ToTensor()
        ]),
    }