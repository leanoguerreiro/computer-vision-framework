"""Transforms compartilhadas para treino, avaliação e visualização."""

from __future__ import annotations


import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SquarePad:
    """Adiciona padding para tornar a imagem quadrada antes do resize."""

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) // 2)
        vp = int((max_wh - h) // 2)
        padding = [hp, vp, int(max_wh - w - hp), int(max_wh - h - vp)]
        return TF.pad(image, padding, 0, "constant")


class AdjustContrast:
    """Ajuste de contraste serializável para usar em pipelines de transforms."""

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, img):
        return TF.adjust_contrast(img, self.factor)


class AddGaussianNoise:
    """Adiciona ruído gaussiano ao tensor de entrada."""

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def _base_pipeline(image_size: int = IMAGE_SIZE):
    return [SquarePad(), transforms.Resize((image_size, image_size))]


def build_train_transform(image_size: int = IMAGE_SIZE):
    """Transform de treino com augmentation e normalização ImageNet."""
    return transforms.Compose(
        _base_pipeline(image_size)
        + [
            transforms.RandomRotation(degrees=360),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
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
    """Transformações de robustez com as chaves usadas na avaliação."""
    base = _base_pipeline(image_size)
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    return {
        "Clean": transforms.Compose(base + [transforms.ToTensor(), normalize]),
        "Noise_Leve": transforms.Compose(base + [transforms.ToTensor(), AddGaussianNoise(std=0.05), normalize]),
        "Noise_Moderada": transforms.Compose(base + [transforms.ToTensor(), AddGaussianNoise(std=0.15), normalize]),
        "Noise_Extrema": transforms.Compose(base + [transforms.ToTensor(), AddGaussianNoise(std=0.30), normalize]),
        "Blur_Leve": transforms.Compose(base + [transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ToTensor(), normalize]),
        "Blur_Moderada": transforms.Compose(base + [transforms.GaussianBlur(kernel_size=5, sigma=2.0), transforms.ToTensor(), normalize]),
        "Blur_Extrema": transforms.Compose(base + [transforms.GaussianBlur(kernel_size=9, sigma=4.0), transforms.ToTensor(), normalize]),
        "Contrast_Leve": transforms.Compose(base + [AdjustContrast(0.6), transforms.ToTensor(), normalize]),
        "Contrast_Moderada": transforms.Compose(base + [AdjustContrast(0.3), transforms.ToTensor(), normalize]),
        "Contrast_Extrema": transforms.Compose(base + [AdjustContrast(0.1), transforms.ToTensor(), normalize]),
    }


def build_visualization_perturbation_transforms(image_size: int = IMAGE_SIZE):
    """Transforms para visualização de exemplos com nomes amigáveis."""
    base = _base_pipeline(image_size)

    return {
        "Original / Clean": transforms.Compose(base + [transforms.ToTensor()]),
        "Noise Leve\n(std=0.05)": transforms.Compose(base + [transforms.ToTensor(), AddGaussianNoise(std=0.05)]),
        "Noise Moderado\n(std=0.15)": transforms.Compose(base + [transforms.ToTensor(), AddGaussianNoise(std=0.15)]),
        "Noise Extremo\n(std=0.30)": transforms.Compose(base + [transforms.ToTensor(), AddGaussianNoise(std=0.30)]),
        "Blur Leve\n(k=3, s=1.0)": transforms.Compose(base + [transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ToTensor()]),
        "Blur Moderado\n(k=5, s=2.0)": transforms.Compose(base + [transforms.GaussianBlur(kernel_size=5, sigma=2.0), transforms.ToTensor()]),
        "Blur Extremo\n(k=9, s=4.0)": transforms.Compose(base + [transforms.GaussianBlur(kernel_size=9, sigma=4.0), transforms.ToTensor()]),
        "Contrast Leve\n(60%)": transforms.Compose(base + [AdjustContrast(0.6), transforms.ToTensor()]),
        "Contrast Moderado\n(30%)": transforms.Compose(base + [AdjustContrast(0.3), transforms.ToTensor()]),
        "Contrast Extremo\n(10%)": transforms.Compose(base + [AdjustContrast(0.1), transforms.ToTensor()]),
    }

