"""Factories de modelos timm compartilhadas."""

from __future__ import annotations

import os
import urllib.request
from typing import Optional

import torch
import torch.nn as nn
import timm


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    results_dir: Optional[str] = None,
    radimagenet_weights_url: Optional[str] = None,
) -> nn.Module:
    """Cria um modelo timm padrão ou a variante RadImageNet do ResNet50."""
    if model_name != "resnet50_radimagenet":
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    if results_dir is None:
        raise ValueError("results_dir é obrigatório para carregar o ResNet50 RadImageNet.")
    if radimagenet_weights_url is None:
        raise ValueError("radimagenet_weights_url é obrigatório para carregar o ResNet50 RadImageNet.")

    weights_path = os.path.join(results_dir, "RadImageNet-ResNet50_notop.pth")
    if not os.path.exists(weights_path):
        print("  Baixando pesos RadImageNet...")
        urllib.request.urlretrieve(radimagenet_weights_url, weights_path)

    model = timm.create_model("resnet50", pretrained=False, num_classes=0)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    num_features = model.num_features.item() if hasattr(model.num_features, "item") else model.num_features
    model.fc = nn.Linear(int(num_features), num_classes)
    return model



