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

    if model_name == "multicancernet_attention":
        return MultiCancerNet_Attention(num_classes=num_classes)

    if model_name == "multicancernet_attention_hybrid":
        return MultiCancerNet_Attention_Hybrid(num_classes=num_classes)

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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class MultiCancerNet_Attention(nn.Module):
    def __init__(self, num_classes):
        super(MultiCancerNet_Attention, self).__init__()

        # Bloco 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.cbam1 = CBAM(32)

        # Bloco 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)

        # Bloco 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)

        # Bloco 4
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.cbam4 = CBAM(256)

        # Camadas base compartilhadas
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # --- Novo Classificador Otimizado com GAP ---
        self.gap = nn.AdaptiveAvgPool2d(1)  # Reduz a resolução (14x14) para (1x1) mantendo os 256 canais
        self.flatten = nn.Flatten()         # Transforma o formato (256, 1, 1) em um vetor linear de (256)
        self.fc1 = nn.Linear(256, 512)      # A entrada caiu de 50.176 para apenas 256!
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Bloco 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)  # Atenção aplicada na resolução rica antes do pooling
        x = self.pool(x)

        # Bloco 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.pool(x)

        # Bloco 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = self.pool(x)

        # Bloco 4
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.pool(x)

        # Classificador Otimizado
        x = self.gap(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiCancerNet_Attention_Hybrid(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=2):
        super(MultiCancerNet_Attention_Hybrid, self).__init__()

        # --- 1. Extrator de Características Local (CNN Backbone) ---
        self.cnn_backbone = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            # Bloco 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            # Bloco 3
            nn.Conv2d(128, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(), nn.MaxPool2d(2),
            # Bloco 4
            nn.Conv2d(d_model, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(), nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((14, 14))
        )

        # --- 2. Preparação para o Transformer (Tokenização) ---
        self.num_patches = 14 * 14
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # --- 3. Transformer Encoder (O Cérebro Global) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            activation="gelu",
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- 4. Classificador ---
        self.fc1 = nn.Linear(d_model, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 1. Extração Convolucional Dinâmica
        # Entrada: (B, 3, H, W) -> Saída Garantida: (B, 256, 14, 14)
        x = self.cnn_backbone(x)

        # 2. Flattening Espacial
        B, C, H, W = x.shape
        x = x.flatten(2)
        x = x.transpose(1, 2)

        # Adiciona a informação de posição de forma segura
        x = x + self.pos_embedding

        # 3. Processamento Global (Self-Attention)
        x = self.transformer(x)

        # 4. Global Average Pooling na dimensão dos tokens
        x = x.mean(dim=1)

        # 5. Classificação
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x