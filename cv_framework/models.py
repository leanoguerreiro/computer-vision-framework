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

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.cbam1 = CBAM(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)

        # Block 4
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.cbam4 = CBAM(256)

        # Pooling
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # --- Classifier ---
        self.flatten = nn.Flatten()
        # Input calculation: 224 -> 112 -> 56 -> 28 -> 14. 256 channels * 14 * 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.cbam1(x)  # Apply Attention

        # Block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.cbam2(x)  # Apply Attention

        # Block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.cbam3(x)  # Apply Attention

        # Block 4
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.cbam4(x)  # Apply Attention

        # Classifier
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x