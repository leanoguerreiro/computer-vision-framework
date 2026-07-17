"""Factories de modelos timm compartilhadas (Versão Refatorada - DRY)."""

from __future__ import annotations

import math
import os
import urllib.request
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel


# =====================================================================
# 1. CLASSES BASE E BLOCOS REUTILIZÁVEIS (DRY HELPERS)
# =====================================================================

class BaseDinoWrapper(nn.Module):
    """Classe base que centraliza o carregamento, congelamento e extração de atenção do DINO."""
    def __init__(self, model_name: str, freeze_backbone: bool = True):
        super().__init__()
        if "dinov3" in model_name:
            repo_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        else:
            repo_id = "facebook/dinov2-base"

        print(f"  🧠 Carregando backbone semântico: {repo_id}...")
        self.dino = AutoModel.from_pretrained(repo_id, output_attentions=True)

        if freeze_backbone:
            for param in self.dino.parameters():
                param.requires_grad = False

        self.dino_dim = self.dino.config.hidden_size
        self.embedding_dim = self.dino_dim  # Alias para compatibilidade

    def get_last_self_attention(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.dino(pixel_values=x)
        return outputs.attentions[-1]


class ConvPoolBlock(nn.Module):
    """Bloco reutilizável: Conv2d -> BatchNorm -> ReLU -> [CBAM opcional] -> MaxPool2d."""
    def __init__(self, in_channels: int, out_channels: int, use_cbam: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_cbam:
            layers.append(CBAM(out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =====================================================================
# 2. MODELOS DINO (HÍBRIDOS E VIT PURO)
# =====================================================================

class DinoSpatialHybrid(BaseDinoWrapper):
    """Combina a extração semântica do DINOv2/v3 com QUALQUER backbone do timm."""
    def __init__(
            self,
            dino_model_name: str = "facebook/dinov2-base",
            head_model_name: str = "convnext_base.fb_in22k",
            num_classes: int = 2,
            freeze_backbone: bool = True,
            head_checkpoint_path: Optional[str] = None,
    ):
        super().__init__(model_name=dino_model_name, freeze_backbone=freeze_backbone)

        self.universal_adapter = nn.Sequential(
            nn.Conv2d(self.dino_dim, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
        )

        if head_checkpoint_path and os.path.exists(head_checkpoint_path):
            print(f"  📂 Carregando pesos locais para o Head ({head_model_name}): {head_checkpoint_path}")
            self.head_model = timm.create_model(
                head_model_name, pretrained=False, num_classes=num_classes, checkpoint_path=head_checkpoint_path
            )
        else:
            self.head_model = timm.create_model(head_model_name, pretrained=True, num_classes=num_classes)

    def _extract_spatial_grid(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.dino(pixel_values=x)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        B, N, C = patch_tokens.shape
        grid_size = int(math.sqrt(N))
        return patch_tokens.reshape(B, grid_size, grid_size, C).permute(0, 3, 1, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        spatial_map = self._extract_spatial_grid(x)
        adapted_map = self.universal_adapter(spatial_map)
        features = self.head_model.forward_features(adapted_map)

        if hasattr(self.head_model, "forward_head"):
            return self.head_model.forward_head(features, pre_logits=True)

        if features.dim() == 4:
            return F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return features.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_map = self._extract_spatial_grid(x)
        adapted_map = self.universal_adapter(spatial_map)
        return self.head_model(adapted_map)


class DinoVisionTransformer(BaseDinoWrapper):
    """Wrapper para integrar DINOv2/v3 ao pipeline de explicabilidade e classificação."""
    def __init__(self, model_name: str, num_classes: int, freeze_backbone: bool = True):
        super().__init__(model_name=model_name, freeze_backbone=freeze_backbone)
        self.fc = nn.Linear(self.dino_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.dino(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.fc(features)


# =====================================================================
# 3. MÓDULOS DE ATENÇÃO (CBAM)
# =====================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.ca(x)
        return out * self.sa(out)


# =====================================================================
# 4. REDES CUSTOMIZADAS
# =====================================================================

class MultiCancerNet_Attention(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        channels = [3, 32, 64, 128, 256]

        # Constrói os 4 blocos convolucionais com CBAM dinamicamente
        self.features = nn.Sequential(*[
            ConvPoolBlock(in_c, out_c, use_cbam=True)
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class MultiCancerNet_Attention_Hybrid(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 256, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        channels = [3, 64, 128, d_model, d_model]

        # Constrói os 4 blocos convolucionais padrão dinamicamente
        self.cnn_backbone = nn.Sequential(
            *[ConvPoolBlock(in_c, out_c, use_cbam=False) for in_c, out_c in zip(channels[:-1], channels[1:])],
            nn.AdaptiveAvgPool2d((14, 14))
        )

        self.num_patches = 14 * 14
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            activation="gelu", batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_backbone(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# =====================================================================
# 5. FÁBRICA DE MODELOS (ROTEADOR PRINCIPAL)
# =====================================================================

HYBRID_CONFIGS = {
    "dinov2_mobilenet_hybrid": ("facebook/dinov2-base", "mobilenetv3_large_100"),
    "dinov2_efficientnet_hybrid": ("facebook/dinov2-base", "efficientnet_b3"),
    "dinov3_mobilenet_hybrid": ("facebook/dinov3-vitb16-pretrain-lvd1689m", "mobilenetv3_large_100"),
    "dinov3_efficientnet_hybrid": ("facebook/dinov3-vitb16-pretrain-lvd1689m", "efficientnet_b3"),
}

def _load_radimagenet_resnet50(num_classes: int, results_dir: Optional[str], url: Optional[str]) -> nn.Module:
    """Função auxiliar para isolar o download e montagem do ResNet50 RadImageNet."""
    if not results_dir or not url:
        raise ValueError("results_dir e radimagenet_weights_url são obrigatórios para o RadImageNet.")

    weights_path = os.path.join(results_dir, "RadImageNet-ResNet50_notop.pth")
    if not os.path.exists(weights_path):
        print("  Baixando pesos RadImageNet...")
        urllib.request.urlretrieve(url, weights_path)

    model = timm.create_model("resnet50", pretrained=False, num_classes=0)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    num_features = model.num_features.item() if hasattr(model.num_features, "item") else model.num_features
    model.fc = nn.Linear(int(num_features), num_classes)
    return model


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    results_dir: Optional[str] = None,
    radimagenet_weights_url: Optional[str] = None,
) -> nn.Module:
    # 1. Roteamento de Modelos Híbridos DINO
    if model_name in HYBRID_CONFIGS:
        dino_id, head_id = HYBRID_CONFIGS[model_name]
        return DinoSpatialHybrid(dino_model_name=dino_id, head_model_name=head_id, num_classes=num_classes)

    # 2. Roteamento de Transformers DINO Puros
    if "dinov2" in model_name or "dinov3" in model_name:
        return DinoVisionTransformer(model_name=model_name, num_classes=num_classes, freeze_backbone=True)

    # 3. Roteamento de Redes Customizadas CBAM
    if model_name == "cbam_attention":
        return MultiCancerNet_Attention(num_classes=num_classes)

    if model_name == "cbam_attention_hybrid":
        return MultiCancerNet_Attention_Hybrid(num_classes=num_classes)

    # 4. Roteamento RadImageNet
    if model_name == "resnet50_radimagenet":
        return _load_radimagenet_resnet50(num_classes, results_dir, radimagenet_weights_url)

    # 5. Fallback padrão para a biblioteca timm
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)