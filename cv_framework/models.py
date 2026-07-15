"""Factories de modelos timm compartilhadas."""

from __future__ import annotations

import os
import urllib.request
from typing import Optional

import torch
import torch.nn as nn
import timm
from transformers import AutoModel

class DinoSpatialHybrid(nn.Module):
    """
    Combina extração semântica do DINOv2/v3 (ViT) com refinamento espacial
    utilizando blocos pré-treinados de ConvNeXt ou ConvFormer (timm).
    """
    def __init__(
        self,
        dino_model_name: str = "facebook/dinov2-base",
        head_model_name: str = "convnext_base.fb_in22k",
        num_classes: int = 2,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        # 1. Carregamento do Backbone DINOv2 / DINOv3
        repo_id = "facebook/dinov3-vit-base" if "dinov3" in dino_model_name else "facebook/dinov2-base"
        self.dino = AutoModel.from_pretrained(repo_id, output_attentions=True)
        
        if freeze_backbone:
            for param in self.dino.parameters():
                param.requires_grad = False
                
        self.dino_dim = self.dino.config.hidden_size  # Ex: 768 para ViT-Base

        # 2. Carregamento do Head (ConvNeXt/ConvFormer pré-treinado no timm)
        # Criamos o modelo timm sem o classificador linear padrão (num_classes=0)
        self.spatial_head = timm.create_model(head_model_name, pretrained=True, num_classes=0)
        
        # O ConvNeXt do timm é dividido em 'stem', 'stages' e 'head' (GAP + Norm).
        # Como o DINO já fornece um mapa de resolução reduzida (ex: 14x14 ou 16x16),
        # pulamos o stem (que faria downsampling 4x4 agressivo) e usamos os estágios finais.
        if hasattr(self.spatial_head, "stages"):
            # Identifica a dimensão de entrada do penúltimo estágio do ConvNeXt (Ex: 512 em convnext_base)
            target_channels = self.spatial_head.stages[-2].blocks[0].conv_dw.in_channels
            
            # Projeção 1x1 para alinhar os 768 canais do DINO aos canais do ConvNeXt
            self.channel_proj = nn.Conv2d(self.dino_dim, target_channels, kernel_size=1)
            
            # Usamos os últimos blocos convolucionais pré-treinados (Estágios 2 e 3)
            self.conv_blocks = nn.Sequential(
                self.spatial_head.stages[-2],
                self.spatial_head.stages[-1]
            )
            head_out_dim = self.spatial_head.num_features
        else:
            # Fallback para outros modelos do timm: projeção direta e convolução espacial genérica
            self.channel_proj = nn.Conv2d(self.dino_dim, 512, kernel_size=1)
            self.conv_blocks = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512),  # Depthwise
                nn.GELU(),
                nn.Conv2d(512, 1024, kernel_size=1),                     # Pointwise
                nn.AdaptiveAvgPool2d((1, 1))
            )
            head_out_dim = 1024

        # 3. Pooling Global e Classificador Final
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(head_out_dim, num_classes)

    def _extract_spatial_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai tokens do DINO e converte para tensor espacial (B, C, H, W)."""
        outputs = self.dino(pixel_values=x)
        
        # Pega todos os tokens exceto o [CLS] -> formato: (B, N_patches, 768)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        
        B, N, C = patch_tokens.shape
        grid_size = int(math.sqrt(N))  # 14x14 para imagens 224x224 com patch_size=16 (ou 16x16 com patch_size=14)
        
        # Reshape e Permute: (B, H, W, C) -> (B, C, H, W)
        spatial_map = patch_tokens.reshape(B, grid_size, grid_size, C).permute(0, 3, 1, 2)
        return spatial_map

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna o embedding final 1D antes da camada linear.
        Garante compatibilidade total com plot_latent_space (UMAP) no explainability.py.
        """
        spatial_map = self._extract_spatial_grid(x)
        projected_map = self.channel_proj(spatial_map)
        conv_features = self.conv_blocks(projected_map)
        
        # Achata de (B, C, 1, 1) ou (B, C, H, W) para (B, C)
        return self.global_pool(conv_features).flatten(1)

    def forward(self, x: torch.Tensor):
        features = self.forward_features(x)
        return self.fc(features)

    def get_last_self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Garante que o explainability.py consiga gerar os mapas de atenção
        do Transformer normalmente através da função generate_transformer_samples.
        """
        outputs = self.dino(pixel_values=x)
        return outputs.attentions[-1]

class DinoVisionTransformer(nn.Module):
    """Wrapper robusto para integrar DINOv2/v3 perfeitamente ao pipeline de explicabilidade."""

    def __init__(self, model_name: str, num_classes: int, freeze_backbone: bool = True):
        super().__init__()

        if "dinov3" in model_name:
            repo_id = "facebook/dinov3-vit-base"
        else:
            repo_id = "facebook/dinov2-base"

        self.dino = AutoModel.from_pretrained(repo_id, output_attentions=True)

        if freeze_backbone:
            for param in self.dino.parameters():
                param.requires_grad = False

        self.embedding_dim = self.dino.config.hidden_size
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Salva o seu script UMAP! Retorna o embedding puro (768) antes do classificador."""
        outputs = self.dino(pixel_values=x)
        # Retorna o token [CLS] (B, 768) para o UMAP mapear a morfologia real da célula
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, x: torch.Tensor):
        # Usamos o forward_features interno
        features = self.forward_features(x)
        return self.fc(features)

    def get_last_self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Substitui a necessidade de Hooks manuais complexos no seu pipeline."""
        outputs = self.dino(pixel_values=x)
        # outputs.attentions é uma tupla com as atenções de todas as camadas
        # Pegamos a última camada [-1]: formato (batch_size, num_heads, sequence_length, sequence_length)
        return outputs.attentions[-1]

def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    results_dir: Optional[str] = None,
    radimagenet_weights_url: Optional[str] = None,
) -> nn.Module:

    if model_name == "dino_hybrid":
        return DinoSpatialHybrid(
            dino_model_name="facebook/dinov2-base",
            head_model_name="convnext_base.fb_in22k",
            num_classes=num_classes,
            freeze_backbone=True
        )

    if "dinov2" in model_name or "dinov3" in model_name:
        return DinoVisionTransformer(model_name=model_name, num_classes=num_classes, freeze_backbone=True)

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