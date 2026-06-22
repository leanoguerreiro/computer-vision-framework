from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import umap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader

from cv_framework.transforms import IMAGENET_MEAN, IMAGENET_STD

"""Módulo focado na abertura da "caixa-preta" (Grad-CAM, Atenção, Espaço Latente)."""

# --- Funções Puras de Processamento de Imagem/Tensor ---

def extract_attention_map(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Extrai a matriz de atenção da última camada do Transformer via Forward Pre-Hook."""
    captured_inputs = []

    def pre_hook(module, input_args):
        captured_inputs.append(input_args[0].detach())

    # Registra o hook
    handle = model.transformer.layers[-1].self_attn.register_forward_pre_hook(pre_hook)

    with torch.no_grad():
        _ = model(input_tensor)

    handle.remove()

    # Força a extração dos pesos
    qkv_input = captured_inputs[0]
    _, attn_weights = model.transformer.layers[-1].self_attn(qkv_input, qkv_input, qkv_input, need_weights=True)

    attention_matrix = attn_weights[0]
    patch_attention = attention_matrix.mean(dim=0)
    grid_size = int(np.sqrt(patch_attention.size(0)))
    attention_map = patch_attention.reshape(grid_size, grid_size).detach().cpu().numpy()

    # Normalização min-max
    return (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)


def create_heatmap_overlay(input_tensor: torch.Tensor, attention_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Combina a matriz de atenção com a imagem original retornando as duas em formato visualizável."""
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    # Processamento da Imagem Original
    img_np = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
    img_np = np.asarray(np.clip(std * img_np + mean, 0, 1), dtype=np.float32)

    # Criação do Heatmap (OpenCV)
    attention_map_resized = cv2.resize(attention_map, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # Mistura: 60% calor, 40% imagem original
    overlay = np.clip(heatmap * 0.6 + img_np * 0.4, 0, 1)

    return img_np, overlay


# --- Orquestradores de Interpretabilidade ---

def generate_transformer_samples(
        model: nn.Module, model_name: str, test_loader: DataLoader, device: torch.device,
        plot_dir: Path, class_names: List[str], samples_per_class: int = 1
) -> None:
    """Orquestrador que itera no dataset e salva os mapas de atenção do Transformer."""
    out_dir = plot_dir / model_name / "transformer_attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    counts_per_class = {i: 0 for i in range(len(class_names))}

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        for i in range(inputs.size(0)):
            real_label = labels[i].item()

            if counts_per_class[real_label] >= samples_per_class:
                continue

            input_tensor = inputs[i].unsqueeze(0)
            with torch.no_grad():
                pred_class = model(input_tensor).argmax(dim=1).item()

            # --- ONDE ADICIONAR A VERIFICAÇÃO (BLOCO TRY) ---
            try:
                if hasattr(model, 'get_last_self_attention'):
                    # 1. Extrai o tensor bruto do DINOv2/v3: formato (1, num_heads, 197, 197)
                    attn_tensors = model.get_last_self_attention(input_tensor)

                    # 2. Filtra a atenção que sai do Token [CLS] (índice 0) para os 196 patches de imagem (índice 1 em diante)
                    # Formato resultante: (num_heads, 196)
                    cls_attn = attn_tensors[0, :, 0, 1:]

                    # 3. Calcula a média aritmética entre todas as cabeças de atenção do Transformer
                    # Formato resultante: (196,)
                    mean_attn = cls_attn.mean(dim=0)

                    # 4. Faz o reshape do vetor plano para a matriz espacial 2D compatível com o Grid (14x14)
                    # Formato resultante: numpy array (14, 14)
                    attention_map = mean_attn.reshape(14, 14).cpu().numpy()
                else:
                    # Mantém o comportamento original via Hooks para os modelos antigos (ex: MultiCancerNet)
                    attention_map = extract_attention_map(model, input_tensor)

                # Renderiza o mapa sobreposto à célula de leucemia
                img_np, overlay = create_heatmap_overlay(input_tensor, attention_map)

            except Exception as exc:
                print(f"  ⚠️ Erro ao processar atenção para classe {class_names[real_label]}: {exc}")
                continue
            # ------------------------------------------------

            # Plotagem simples
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(img_np)
            ax1.set_title(f"Original (Real: {class_names[real_label]})")
            ax1.axis("off")

            color_text = "darkgreen" if real_label == pred_class else "darkred"
            ax2.imshow(overlay)
            ax2.set_title(f"Foco Híbrido (Pred: {class_names[pred_class]})", color=color_text, fontweight="bold")
            ax2.axis("off")

            fig.tight_layout()
            counts_per_class[real_label] += 1
            file_name = f"{class_names[real_label].replace(' ', '_')}_sample_{counts_per_class[real_label]}_attn.png"
            fig.savefig(out_dir / file_name, bbox_inches="tight")
            plt.close(fig)

            if all(count >= samples_per_class for count in counts_per_class.values()):
                return


def get_target_layer_for_cam(model: nn.Module, model_name: str) -> Optional[list]:
    """Identifica automaticamente a última camada convolucional para extração do Grad-CAM."""
    if "mobilenetv3" in model_name and hasattr(model, "blocks"):
        return [model.blocks[-1]]
    if "resnet" in model_name and hasattr(model, "layer4"):
        return [model.layer4[-1]]
    if "densenet" in model_name and hasattr(model, "features") and hasattr(model.features, "norm5"):
        return [model.features.norm5]
    if "efficientnet" in model_name and hasattr(model, "blocks"):
        return [model.blocks[-1]]
    if "multicancernet_attention" in model_name.lower() or hasattr(model, "cbam4"):
        return [model.cbam4]

    for _, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return [module]
    return None


def generate_gradcam_samples(
        model: nn.Module, model_name: str, test_loader: DataLoader, device: torch.device,
        plot_dir: Path, class_names: List[str], samples_per_class: int = 1
) -> None:
    """Gera visualizações do Grad-CAM garantindo amostras de TODAS as classes."""
    target_layers = get_target_layer_for_cam(model, model_name)
    if not target_layers:
        print(f"  ⚠️  Grad-CAM pulado para {model_name}: não foi possível identificar a target layer.")
        return

    out_dir = plot_dir / model_name / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
    except Exception as exc:
        print(f"  ⚠️  Erro ao inicializar Grad-CAM para {model_name}: {exc}")
        return

    model.eval()
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    counts_per_class = {i: 0 for i in range(len(class_names))}

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        for i in range(inputs.size(0)):
            real_label = labels[i].item()

            if counts_per_class[real_label] >= samples_per_class:
                continue

            input_tensor = inputs[i].unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                pred_class = output.argmax(dim=1).item()

            cam_targets = [ClassifierOutputTarget(pred_class)]
            try:
                grayscale_cam = cam(input_tensor=input_tensor, targets=cam_targets)[0, :]
            except Exception as exc:
                print(f"  ⚠️  Erro ao gerar Grad-CAM para classe {class_names[real_label]}: {exc}")
                continue

            img_np = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
            img_np = np.asarray(np.clip(std * img_np + mean, 0, 1), dtype=np.float32)
            grayscale_cam = np.asarray(grayscale_cam, dtype=np.float32)

            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(img_np)
            ax1.set_title(f"Original (Real: {class_names[real_label]})")
            ax1.axis("off")

            color_text = "darkgreen" if real_label == pred_class else "darkred"
            ax2.imshow(cam_image)
            ax2.set_title(f"Grad-CAM (Pred: {class_names[pred_class]})", color=color_text, fontweight="bold")
            ax2.axis("off")

            fig.tight_layout()
            counts_per_class[real_label] += 1
            file_name = f"{class_names[real_label].replace(' ', '_')}_sample_{counts_per_class[real_label]}_cam.png"
            fig.savefig(out_dir / file_name, bbox_inches="tight")
            plt.close(fig)

            if all(count >= samples_per_class for count in counts_per_class.values()):
                return


def plot_latent_space(model: nn.Module, model_name: str, test_loader: DataLoader, device: torch.device, plot_dir: Path,
                      class_names: List[str], seed: int) -> None:
    """Gera a projeção UMAP do espaço latente da última camada de características."""
    print(f"  🌌 Gerando projeção do Espaço Latente (UMAP) para {model_name}...")
    model.eval()
    features, labels_list = [], []
    out_dir = plot_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            try:
                feats = model.forward_features(inputs)
                if feats.ndim == 4:
                    feats = feats.mean(dim=[-2, -1])
                elif feats.ndim == 3:
                    feats = feats[:, 0]
            except Exception:
                feats = model(inputs)

            features.append(feats.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    features_np = np.vstack(features)
    labels_np = np.array(labels_list)

    reducer = umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.1)
    try:
        embedding = reducer.fit_transform(features_np)
    except Exception as exc:
        print(f"  ⚠️ Erro no UMAP para {model_name}: {exc}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels_np, cmap="coolwarm", alpha=0.7, s=50,
                         edgecolors="k")
    handles, _ = scatter.legend_elements()

    ax.legend(handles, class_names, title="Classes")
    ax.set_title(f"Espaço Latente (UMAP) — {model_name}")
    ax.set_xlabel("UMAP Dimensão 1")
    ax.set_ylabel("UMAP Dimensão 2")

    fig.tight_layout()
    fig.savefig(out_dir / "umap_latent_space.png", dpi=300)
    plt.close(fig)