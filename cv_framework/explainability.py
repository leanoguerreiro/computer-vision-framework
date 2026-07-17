"""Módulo de explicabilidade refatorado (Grad-CAM, Atenção, Espaço Latente -
DRY)."""

from pathlib import Path
from typing import Callable, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import umap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader

from cv_framework.transforms import IMAGENET_MEAN, IMAGENET_STD


# =====================================================================
# 1. HELPERS ATÔMICOS COMPARTILHADOS
# =====================================================================

def _unnormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Converte o tensor normalizado do PyTorch de volta para uma imagem RGB
    visualizável."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(std * img_np + mean, 0, 1)


def _sample_and_plot_loop(
        model: nn.Module, model_name: str, test_loader: DataLoader,
        device: torch.device,
        plot_dir: Path, class_names: List[str], sub_dir: str, title_prefix: str,
        overlay_fn: Callable[[torch.Tensor, int], np.ndarray],
        samples_per_class: int = 1,
) -> None:
    """Orquestrador genérico que elimina a repetição de loops, contagem e
    plotagem de amostras."""
    out_dir = plot_dir / model_name / sub_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    counts = {i: 0 for i in range(len(class_names))}

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        for i in range(inputs.size(0)):
            real_label = labels[i].item()
            if counts[real_label] >= samples_per_class:
                continue

            input_tensor = inputs[i].unsqueeze(0)
            with torch.no_grad():
                pred_class = model(input_tensor).argmax(dim=1).item()

            try:
                img_np = _unnormalize_image(input_tensor[0])
                overlay = overlay_fn(input_tensor, pred_class, img_np)
            except Exception as exc:
                print(
                    f"  ⚠️ Erro em {sub_dir} para classe "
                    f"{class_names[real_label]}: {exc}",
                )
                continue

            # Plotagem lado a lado padronizada
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(img_np)
            ax1.set_title(f"Original (Real: {class_names[real_label]})")
            ax1.axis("off")

            color_text = "darkgreen" if real_label == pred_class else "darkred"
            ax2.imshow(overlay)
            ax2.set_title(
                f"{title_prefix} (Pred: {class_names[pred_class]})",
                color=color_text, fontweight="bold",
            )
            ax2.axis("off")

            fig.tight_layout()
            counts[real_label] += 1
            file_name = (
                f"{class_names[real_label].replace(' ', '_')}_sample_"
                f"{counts[real_label]}_{sub_dir}.png")
            fig.savefig(out_dir / file_name, bbox_inches="tight")
            plt.close(fig)

            if all(c >= samples_per_class for c in counts.values()):
                return


# =====================================================================
# 2. FUNÇÕES DE ATENÇÃO DO TRANSFORMER
# =====================================================================

def extract_attention_map(
        model: nn.Module, input_tensor: torch.Tensor,
) -> np.ndarray:
    """Extrai a matriz de atenção via Forward Pre-Hook para modelos legados."""
    captured_inputs = []

    def pre_hook(module, input_args):
        captured_inputs.append(input_args[0].detach())

    handle = model.transformer.layers[-1].self_attn.register_forward_pre_hook(
        pre_hook,
    )
    with torch.no_grad():
        _ = model(input_tensor)
    handle.remove()

    qkv_input = captured_inputs[0]
    _, attn_weights = model.transformer.layers[-1].self_attn(
        qkv_input, qkv_input, qkv_input, need_weights=True,
    )
    patch_attention = attn_weights[0].mean(dim=0)
    grid_size = int(np.sqrt(patch_attention.size(0)))
    attention_map = patch_attention.reshape(
        grid_size, grid_size,
    ).detach().cpu().numpy()

    return (attention_map - attention_map.min()) / (
            attention_map.max() - attention_map.min() + 1e-8)


def generate_transformer_samples(
        model: nn.Module, model_name: str, test_loader: DataLoader,
        device: torch.device,
        plot_dir: Path, class_names: List[str], samples_per_class: int = 1,
) -> None:
    """Orquestra a geração dos mapas de atenção do Transformer (compatível
    com DINOv3)."""

    def _transformer_overlay(
            input_tensor: torch.Tensor, pred_class: int, img_np: np.ndarray,
    ) -> np.ndarray:
        if hasattr(model, 'get_last_self_attention'):
            attn_tensors = model.get_last_self_attention(input_tensor)
            # Fatiamento negativo (-196:) é à prova de falhas contra Register
            # Tokens do DINOv3!
            cls_attn = attn_tensors[0, :, 0, -196:]
            attention_map = cls_attn.mean(dim=0).reshape(14, 14).cpu().numpy()
        else:
            attention_map = extract_attention_map(model, input_tensor)

        attention_resized = cv2.resize(
            attention_map, (img_np.shape[1], img_np.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        heatmap = cv2.applyColorMap(
            np.uint8(255 * attention_resized), cv2.COLORMAP_JET,
        )
        heatmap = np.float32(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)) / 255
        return np.clip(heatmap * 0.6 + img_np * 0.4, 0, 1)

    _sample_and_plot_loop(
        model, model_name, test_loader, device, plot_dir, class_names,
        sub_dir="transformer_attention", title_prefix="Foco Híbrido",
        overlay_fn=_transformer_overlay, samples_per_class=samples_per_class,
    )


# =====================================================================
# 3. GRAD-CAM
# =====================================================================

def get_target_layer_for_cam(model: nn.Module, model_name: str) -> Optional[
    list]:
    """Identifica automaticamente a última camada convolucional para extração
    do Grad-CAM."""
    if "dino_hybrid" in model_name:
        return [model.conv_blocks[-1]]
    if "mobilenetv3" in model_name and hasattr(model, "blocks"):
        return [model.blocks[-1]]
    if "resnet" in model_name and hasattr(model, "layer4"):
        return [model.layer4[-1]]
    if "densenet" in model_name and hasattr(model, "features") and hasattr(
            model.features, "norm5",
    ):
        return [model.features.norm5]
    if "efficientnet" in model_name and hasattr(model, "blocks"):
        return [model.blocks[-1]]
    if "multicancernet_attention" in model_name.lower() or hasattr(
            model, "cbam4",
    ):
        return [model.cbam4]

    for _, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return [module]
    return None


def generate_gradcam_samples(
        model: nn.Module, model_name: str, test_loader: DataLoader,
        device: torch.device,
        plot_dir: Path, class_names: List[str], samples_per_class: int = 1,
) -> None:
    """Orquestra a geração de visualizações do Grad-CAM."""
    target_layers = get_target_layer_for_cam(model, model_name)
    if not target_layers:
        print(
            f"  ⚠️ Grad-CAM pulado para {model_name}: target layer não "
            f"encontrada.",
        )
        return

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
    except Exception as exc:
        print(f"  ⚠️ Erro ao inicializar Grad-CAM para {model_name}: {exc}")
        return

    def _gradcam_overlay(
            input_tensor: torch.Tensor, pred_class: int, img_np: np.ndarray,
    ) -> np.ndarray:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(pred_class)],
        )[0, :]
        return show_cam_on_image(
            img_np, np.asarray(grayscale_cam, dtype=np.float32), use_rgb=True,
        )

    _sample_and_plot_loop(
        model, model_name, test_loader, device, plot_dir, class_names,
        sub_dir="gradcam", title_prefix="Grad-CAM",
        overlay_fn=_gradcam_overlay, samples_per_class=samples_per_class,
    )


# =====================================================================
# 4. ESPAÇO LATENTE (UMAP)
# =====================================================================

def plot_latent_space(
        model: nn.Module, model_name: str, test_loader: DataLoader,
        device: torch.device,
        plot_dir: Path, class_names: List[str], seed: int,
) -> None:
    """Gera a projeção UMAP com proteção contra datasets pequenos."""
    print(
        f"  🌌 Gerando projeção do Espaço Latente (UMAP) para "
        f"{model_name}...",
    )
    model.eval()
    features, labels_list = [], []

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

    # Proteção de vizinhos dinâmica (evita avisos se rodar testes com batch
    # pequeno)
    safe_neighbors = min(15, max(2, features_np.shape[0] - 1))
    reducer = umap.UMAP(
        random_state=seed, n_neighbors=safe_neighbors, min_dist=0.1,
    )

    try:
        embedding = reducer.fit_transform(features_np)
    except Exception as exc:
        print(f"  ⚠️ Erro no UMAP para {model_name}: {exc}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=labels_np, cmap="coolwarm",
        alpha=0.7, s=50, edgecolors="k",
    )
    handles, _ = scatter.legend_elements()

    ax.legend(handles, class_names, title="Classes")
    ax.set_title(f"Espaço Latente (UMAP) — {model_name}")
    ax.set_xlabel("UMAP Dimensão 1")
    ax.set_ylabel("UMAP Dimensão 2")

    out_dir = plot_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "umap_latent_space.png", dpi=300)
    plt.close(fig)
