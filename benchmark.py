import os
import gc
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import warnings
import umap
from statsmodels.stats.contingency_tables import mcnemar
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================
INPUT_DIR = "datasets"
PASTA_RAIZ = f"{INPUT_DIR}/terrain_split_70_20_10"
PLOT_DIR = f"plots/{os.path.basename(PASTA_RAIZ)}"
RESULTS_DIR = f"results/{os.path.basename(PASTA_RAIZ)}"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-3
ES_PATIENCE = 5
ES_MIN_DELTA = 0.001

BATCH_SIZE_OVERRIDE = {
    "convnext_base": 32,
    "efficientnetv2_m": 32,
}

MODELOS = [
    # "mobilenetv3_large_100",
    # "efficientnet_b0",
    # "resnet18",
    # "resnet50",
    # "efficientnet_b3",
    # "convnext_small",
    # "mobilevit_s",
    # "fastvit_t8",
    # "tiny_vit_11m_224",
    # "vit_small_patch16_224",
    # "swin_tiny_patch4_window7_224",
    # "vit_base_patch16_224",
    # "efficientnet_b1",
    # "efficientnet_b2",
    # "mobilenetv3_small_100",
    # "ghostnet_100",
    # "resnet34",
    # "densenet121",
    # "convnext_tiny",
    # "swin_s3_tiny_224",
    # "xcit_small_12_p16_224",
    # "vit_base_patch32_224",
    # "densenet169"

    # 🔹 Leves
    "mobilenetv3_large_100",
    "mobilevit_s",

    # 🔹 Médios
    "resnet50",
    "convformer_s18",

    # 🔹 Médio-alto
    "efficientnet_b3",
    "maxvit_tiny_tf_224",

    # 🔹 Transformers (ajustados)
    "swin_tiny_patch4_window7_224",
    "vit_base_patch16_224",

    # 🔹 Alto
    "convnext_base",
    "swin_base_patch4_window7_224",
]

RADIMAGENET_WEIGHTS_URL = (
    "https://huggingface.co/BMEII/RadImageNet/resolve/main/"
    "RadImageNet-ResNet50_notop.pth"
)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REPRODUTIBILIDADE
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)


# =============================================================================
# PRÉ-PROCESSAMENTO
# =============================================================================

class SquarePad:
    """Adiciona padding para tornar a imagem quadrada antes do resize."""

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) // 2)
        vp = int((max_wh - h) // 2)
        padding = [hp, vp, int(max_wh - w - hp), int(max_wh - h - vp)]
        return TF.pad(image, padding, 0, "constant")


transform_train = v2.Compose([
    SquarePad(),
    v2.Resize((224, 224)),

    v2.RandomRotation(degrees=360),

    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.Grayscale(num_output_channels=3),

    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    v2.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random')
])

transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# =============================================================================
# DATASETS E DATALOADERS
# =============================================================================

dataset_train = datasets.ImageFolder(os.path.join(PASTA_RAIZ, "train"), transform=transform_train)
dataset_val = datasets.ImageFolder(os.path.join(PASTA_RAIZ, "val"), transform=transform)
dataset_test = datasets.ImageFolder(os.path.join(PASTA_RAIZ, "test"), transform=transform)

NOME_CLASSES = dataset_train.classes
NUM_CLASSES = len(NOME_CLASSES)
print(f"Classes detectadas ({NUM_CLASSES}): {NOME_CLASSES}")


def make_loaders(batch_size: int):
    """Cria dataloaders com o batch_size solicitado."""
    train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)
    test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)
    return train, val, test


# =============================================================================
# FUNÇÃO DE PERDA COM PESOS (para datasets desbalanceados)
# =============================================================================

def build_criterion() -> nn.CrossEntropyLoss:
    """Calcula pesos inversamente proporcionais ao tamanho de cada classe."""
    counts = torch.tensor(
        [dataset_train.targets.count(i) for i in range(NUM_CLASSES)],
        dtype=torch.float,
    )
    weights = (1.0 / counts)
    weights = (weights / weights.sum()).to(device)
    return nn.CrossEntropyLoss(weight=weights)


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Monitora o F1 macro e para o treino se não houver melhora."""

    def __init__(self, patience: int, min_delta: float, path: str) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.triggered = False
        self.best_data = {}

    def step(self, score: float, model: nn.Module, epoch_data: dict) -> bool:
        improved = (
                self.best_score is None
                or score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
            self.best_data = epoch_data
            torch.save(model.state_dict(), self.path)
            print(f"   ✅ Novo melhor F1 Val: {score:.4f} — modelo salvo.")
        else:
            self.counter += 1
            print(f"   ⏳ Sem melhora ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.triggered = True

        return self.triggered


# =============================================================================
# GERAÇÃO DE GRÁFICOS (Validação)
# =============================================================================

def save_plots(
        model_name: str,
        history: dict,
        y_true: list,
        y_probs_matrix: np.ndarray,
        y_preds: list,
) -> float:
    """Gera e salva os gráficos de validação. Retorna o AUC macro."""
    out = os.path.join(PLOT_DIR, model_name)
    os.makedirs(out, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Curvas de loss e acurácia
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], label="Treino")
    ax1.plot(epochs, history["val_loss"], label="Validação")
    ax1.set_title(f"Loss — {model_name}")
    ax1.set_xlabel("Época")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Treino")
    ax2.plot(epochs, history["val_acc"], label="Validação")
    ax2.set_title(f"Acurácia — {model_name}")
    ax2.set_xlabel("Época")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "learning_curves.png"))
    plt.close(fig)

    # 2. F1 por classe
    f1_matrix = np.array(history["f1_per_class"])
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, class_name in enumerate(NOME_CLASSES):
        ax.plot(epochs, f1_matrix[:, i], label=class_name, color=colors[i % len(colors)], linewidth=2, marker="o",
                markersize=4)
    f1_macro_curve = f1_matrix.mean(axis=1)
    ax.plot(epochs, f1_macro_curve, label="macro (média)", color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_title(f"F1 Validação por classe — {model_name}")
    ax.set_xlabel("Época")
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "f1_per_class.png"))
    plt.close(fig)

    # 3. Curvas ROC multiclasse
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    if NUM_CLASSES == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, class_name in enumerate(NOME_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_matrix[:, i])
        roc_auc_i = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc_i:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"Curvas ROC Validação (OvR) — {model_name}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "roc_auc_curve_val.png"))
    plt.close(fig)

    if NUM_CLASSES == 2:
        macro_auc = roc_auc_score(y_true, y_probs_matrix[:, 1])
    else:
        macro_auc = roc_auc_score(y_true, y_probs_matrix, multi_class="ovr", average="macro")

    # 4. Matriz de confusão da Validação
    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(figsize=(max(6, NUM_CLASSES * 1.4), max(5, NUM_CLASSES * 1.2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=NOME_CLASSES, yticklabels=NOME_CLASSES,
                ax=ax)
    ax.set_title(f"Matriz de Confusão Validação — {model_name}")
    ax.set_xlabel("Previsão")
    ax.set_ylabel("Rótulo real")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrix_val.png"))
    plt.close(fig)

    return macro_auc


# =============================================================================
# PROJEÇÃO DO ESPAÇO LATENTE (UMAP)
# =============================================================================

def plot_latent_space(model, model_name, test_loader):
    """Extrai embeddings do modelo e projeta em 2D usando UMAP."""
    print(f"  🌌 Gerando projeção do Espaço Latente (UMAP) para {model_name}...")

    model.eval()
    features = []
    labels_list = []

    out_dir = os.path.join(PLOT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # Tenta extrair as features antes da camada de classificação (timm models)
            try:
                feats = model.forward_features(inputs)
                # Se for matriz (B, C, H, W), faz pooling espacial para virar vetor (B, C)
                if feats.ndim == 4:
                    feats = feats.mean(dim=[-2, -1])
                    # Se for transformer (B, N, C), pega o token CLS ou faz média
                elif feats.ndim == 3:
                    feats = feats[:, 0]
            except Exception:
                # Fallback genérico: usa a saída final (logits)
                feats = model(inputs)

            features.append(feats.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    features_np = np.vstack(features)
    labels_np = np.array(labels_list)

    # Executa o UMAP
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    try:
        embedding = reducer.fit_transform(features_np)
    except Exception as e:
        print(f"  ⚠️ Erro no UMAP para {model_name}: {e}")
        return

    # Plotagem
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels_np, cmap="coolwarm", alpha=0.7, s=50,
                         edgecolors='k')

    # Cria a legenda usando os nomes reais das classes
    handles, _ = scatter.legend_elements()
    ax.legend(handles, NOME_CLASSES, title="Classes")

    ax.set_title(f"Espaço Latente (UMAP) — {model_name}")
    ax.set_xlabel("UMAP Dimensão 1")
    ax.set_ylabel("UMAP Dimensão 2")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "umap_latent_space.png"), dpi=300)
    plt.close(fig)

# =============================================================================
# GRAD-CAM (MAPA DE ATIVAÇÃO)
# =============================================================================

def get_target_layer_for_cam(model, model_name):
    """Tenta encontrar a última camada de features para o Grad-CAM dinamicamente."""


    if "mobilenetv3" in model_name:
        return [model.blocks[-1]]

    if "resnet" in model_name:
        return [model.layer4[-1]]

    if "densenet" in model_name:
        return [model.features.norm5]

    if "efficientnet" in model_name:
        return [model.blocks[-1]]

    if "convnext" in model_name:
        return [model.stages[-1].blocks[-1]]

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return [module]

    return None


def save_gradcam_samples(model, model_name, test_loader, num_samples=5):
    """Pega amostras do teste e gera heatmaps mostrando onde o modelo focou."""
    target_layers = get_target_layer_for_cam(model, model_name)

    if not target_layers:
        print(
            f"  ⚠️  Grad-CAM pulado para {model_name}: não foi possível identificar a target layer de forma automática (comum em Transformers puros).")
        return

    out_dir = os.path.join(PLOT_DIR, model_name, "gradcam")
    os.makedirs(out_dir, exist_ok=True)

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
    except Exception as e:
        print(f"  ⚠️  Erro ao inicializar Grad-CAM para {model_name}: {e}")
        return

    model.eval()
    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    n = min(num_samples, inputs.size(0))

    for i in range(n):
        input_tensor = inputs[i].unsqueeze(0)
        real_label = labels[i].item()

        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()

        targets = [ClassifierOutputTarget(pred_class)]

        try:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        except Exception:
            continue

        img_np = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img_np = np.clip(std * img_np + mean, 0, 1)

        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_np)
        ax1.set_title(f"Original (Real: {NOME_CLASSES[real_label]})")
        ax1.axis('off')

        cor_texto = "darkgreen" if real_label == pred_class else "darkred"
        ax2.imshow(cam_image)
        ax2.set_title(f"Grad-CAM (Pred: {NOME_CLASSES[pred_class]})", color=cor_texto, fontweight="bold")
        ax2.axis('off')

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sample_{i + 1}_cam.png"), bbox_inches='tight')
        plt.close(fig)

# =============================================================================
# FACTORY DE MODELOS
# =============================================================================

def build_model(model_name: str) -> nn.Module:
    """Cria o modelo pelo nome."""
    if model_name != "resnet50_radimagenet":
        return timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)

    import urllib.request
    weights_path = os.path.join(RESULTS_DIR, "RadImageNet-ResNet50_notop.pth")
    if not os.path.exists(weights_path):
        print("  Baixando pesos RadImageNet...")
        urllib.request.urlretrieve(RADIMAGENET_WEIGHTS_URL, weights_path)

    model = timm.create_model("resnet50", pretrained=False, num_classes=0)
    state = torch.load(weights_path, map_location="cpu")
    missing, _ = model.load_state_dict(state, strict=False)

    model.fc = nn.Linear(model.num_features.item(), NUM_CLASSES)
    return model


# =============================================================================
# TREINO E INFERÊNCIA
# =============================================================================

def train_model(model_name: str) -> dict:
    batch_size = BATCH_SIZE_OVERRIDE.get(model_name, BATCH_SIZE)

    print(f"\n{'=' * 55}\n  PROCESSANDO: {model_name.upper()}\n{'=' * 55}")
    if batch_size != BATCH_SIZE:
        print(f"  ⚠️  Batch reduzido para {batch_size} (limite de VRAM)")

    model_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(batch_size)

    model = build_model(model_name).to(device)
    criterion = build_criterion()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,  # O LR definido no topo do arquivo será o pico máximo
        steps_per_epoch=steps_per_epoch,
        epochs=NUM_EPOCHS,
        pct_start=0.3  # Gasta 30% do treino subindo o LR, e 70% descendo
    )

    best_weights_path = os.path.join(model_dir, "best.pth")
    early_stopping = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA, path=best_weights_path)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "f1_per_class": []}

    train_time_acumulado = 0.0

    for epoch in range(NUM_EPOCHS):
        # ── Treino ────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct = 0.0, 0

        if torch.cuda.is_available(): torch.cuda.synchronize()
        start_train_epoch = time.time()

        for inputs, labels in tqdm(train_loader, desc=f"Treino E{epoch + 1:02}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        if torch.cuda.is_available(): torch.cuda.synchronize()
        train_time_acumulado += (time.time() - start_train_epoch)

        # ── Validação ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)

                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                val_correct += (preds == labels).sum().item()

        # ── Métricas da época ─────────────────────────────────────────────────
        epoch_val_loss = val_loss / len(dataset_val)
        epoch_f1_classes = f1_score(all_labels, all_preds, average=None, zero_division=0)
        epoch_f1_macro = epoch_f1_classes.mean()

        history["train_loss"].append(train_loss / len(dataset_train))
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(train_correct / len(dataset_train))
        history["val_acc"].append(val_correct / len(dataset_val))
        history["f1_per_class"].append(epoch_f1_classes.tolist())

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"E{epoch + 1:02} | LR: {current_lr:.6f} | Val Loss: {epoch_val_loss:.4f} | Val F1-Macro: {epoch_f1_macro:.4f}")

        epoch_data = {"y_true": all_labels, "y_probs": np.vstack(all_probs), "y_preds": all_preds}
        if early_stopping.step(epoch_f1_macro, model, epoch_data):
            print(f"\n  Early stopping na época {epoch + 1} — melhor F1 Val: {early_stopping.best_score:.4f}")
            break

    # ── Plots da Validação ────────────────────────────────────────────────────
    best_val = early_stopping.best_data
    val_auc_score = save_plots(model_name, history, best_val["y_true"], best_val["y_probs"], best_val["y_preds"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # =========================================================================
    # INFERÊNCIA FINAL COM OS MELHORES PESOS (TESTE E TREINO)
    # =========================================================================
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    # ── 1. Inferência no Teste ───────────────────────────────────────────────
    print(f"\n  🔍 Extraindo métricas finais do conjunto de TESTE...")
    test_preds, test_labels, test_probs = [], [], []

    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_infer_time = time.time()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Teste {model_name}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.append(probs.cpu().numpy())

    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_infer_time = time.time() - start_infer_time
    infer_ms_per_img = (total_infer_time / len(dataset_test)) * 1000

    test_probs_matrix = np.vstack(test_probs)
    test_f1_classes = f1_score(test_labels, test_preds, average=None, zero_division=0)
    test_f1_macro = test_f1_classes.mean()
    if NUM_CLASSES == 2:
        test_auc_macro = roc_auc_score(test_labels, test_probs_matrix[:, 1])
    else:
        test_auc_macro = roc_auc_score(test_labels, test_probs_matrix, multi_class="ovr", average="macro")

    cm_test = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(max(6, NUM_CLASSES * 1.4), max(5, NUM_CLASSES * 1.2)))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=NOME_CLASSES, yticklabels=NOME_CLASSES, ax=ax)
    ax.set_title(f"Matriz de Confusão TESTE — {model_name}")
    ax.set_xlabel("Previsão")
    ax.set_ylabel("Rótulo real")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, model_name, "confusion_matrix_test.png"))
    plt.close(fig)

    # ==============================================================
    # DISTRIBUIÇÃO DE CONFIANÇA E GRAD-CAM
    # ==============================================================

    test_confidences = np.max(test_probs_matrix, axis=1)
    correct_mask = np.array(test_preds) == np.array(test_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(test_confidences[correct_mask], bins=20, color='green', label='Corretos', kde=True, alpha=0.6, ax=ax)
    sns.histplot(test_confidences[~correct_mask], bins=20, color='red', label='Incorretos', kde=True, alpha=0.6, ax=ax)

    ax.set_title(f"Distribuição de Confiança (Teste) — {model_name}")
    ax.set_xlabel("Confiança (Probabilidade da Classe Majoritária)")
    ax.set_ylabel("Frequência de Imagens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, model_name, "confidence_distribution.png"))
    plt.close(fig)

    print("  🎨 Gerando heatmaps do Grad-CAM...")
    for param in model.parameters(): param.requires_grad = True
    save_gradcam_samples(model, model_name, test_loader, num_samples=5)
    for param in model.parameters(): param.requires_grad = False

    # ── 2. Inferência no Treino ──────────────────────────────────────────────
    print(f"  🔍 Extraindo métricas finais do conjunto de TREINO...")
    train_preds, train_labels, train_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(train_loader, desc=f"Treino Final {model_name}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_probs.append(probs.cpu().numpy())

    train_probs_matrix = np.vstack(train_probs)
    train_f1_classes = f1_score(train_labels, train_preds, average=None, zero_division=0)
    train_f1_macro = train_f1_classes.mean()
    if NUM_CLASSES == 2:
        train_auc_macro = roc_auc_score(train_labels, train_probs_matrix[:, 1])
    else:
        train_auc_macro = roc_auc_score(train_labels, train_probs_matrix, multi_class="ovr", average="macro")

    print(f"  🏆 TESTE  | F1-Macro: {test_f1_macro:.4f} | AUC-Macro: {test_auc_macro:.4f}")
    print(f"  🏆 TREINO | F1-Macro: {train_f1_macro:.4f} | AUC-Macro: {train_auc_macro:.4f}")
    # 3. UMAP
    plot_latent_space(model, model_name, test_loader)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "Modelo": model_name,
        "Train_F1-Macro": train_f1_macro,
        "Train_AUC-Macro": train_auc_macro,
        "Val_F1-Macro": early_stopping.best_score,
        "Test_F1-Macro": test_f1_macro,
        "Test_AUC-Macro": test_auc_macro,
        "Train_Time_s": train_time_acumulado,
        "Inference_Time_s": total_infer_time,
        "Inference_ms_per_img": infer_ms_per_img,
        "Parâmetros": num_params,
        "Batch": batch_size,
        "test_preds": test_preds,
        "test_labels": test_labels
    }


# =============================================================================
# EXECUÇÃO E ATUALIZAÇÃO DO RANKING GLOBAL (APPEND)
# =============================================================================

if __name__ == "__main__":
    # Treina apenas os modelos que estão na lista MODELOS
    resultados_novos = []

    for nome in MODELOS:
        try:
            resultado = train_model(nome)
            resultados_novos.append(resultado)
        except Exception as e:
            print(f"\n❌ ERRO CRÍTICO ao treinar o modelo '{nome}': {e}")
            print(f"PULANDO '{nome}' e limpando a memória para o próximo modelo...\n")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

    # Verificação de segurança: se todos os modelos falharem, encerra o script sem quebrar as análises
    if not resultados_novos:
        print("\n⚠️ Nenhum modelo foi treinado com sucesso. Encerrando o script.")
        exit()

    df_novo = pd.DataFrame(resultados_novos)
    df_novo["Params_M"] = df_novo["Parâmetros"] / 1e6

    # =========================================================================
    # ACORDO INTER-MODELO E ANÁLISE DE ERROS (Usa os dados recém-treinados)
    # =========================================================================
    print("\n" + "=" * 85)
    print("GERANDO ANÁLISE INTER-MODELO (Acordo e Significância)")
    print("=" * 85)

    preds_dict = {res["Modelo"]: res["test_preds"] for res in resultados_novos}
    true_labels = np.array(resultados_novos[0]["test_labels"])
    nomes_modelos = list(preds_dict.keys())

    if len(nomes_modelos) >= 2:
        acordo_matrix = np.zeros((len(nomes_modelos), len(nomes_modelos)))
        for i, mod1 in enumerate(nomes_modelos):
            for j, mod2 in enumerate(nomes_modelos):
                p1 = np.array(preds_dict[mod1])
                p2 = np.array(preds_dict[mod2])
                acordo = np.mean(p1 == p2)
                acordo_matrix[i, j] = acordo

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(acordo_matrix, annot=True, fmt=".2%", cmap="Purples",
                    xticklabels=nomes_modelos, yticklabels=nomes_modelos, ax=ax)
        ax.set_title("Matriz de Acordo Inter-Modelo (Predições Idênticas)")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "inter_model_agreement.png"), dpi=300)
        plt.close(fig)

        print("\n📊 Teste Estatístico de McNemar (Top 1 vs Outros):")
        top_model = df_novo.sort_values("Test_F1-Macro", ascending=False).iloc[0]["Modelo"]
        p1 = np.array(preds_dict[top_model])

        for mod2 in nomes_modelos:
            if mod2 == top_model: continue
            p2 = np.array(preds_dict[mod2])

            ambos_acertam = np.sum((p1 == true_labels) & (p2 == true_labels))
            p1_acerta_p2_erra = np.sum((p1 == true_labels) & (p2 != true_labels))
            p1_erra_p2_acerta = np.sum((p1 != true_labels) & (p2 == true_labels))
            ambos_erram = np.sum((p1 != true_labels) & (p2 != true_labels))

            table = [[ambos_acertam, p1_acerta_p2_erra],
                     [p1_erra_p2_acerta, ambos_erram]]

            result = mcnemar(table, exact=True)
            alpha = 0.05
            if result.pvalue < alpha:
                print(f"  ✅ {top_model} vs {mod2}: Diferença SIGNIFICATIVA (p={result.pvalue:.4f})")
            else:
                print(f"  ⚖️  {top_model} vs {mod2}: Empate Estatístico (p={result.pvalue:.4f})")

        # 3. IMAGENS DIFÍCEIS (Erros Consensuais)
        todas_preds = np.array([preds_dict[m] for m in nomes_modelos])
        acertos = (todas_preds == true_labels)
        erros_totais_idx = np.where(np.sum(acertos, axis=0) == 0)[0]

        print(f"\n🔥 Análise de Falha Genuína:")
        print(f"  Ocorreram {len(erros_totais_idx)} imagens de teste que TODOS os modelos erraram.")
        if len(erros_totais_idx) > 0:
            print("  Índices dessas imagens no Dataset de Teste:", erros_totais_idx)

    # =========================================================================
    # SALVAR CSV E GERAR GRÁFICOS GLOBAIS
    # =========================================================================

    df_novo_limpo = df_novo.drop(columns=["test_preds", "test_labels"])

    csv_path = os.path.join(PLOT_DIR, "benchmark_results_test_train.csv")

    if os.path.exists(csv_path):
        print(f"\nAtualizando o arquivo existente: {csv_path}")
        df_antigo = pd.read_csv(csv_path)

        # Remove modelos que acabaram de ser treinados para evitar duplicatas
        modelos_treinados_agora = df_novo_limpo["Modelo"].tolist()
        df_antigo = df_antigo[~df_antigo["Modelo"].isin(modelos_treinados_agora)]

        df_final = pd.concat([df_antigo, df_novo_limpo], ignore_index=True)
    else:
        df_final = df_novo_limpo

    # Ranking final baseado no Teste atualizado
    df_final = df_final.sort_values("Test_F1-Macro", ascending=False)
    df_final.to_csv(csv_path, index=False)

    print("\n" + "=" * 85)
    print("RANKING FINAL ATUALIZADO (MÉTRICAS DE TREINO, VALIDAÇÃO E TESTE)")
    print("=" * 85)
    colunas_exibicao = ["Modelo", "Train_F1-Macro", "Val_F1-Macro", "Test_F1-Macro", "Test_AUC-Macro"]
    print(df_final[colunas_exibicao].to_string(index=False))

    # ── Ranking por F1 (Teste) ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=df_final, x="Test_F1-Macro", y="Modelo", hue="Modelo", palette="viridis", legend=False, ax=ax)
    ax.set_title("Ranking Global — F1-Score Macro (CONJUNTO DE TESTE)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "ranking_f1_test.png"), dpi=300)
    plt.close(fig)

    # ── Eficiência: tamanho vs performance no Teste ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df_final, x="Params_M", y="Test_F1-Macro", hue="Modelo",
                    s=200, palette="tab20", legend=False, ax=ax)
    for _, row in df_final.iterrows():
        ax.text(row["Params_M"] + 0.5, row["Test_F1-Macro"], row["Modelo"], fontsize=9)
    ax.set_title("Eficiência (Teste) — Parâmetros vs F1")
    ax.set_xlabel("Parâmetros (M)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eficiencia_test.png"), dpi=300)
    plt.close(fig)

    # ── Trade-off: Velocidade vs F1 (Teste) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df_final, x="Inference_ms_per_img", y="Test_F1-Macro",
                    size="Params_M", sizes=(50, 800), hue="Modelo",
                    alpha=0.7, palette="tab20", legend=False, ax=ax)
    for _, row in df_final.iterrows():
        ax.text(row["Inference_ms_per_img"] * 1.02, row["Test_F1-Macro"], row["Modelo"], fontsize=9)
    ax.set_title("Trade-off de Produção: Velocidade vs F1 (Bolhas = Tamanho do Modelo)")
    ax.set_xlabel("Tempo de Inferência por Imagem (ms)")
    ax.set_ylabel("F1-Score Macro (Teste)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "velocidade_vs_performance.png"), dpi=300)
    plt.close(fig)

    # ── Análise de Overfitting: F1 Treino vs Teste ────────────────────────────────
    df_melted = df_final.melt(id_vars=["Modelo"],
                              value_vars=["Train_F1-Macro", "Test_F1-Macro"],
                              var_name="Conjunto", value_name="F1-Score")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(data=df_melted, x="F1-Score", y="Modelo", hue="Conjunto", palette="Set1", ax=ax)
    ax.set_title("Análise de Overfitting (Treino vs Teste)")
    ax.set_xlabel("F1-Score Macro")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "overfitting_analysis.png"), dpi=300)
    plt.close(fig)

    # ── Tempo Total de Treinamento ────────────────────────────────────────────────
    df_final["Train_Time_min"] = df_final["Train_Time_s"] / 60.0
    df_final_sorted_time = df_final.sort_values("Train_Time_min", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=df_final_sorted_time, x="Train_Time_min", y="Modelo", palette="rocket", ax=ax)
    ax.set_title("Custo de Treinamento: Tempo Acumulado na GPU")
    ax.set_xlabel("Tempo de Treino (Minutos)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "tempo_treinamento.png"), dpi=300)
    plt.close(fig)

    print(f"\nTreino e atualização concluídos. Relatórios atualizados salvos em: {PLOT_DIR}/")