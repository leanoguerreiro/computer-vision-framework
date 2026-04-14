import os
import gc
import random
import time

import numpy as np
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
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================
INPUT_DIR = "datasets"
PASTA_RAIZ = f"{INPUT_DIR}/mri_split_70_20_10"
PLOT_DIR = f"plots_standart/{os.path.basename(PASTA_RAIZ)}"
RESULTS_DIR = f"results_standart/{os.path.basename(PASTA_RAIZ)}"
CSV_PATH = os.path.join(RESULTS_DIR, "training_metrics.csv")
# Configurações de Treino
MODEL_NAME = "resnet18"  # Escolha apenas UM modelo aqui
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-3
ES_PATIENCE = 5
ES_MIN_DELTA = 0.001

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
# FUNÇÃO DE PERDA COM PESOS
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
        improved = (self.best_score is None or score > self.best_score + self.min_delta)
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

def save_plots(model_name: str, history: dict, y_true: list, y_probs_matrix: np.ndarray, y_preds: list) -> float:
    """Gera e salva os gráficos de validação."""
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
    features, labels_list = [], []
    out_dir = os.path.join(PLOT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

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

    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    try:
        embedding = reducer.fit_transform(features_np)
    except Exception as e:
        print(f"  ⚠️ Erro no UMAP para {model_name}: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels_np, cmap="coolwarm", alpha=0.7, s=50,
                         edgecolors='k')
    handles, _ = scatter.legend_elements()
    ax.legend(handles, NOME_CLASSES, title="Classes")
    ax.set_title(f"Espaço Latente (UMAP) — {model_name}")
    ax.set_xlabel("UMAP Dimensão 1")
    ax.set_ylabel("UMAP Dimensão 2")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "umap_latent_space.png"), dpi=300)
    plt.close(fig)


# =============================================================================
# GRAD-CAM
# =============================================================================

def get_target_layer_for_cam(model, model_name):
    if "mobilenetv3" in model_name: return [model.blocks[-1]]
    if "resnet" in model_name: return [model.layer4[-1]]
    if "densenet" in model_name: return [model.features.norm5]
    if "efficientnet" in model_name: return [model.blocks[-1]]
    if "convnext" in model_name: return [model.stages[-1].blocks[-1]]
    if "convformer" in model_name: return [model.stages[-1].blocks[-1]]

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return [module]
    return None


def save_gradcam_samples(model, model_name, test_loader, num_samples=5):
    target_layers = get_target_layer_for_cam(model, model_name)
    if not target_layers:
        print(f"  ⚠️  Grad-CAM pulado para {model_name}: target layer não identificada.")
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

def train_single_model(model_name: str):
    print(f"\n{'=' * 55}\n  INICIANDO TREINAMENTO: {model_name.upper()}\n{'=' * 55}")

    model_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(BATCH_SIZE)

    model = build_model(model_name).to(device)
    criterion = build_criterion()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

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

        scheduler.step(epoch_val_loss)

        print(f"E{epoch + 1:02} | Val Loss: {epoch_val_loss:.4f} | Val F1-Macro: {epoch_f1_macro:.4f}")

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
        for inputs, labels in tqdm(test_loader, desc="Avaliando Teste", leave=False):
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
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", cbar=False, xticklabels=NOME_CLASSES,
                yticklabels=NOME_CLASSES, ax=ax)
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
        for inputs, labels in tqdm(train_loader, desc="Avaliando Treino", leave=False):
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

    print(f"\n  🏆 TESTE  | F1-Macro: {test_f1_macro:.4f} | AUC-Macro: {test_auc_macro:.4f}")
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
        "Parâmetros": f"{num_params / 1e6:.2f} M",
    }


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    try:
        resultado = train_single_model(MODEL_NAME)
        df_novo = pd.DataFrame([resultado])

        # Lógica de salvamento em CSV (Smart Append)
        if os.path.exists(CSV_PATH):
            df_existente = pd.read_csv(CSV_PATH)
            # Remove entrada anterior do mesmo modelo para não duplicar no histórico
            df_existente = df_existente[df_existente["Modelo"] != MODEL_NAME]
            df_final = pd.concat([df_existente, df_novo], ignore_index=True)
        else:
            df_final = df_novo

        df_final.to_csv(CSV_PATH, index=False)

        print("\n" + "=" * 55)
        print(f"  RESULTADOS SALVOS EM: {CSV_PATH}")
        print("=" * 55)
        print(df_novo.to_string(index=False))

    except Exception as e:
        print(f"\n❌ Falha no processo: {e}")