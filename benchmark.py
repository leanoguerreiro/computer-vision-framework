import os
import gc
import random

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
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================

PASTA_RAIZ = "mri_split_70_20_10"
PLOT_DIR = "plots"
RESULTS_DIR = "results"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-4
ES_PATIENCE = 5
ES_MIN_DELTA = 0.001

BATCH_SIZE_OVERRIDE = {
    "convnext_base": 16,
    "efficientnetv2_m": 16,
}

MODELOS = [
    # ── originais ──────────────────────────────────────────────────────────────
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "resnet18",
    "resnet50",
    "efficientnet_b3",
    "convnext_small",
    "mobilevit_s",
    "fastvit_t8",
    "tiny_vit_11m_224",
    "vit_small_patch16_224",
    "swin_tiny_patch4_window7_224",
    "vit_base_patch16_224",
    # ── leves ──────────────────────────────────────────────────────────────────
    "efficientnet_b1",
    "efficientnet_b2",
    "mobilenetv3_small_100",
    "ghostnet_100",
    # ── médios ─────────────────────────────────────────────────────────────────
    "resnet34",
    "densenet121",
    "convnext_tiny",
    "swin_s3_tiny_224",
    "xcit_small_12_p16_224",
    "vit_base_patch32_224",
    # ── médico: densenet169 com pesos ImageNet ─────────────────────────────────
    "densenet169"
]

# URL dos pesos RadImageNet (resnet50)
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


transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# =============================================================================
# DATASETS E DATALOADERS
# =============================================================================

dataset_train = datasets.ImageFolder(os.path.join(PASTA_RAIZ, "train"), transform=transform)
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

    macro_auc = roc_auc_score(y_true_bin, y_probs_matrix, multi_class="ovr", average="macro")

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
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_weights_path = os.path.join(model_dir, "best.pth")
    early_stopping = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA, path=best_weights_path)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "f1_per_class": []}

    for epoch in range(NUM_EPOCHS):
        # ── Treino ────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct = 0.0, 0

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
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Teste {model_name}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.append(probs.cpu().numpy())

    test_probs_matrix = np.vstack(test_probs)
    test_f1_classes = f1_score(test_labels, test_preds, average=None, zero_division=0)
    test_f1_macro = test_f1_classes.mean()
    test_true_bin = label_binarize(test_labels, classes=list(range(NUM_CLASSES)))
    test_auc_macro = roc_auc_score(test_true_bin, test_probs_matrix, multi_class="ovr", average="macro")

    # Salvar matriz de confusão exclusiva do Teste
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
    train_true_bin = label_binarize(train_labels, classes=list(range(NUM_CLASSES)))
    train_auc_macro = roc_auc_score(train_true_bin, train_probs_matrix, multi_class="ovr", average="macro")

    print(f"  🏆 TESTE  | F1-Macro: {test_f1_macro:.4f} | AUC-Macro: {test_auc_macro:.4f}")
    print(f"  🏆 TREINO | F1-Macro: {train_f1_macro:.4f} | AUC-Macro: {train_auc_macro:.4f}")

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
        "Parâmetros": num_params,
        "Batch": batch_size,
    }


# =============================================================================
# EXECUÇÃO E ATUALIZAÇÃO DO RANKING GLOBAL (APPEND)
# =============================================================================

if __name__ == "__main__":
    # Treina apenas os modelos que estão na lista MODELOS (neste caso, só a SNN)
    resultados_novos = [train_model(nome) for nome in MODELOS]

    df_novo = pd.DataFrame(resultados_novos)
    df_novo["Params_M"] = df_novo["Parâmetros"] / 1e6

    csv_path = os.path.join(PLOT_DIR, "benchmark_results_test_train.csv")

    # Se o CSV antigo já existir, nós unimos os dados
    if os.path.exists(csv_path):
        print(f"\nAtualizando o arquivo existente: {csv_path}")
        df_antigo = pd.read_csv(csv_path)

        # Remove o modelo atual caso ele já esteja no CSV (evita duplicatas se você rodar 2 vezes)
        modelos_treinados_agora = df_novo["Modelo"].tolist()
        df_antigo = df_antigo[~df_antigo["Modelo"].isin(modelos_treinados_agora)]

        # Junta o resultado antigo com o novo
        df_final = pd.concat([df_antigo, df_novo], ignore_index=True)
    else:
        df_final = df_novo

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

    print(f"\nTreino e atualização concluídos. Relatórios atualizados salvos em: {PLOT_DIR}/")