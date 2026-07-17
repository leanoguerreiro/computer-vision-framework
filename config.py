"""Configurações centralizadas do projeto de visão computacional."""

from __future__ import annotations
from pathlib import Path

# =====================================================================
# 1. DIRETÓRIOS E CAMINHOS BASE
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
ROBUSTNESS_DIR = PROJECT_ROOT / "robustness_analysis"

# Diretórios padrão (Standard)
STANDARD_PLOTS_DIR = PROJECT_ROOT / "plots_standard"
STANDARD_RESULTS_DIR = PROJECT_ROOT / "results_standard"
STANDARD_ROBUSTNESS_DIR = PROJECT_ROOT / "robustness_analysis_standard"


# =====================================================================
# 2. CONFIGURAÇÕES DO DATASET E NORMALIZAÇÃO
# =====================================================================
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Dataset principal (Fonte única da verdade)
BENCHMARK_DATASET_ROOT = DATASETS_DIR / "MC_ALL_split_70_20_10"
TRAINING_DATASET_ROOT = BENCHMARK_DATASET_ROOT
ROBUSTNESS_DATASET_ROOT = BENCHMARK_DATASET_ROOT

# Subdiretórios derivados do Dataset
BENCHMARK_PLOT_DIR = PLOTS_DIR / BENCHMARK_DATASET_ROOT.name
BENCHMARK_RESULTS_DIR = RESULTS_DIR / BENCHMARK_DATASET_ROOT.name

TRAINING_PLOT_DIR = STANDARD_PLOTS_DIR / TRAINING_DATASET_ROOT.name
TRAINING_RESULTS_DIR = STANDARD_RESULTS_DIR / TRAINING_DATASET_ROOT.name
TRAINING_CSV_PATH = TRAINING_RESULTS_DIR / "training_metrics.csv"

ROBUSTNESS_PLOT_DIR = ROBUSTNESS_DIR / ROBUSTNESS_DATASET_ROOT.name
ROBUSTNESS_RESULTS_DIR = RESULTS_DIR / ROBUSTNESS_DATASET_ROOT.name
ROBUSTNESS_CSV_PATH = ROBUSTNESS_PLOT_DIR / "robustness_metrics_levels.csv"
VISUALIZATION_ROBUSTNESS_DIR = STANDARD_ROBUSTNESS_DIR / TRAINING_DATASET_ROOT.name


# =====================================================================
# 3. HIPERPARÂMETROS DE TREINAMENTO (DEFAULTS)
# =====================================================================
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 50
DEFAULT_LR = 3e-3
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_SCHEDULER_PATIENCE = 2
DEFAULT_SCHEDULER_MIN_LR = 1e-6
DEFAULT_BATCH_SIZE = 32
DEFAULT_BATCH_SIZE_TRAINING = DEFAULT_BATCH_SIZE
DEFAULT_BATCH_SIZE_ROBUSTNESS = 16

# Early Stopping
DEFAULT_EARLY_STOPPING_PATIENCE = 7
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.001


# =====================================================================
# 4. HARDWARE E PERFORMANCE (WORKERS)
# =====================================================================
DEFAULT_TRAIN_WORKERS = 8
DEFAULT_IMAGE_FOLDER_WORKERS = DEFAULT_TRAIN_WORKERS
DEFAULT_EVAL_WORKERS = 4
DEFAULT_VISUALIZATION_WORKERS = 4


# =====================================================================
# 5. REGISTRO DE MODELOS E PESOS
# =====================================================================
DEFAULT_MODEL_NAME = "convformer_s18"

RADIMAGENET_WEIGHTS_URL = (
    "https://huggingface.co/BMEII/RadImageNet/resolve/main/"
    "RadImageNet-ResNet50_notop.pth"
)

BENCHMARK_MODELS = (

    "cbam_attention",
    "cbam_attention_hybrid",

    # Transformers Puros
    "dinov3",
    "dinov2",


    # Híbridos Universais
    "dinov2_mobilenet_hybrid",
    "dinov3_mobilenet_hybrid",
    "dinov2_efficientnet_hybrid",
    "dinov3_efficientnet_hybrid",



    "mobilevit_s",
    "mobilenetv3_large_100",

    "efficientnet_b3",
    # "dino_efficientnet_hybrid",
    "densenet201",

    "resnet50",
    "convnext_tiny",
    "swin_tiny_patch4_window7_224",
    "convformer_s18",

    "vit_base_patch16_224",
    "dinov2",
)

# Reaproveita a mesma tupla para evitar divergências na manutenção
ROBUSTNESS_MODELS = BENCHMARK_MODELS

BENCHMARK_BATCH_SIZE_OVERRIDES = {
    "convnext_base": 32,
    "efficientnetv2_m": 32,
}


# =====================================================================
# 6. UTILITÁRIOS
# =====================================================================
def output_dir(base_dir: Path, dataset_root: Path) -> Path:
    """Cria e retorna o diretório de saída padronizado para um dataset."""
    path = base_dir / dataset_root.name
    path.mkdir(parents=True, exist_ok=True) # se quiser garantir a criação
    return path