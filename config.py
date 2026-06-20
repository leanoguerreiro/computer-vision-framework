"""Configurações centralizadas do projeto."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
ROBUSTNESS_DIR = PROJECT_ROOT / "robustness_analysis"
STANDARD_PLOTS_DIR = PROJECT_ROOT / "plots_standart"
STANDARD_RESULTS_DIR = PROJECT_ROOT / "results_standart"
STANDARD_ROBUSTNESS_DIR = PROJECT_ROOT / "robustness_analysis_standart"

DEFAULT_SEED = 42
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-3
DEFAULT_EARLY_STOPPING_PATIENCE = 2
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.001
DEFAULT_TRAIN_WORKERS = 8
DEFAULT_EVAL_WORKERS = 4
DEFAULT_IMAGE_FOLDER_WORKERS = 8
DEFAULT_VISUALIZATION_WORKERS = 4
DEFAULT_MODEL_NAME = "densenet201"
DEFAULT_BATCH_SIZE_TRAINING = 32
DEFAULT_BATCH_SIZE_ROBUSTNESS = 16

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BENCHMARK_DATASET_ROOT = DATASETS_DIR / "MC_ALL_split_70_20_10"
TRAINING_DATASET_ROOT = DATASETS_DIR / "MC_ALL_split_70_20_10"
ROBUSTNESS_DATASET_ROOT = DATASETS_DIR / "MC_ALL_split_70_20_10"

BENCHMARK_PLOT_DIR = PLOTS_DIR / BENCHMARK_DATASET_ROOT.name
BENCHMARK_RESULTS_DIR = RESULTS_DIR / BENCHMARK_DATASET_ROOT.name
TRAINING_PLOT_DIR = STANDARD_PLOTS_DIR / TRAINING_DATASET_ROOT.name
TRAINING_RESULTS_DIR = STANDARD_RESULTS_DIR / TRAINING_DATASET_ROOT.name
TRAINING_CSV_PATH = TRAINING_RESULTS_DIR / "training_metrics.csv"
ROBUSTNESS_PLOT_DIR = ROBUSTNESS_DIR / ROBUSTNESS_DATASET_ROOT.name
ROBUSTNESS_RESULTS_DIR = RESULTS_DIR / ROBUSTNESS_DATASET_ROOT.name
ROBUSTNESS_CSV_PATH = ROBUSTNESS_PLOT_DIR / "robustness_metrics_levels.csv"
VISUALIZATION_ROBUSTNESS_DIR = STANDARD_ROBUSTNESS_DIR / TRAINING_DATASET_ROOT.name

RADIMAGENET_WEIGHTS_URL = (
    "https://huggingface.co/BMEII/RadImageNet/resolve/main/"
    "RadImageNet-ResNet50_notop.pth"
)

BENCHMARK_MODELS = (
    "multicancernet_attention_hybrid",
    "multicancernet_attention",
    "densenet201",
    "mobilevit_s",
)

BENCHMARK_BATCH_SIZE_OVERRIDES = {
    "convnext_base": 32,
    "efficientnetv2_m": 32,
}

ROBUSTNESS_MODELS = (
    "densenet201",
    "multicancernet_attention_hybrid",
    "multicancernet_attention",
    "mobilevit_s",
    # "mobilenetv3_large_100",
    # "mobilevit_s",
    # "resnet50",
    # "convformer_s18",
    # "efficientnet_b3",
    # "maxvit_tiny_tf_224",
    # "swin_tiny_patch4_window7_224",
    # "vit_base_patch16_224",
    # "convnext_base",
    # "swin_base_patch4_window7_224",
)


def output_dir(base_dir: Path, dataset_root: Path) -> Path:
    """Cria o diretório de saída padronizado para um dataset."""
    return base_dir / dataset_root.name

