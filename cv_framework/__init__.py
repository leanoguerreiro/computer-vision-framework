"""API pública e componentes reutilizáveis do framework de visão computacional (cv_framework)."""

from __future__ import annotations

# --- 1. Contexto e Reprodutibilidade ---
from .context import BenchmarkContext
from .reproducibility import set_seed

# --- 2. Dados, Augmentation e K-Fold ---
from .data import (
    build_class_weights,
    build_imagefolder_datasets,
    get_class_names,
    setup_data_loaders,
)
from .kfold import get_stratified_kfold_datasets
from .transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    add_gaussian_noise,
    apply_contrast,
    build_eval_transform,
    build_perturbation_transforms,
    build_train_transform,
    build_visualization_perturbation_transforms,
    square_pad,
)

# --- 3. Modelos e Arquiteturas ---
from .models import build_model

# --- 4. Motores de Treinamento, Validação e Inferência ---
from .training import (
    init_early_stopping,
    step_early_stopping,
    train_kfold_fold,
    train_model_pipeline,
)
from .metrics import (
    calculate_auc,
    evaluate_split,
    run_inference,
)

# --- 5. Interpretabilidade e Explicabilidade (XAI) ---
from .explainability import (
    extract_attention_map,
    generate_gradcam_samples,
    generate_transformer_samples,
    get_target_layer_for_cam,
    plot_latent_space,
)

# --- 6. Visualização e Relatórios Gráficos ---
from .plots import (
    generate_global_reports,
    generate_model_plots,
    plot_confidence_distribution,
    plot_confusion_matrix,
    plot_f1_per_class,
    plot_inter_model_agreement,
    plot_learning_curves,
    plot_roc_auc_curve,
)

# --- 7. Estatística Inter-Modelo e I/O ---
from .inter_model import calculate_inter_model_metrics
from .io import update_results_csv


# Lista pública de símbolos exportados (organizada por domínio conceitual)
__all__ = [
    # Contexto e Reprodutibilidade
    "BenchmarkContext",
    "set_seed",

    # Dados, Augmentation e K-Fold
    "build_class_weights",
    "build_imagefolder_datasets",
    "get_class_names",
    "setup_data_loaders",
    "get_stratified_kfold_datasets",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGE_SIZE",
    "add_gaussian_noise",
    "apply_contrast",
    "build_eval_transform",
    "build_perturbation_transforms",
    "build_train_transform",
    "build_visualization_perturbation_transforms",
    "square_pad",

    # Modelos
    "build_model",

    # Treinamento, Validação e Inferência
    "init_early_stopping",
    "step_early_stopping",
    "train_kfold_fold",
    "train_model_pipeline",
    "calculate_auc",
    "evaluate_split",
    "run_inference",

    # Explicabilidade
    "extract_attention_map",
    "generate_gradcam_samples",
    "generate_transformer_samples",
    "get_target_layer_for_cam",
    "plot_latent_space",

    # Gráficos
    "generate_global_reports",
    "generate_model_plots",
    "plot_confidence_distribution",
    "plot_confusion_matrix",
    "plot_f1_per_class",
    "plot_inter_model_agreement",
    "plot_learning_curves",
    "plot_roc_auc_curve",

    # Estatística e IO
    "calculate_inter_model_metrics",
    "update_results_csv",
]