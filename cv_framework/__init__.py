"""Componentes reutilizáveis do projeto de visão computacional."""

from .context import BenchmarkContext
from .data import build_class_weights, build_imagefolder_datasets, get_class_names, setup_data_loaders
from .explainability import (
    create_heatmap_overlay,
    extract_attention_map,
    generate_gradcam_samples,
    generate_transformer_samples,
    get_target_layer_for_cam,
    plot_latent_space,
)
from .inter_model import calculate_inter_model_metrics
from .io import update_results_csv
from .metrics import calculate_auc, evaluate_split, run_inference
from .models import build_model
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
from .reproducibility import set_seed
from .training import init_early_stopping, step_early_stopping, train_model_pipeline
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
from .kfold import get_kfold_indices, get_kfold_subsets

__all__ = [
    "get_kfold_indices",
    "get_kfold_subsets",
    "BenchmarkContext",
    "add_gaussian_noise",
    "apply_contrast",
    "build_class_weights",
    "build_eval_transform",
    "build_imagefolder_datasets",
    "build_model",
    "build_perturbation_transforms",
    "build_train_transform",
    "build_visualization_perturbation_transforms",
    "calculate_auc",
    "calculate_inter_model_metrics",
    "create_heatmap_overlay",
    "evaluate_split",
    "extract_attention_map",
    "generate_global_reports",
    "generate_gradcam_samples",
    "generate_model_plots",
    "generate_transformer_samples",
    "get_class_names",
    "get_target_layer_for_cam",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGE_SIZE",
    "init_early_stopping",
    "plot_confidence_distribution",
    "plot_confusion_matrix",
    "plot_f1_per_class",
    "plot_inter_model_agreement",
    "plot_latent_space",
    "plot_learning_curves",
    "plot_roc_auc_curve",
    "run_inference",
    "set_seed",
    "setup_data_loaders",
    "square_pad",
    "step_early_stopping",
    "train_model_pipeline",
    "update_results_csv",
]