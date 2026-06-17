import gc
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    ROBUSTNESS_CSV_PATH,
    ROBUSTNESS_DATASET_ROOT,
    ROBUSTNESS_PLOT_DIR,
    ROBUSTNESS_RESULTS_DIR,
)
from cv_framework.models import build_model as build_shared_model
from cv_framework.reproducibility import set_seed
from cv_framework.transforms import build_perturbation_transforms

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class RobustnessConfig:
    """Configurações centralizadas para o teste de robustez."""
    model: str = "resnet18"
    batch_size: int = DEFAULT_BATCH_SIZE
    seed: int = DEFAULT_SEED


class RobustnessEvaluator:
    """Orquestra a avaliação de um modelo treinado sob diferentes níveis de degradação."""

    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(self.config.seed)

        # Definição e criação de caminhos estruturados usando pathlib
        self.root_dir = Path(ROBUSTNESS_DATASET_ROOT)
        self.results_dir = Path(ROBUSTNESS_RESULTS_DIR)
        self.plot_dir = Path(ROBUSTNESS_PLOT_DIR)
        self.csv_path = Path(ROBUSTNESS_CSV_PATH)

        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.perturbation_transforms = build_perturbation_transforms()
        self._prepare_dataloaders()

    def _prepare_dataloaders(self):
        print("Carregando datasets de teste perturbados...")
        test_dir = self.root_dir / "test"

        # Extração das classes
        dummy_dataset = datasets.ImageFolder(str(test_dir))
        self.class_names = dummy_dataset.classes
        self.num_classes = len(self.class_names)
        print(f"Classes detectadas ({self.num_classes}): {self.class_names}")

        # Construção do dicionário de DataLoaders para cada perturbação
        self.test_loaders = {}
        for pert_name, trans in self.perturbation_transforms.items():
            ds_test = datasets.ImageFolder(str(test_dir), transform=trans)
            self.test_loaders[pert_name] = DataLoader(
                ds_test,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4
            )

    def evaluate(self) -> dict:
        weights_path = self.results_dir / self.config.model / "best.pth"

        if not weights_path.exists():
            raise FileNotFoundError(f"Pesos não encontrados para {self.config.model} em: {weights_path}")

        print(f"\n{'=' * 60}\n  AVALIANDO ROBUSTEZ: {self.config.model.upper()}\n{'=' * 60}")

        model = build_shared_model(self.config.model, self.num_classes, pretrained=False).to(self.device)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()

        resultados_modelo = {
            "Data_Hora": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Modelo": self.config.model
        }

        for pert_name, loader in self.test_loaders.items():
            preds_list, labels_list, probs_list = [], [], []

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_infer = time.time()

            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc=f"Inferência [{pert_name.ljust(15)}]", leave=False):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    preds_list.extend(preds.cpu().numpy())
                    labels_list.extend(labels.cpu().numpy())
                    probs_list.append(probs.cpu().numpy())

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_infer_time = time.time() - start_infer
            infer_ms_per_img = (total_infer_time / len(loader.dataset)) * 1000

            probs_matrix = np.vstack(probs_list)
            f1_macro = f1_score(labels_list, preds_list, average="macro", zero_division=0)

            if self.num_classes == 2:
                auc_macro = roc_auc_score(labels_list, probs_matrix[:, 1])
            else:
                true_bin = label_binarize(labels_list, classes=list(range(self.num_classes)))
                auc_macro = roc_auc_score(true_bin, probs_matrix, multi_class="ovr", average="macro")

            resultados_modelo[f"F1_{pert_name}"] = f1_macro
            resultados_modelo[f"AUC_{pert_name}"] = auc_macro
            resultados_modelo[f"Time_ms_img_{pert_name}"] = infer_ms_per_img

            print(
                f"  ➔ {pert_name.ljust(17)}: F1-Macro = {f1_macro:.4f} | AUC-Macro = {auc_macro:.4f} | Tempo/img = {infer_ms_per_img:.2f} ms")

        # Limpeza de memória
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return resultados_modelo

    def plot_degradation(self, resultados: dict):
        """Gera um gráfico focado na degradação frente a diferentes ruídos."""
        data = []
        baseline = resultados.get("F1_Clean", 0)

        tipos_pertubacao = ["Noise", "Blur", "Contrast"]
        niveis = ["Leve", "Moderada", "Extrema"]

        for tipo in tipos_pertubacao:
            data.append({"Perturbação": tipo, "Severidade": "Limpo (Baseline)", "F1-Macro": baseline})
            for nivel in niveis:
                chave = f"F1_{tipo}_{nivel}"
                if chave in resultados:
                    data.append({"Perturbação": tipo, "Severidade": nivel, "F1-Macro": resultados[chave]})

        df_plot = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=df_plot,
            x="Perturbação",
            y="F1-Macro",
            hue="Severidade",
            palette="viridis",
            ax=ax
        )

        ax.set_title(f"Análise de Robustez (Degradação) — {self.config.model.upper()}", fontsize=14, pad=15,
                     fontweight='bold')
        ax.set_ylabel("F1-Score Macro", fontsize=12)
        ax.set_xlabel("Tipo de Alteração", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(title="Nível de Severidade", bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        fig.tight_layout()
        plot_path = self.plot_dir / f"robustness_degradation_{self.config.model}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"📊 Gráfico de degradação salvo em: {plot_path}")

    def save_results(self, resultado: dict):
        df_novo = pd.DataFrame([resultado])

        if self.csv_path.exists():
            df_existente = pd.read_csv(self.csv_path)
            df_existente = df_existente[df_existente["Modelo"] != self.config.model]
            df_final = pd.concat([df_existente, df_novo], ignore_index=True)
        else:
            df_final = df_novo

        # Reordena para manter os melhores no topo baseado na imagem limpa
        df_final = df_final.sort_values("F1_Clean", ascending=False)
        df_final.to_csv(self.csv_path, index=False)

        print("\n" + "=" * 80)
        print(f"RESUMO SALVO EM: {self.csv_path}")
        print("=" * 80)
        colunas_exibicao = ["Modelo", "F1_Clean", "AUC_Clean", "Time_ms_img_Clean"]
        print(df_novo[colunas_exibicao].to_string(index=False))


if __name__ == "__main__":
    config = RobustnessConfig(
        model="resnet18",
        batch_size=32
    )

    try:
        evaluator = RobustnessEvaluator(config)
        resultado = evaluator.evaluate()
        evaluator.save_results(resultado)
        evaluator.plot_degradation(resultado)
        print("\n✅ Análise finalizada!")

    except Exception as e:
        print(f"\n❌ Falha durante a avaliação do modelo: {e}")