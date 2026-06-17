# Brain Tumor Classification Benchmark

Projeto de visão computacional voltado à **classificação de imagens com deep learning**, com foco principal em **MRI cerebral**.

O repositório reúne um pipeline completo para:

- organizar e dividir datasets em `train/`, `val/` e `test/`;
- balancear o conjunto de treino com *data augmentation*;
- treinar e comparar dezenas de arquiteturas `timm`;
- registrar métricas, curvas e matrizes de confusão;
- avaliar a robustez dos melhores modelos sob ruído, desfoque e variação de contraste.

---

## Visão geral do projeto

O fluxo principal do projeto é a classificação das quatro classes de MRI presentes em `mri_split_70_20_10/`:

- `glioma`
- `healthy`
- `meningioma`
- `pituitary`

### Estrutura principal

- `cv_framework/`: pacote interno com transforms, dataloaders, modelos, seed e early stopping reutilizáveis.
- `config.py`: centraliza hiperparâmetros, seeds, caminhos e listas de modelos usados pelos scripts.
- `data/mri/`: dados brutos do conjunto de MRI.
- `mri_split_70_20_10/`: particionamento já preparado em treino, validação e teste.
- `plots/`: gráficos e CSV do benchmark principal.
- `results/`: pesos treinados (`best.pth`) e artefatos por modelo.
- `robustness_analysis/`: métricas e gráficos de robustez.

---

## Scripts principais

Os arquivos na raiz continuam funcionando como pontos de entrada, mas agora reutilizam funções do pacote `cv_framework/`.

### `split_data.py`
Cria a divisão estratificada por classe em **70% treino / 20% validação / 10% teste** a partir de `data/mri/`.

### `augment_data.py`
Balanceia o conjunto de treino com *data augmentation* simples:

- `RandomHorizontalFlip`
- `RandomRotation`
- `ColorJitter`

### `benchmark.py`
É o entrypoint do benchmark principal e delega a execução para `cv_framework/benchmarking.py`. Ele:

- lê os dados de `mri_split_70_20_10/`;
- aplica `SquarePad`, `Resize(224, 224)` e normalização ImageNet;
- treina vários modelos `timm` com *early stopping* baseado em `F1-macro` de validação;
- salva os melhores pesos em `results/<modelo>/best.pth`;
- gera gráficos de aprendizado, F1 por classe, curvas ROC e matrizes de confusão;
- consolida um ranking em `plots/benchmark_results_test_train.csv`.

### `config.py`
Arquivo único para manter os hiperparâmetros e caminhos do projeto. Entre os itens centralizados estão:

- `DEFAULT_SEED`
- `DEFAULT_BATCH_SIZE`
- `DEFAULT_EPOCHS`
- `DEFAULT_LR`
- `DEFAULT_EARLY_STOPPING_PATIENCE`
- `DEFAULT_EARLY_STOPPING_MIN_DELTA`
- `BENCHMARK_MODELS`
- `ROBUSTNESS_MODELS`
- diretórios de saída como `BENCHMARK_PLOT_DIR`, `TRAINING_PLOT_DIR` e `ROBUSTNESS_PLOT_DIR`

### `cv_framework/benchmarking.py`
Contém a implementação modular do benchmark principal:

- `BenchmarkContext` para agrupar parâmetros do experimento;
- `BenchmarkRunner` para carregar dados, treinar, avaliar e gerar relatórios;
- `run_benchmark()` como atalho simples para execução.

### `evaluate_models.py`
Reaproveita os melhores pesos salvos em `results/` para medir robustez no conjunto de teste sob três tipos de perturbação:

- ruído gaussiano
- desfoque (`blur`)
- redução de contraste

Cada perturbação é avaliada em três níveis: leve, moderado e extremo.

### `train_model.py`, `test_model.py` e `visualize_img_atack.py`
Esses scripts também passaram a consumir os caminhos e hiperparâmetros de `config.py`, reduzindo duplicação e facilitando ajustes globais.

### Scripts auxiliares

- `eda.py`: exploração/análise dos dados.
- `visualize_img_atack.py`: visualização de ataques/perturbações.
- `test_vram.py`: utilitário para checagem de memória/VRAM.

---

## Principais resultados do benchmark

Os resultados já consolidados em `plots/benchmark_results_test_train.csv` mostram desempenho muito alto no conjunto de teste para vários modelos. O ranking abaixo resume os destaques mais relevantes:

| Modelo | Test F1-Macro | Test AUC-Macro | Val F1-Macro | Parâmetros |
|---|---:|---:|---:|---:|
| `convnext_tiny` | **0.995764** | 0.999919 | 0.994923 | 27.82 M |
| `swin_s3_tiny_224` | 0.995529 | **0.999966** | 0.993485 | 27.56 M |
| `mobilevit_s` | 0.994523 | 0.999931 | 0.992702 | 4.94 M |
| `tiny_vit_11m_224` | 0.994089 | 0.999847 | 0.991829 | 10.55 M |
| `fastvit_t8` | 0.992730 | 0.999883 | 0.992609 | 3.26 M |

### Interpretação rápida

- **Melhor F1 no teste:** `convnext_tiny`
- **Melhor AUC no teste:** `swin_s3_tiny_224`
- **Melhor equilíbrio entre desempenho e eficiência:** `mobilevit_s`
- **Modelo mais leve entre os destaques:** `fastvit_t8`

Em geral, os modelos apresentaram **AUC muito alta em teste** e diferença pequena entre eles na base limpa, o que indica que o problema principal não é separar classes no cenário ideal, e sim medir o quanto cada rede resiste a mudanças de qualidade na imagem.

### Artefatos gerados pelo benchmark

Cada modelo treinado recebe uma pasta em `results/<modelo>/` e `plots/<modelo>/` com, entre outros:

- `best.pth`
- `learning_curves.png`
- `f1_per_class.png`
- `roc_auc_curve_val.png`
- `confusion_matrix_val.png`
- `confusion_matrix_test.png`

Também são gerados os resumos globais:

- `plots/ranking_f1_test.png`
- `plots/eficiencia_test.png`
- `plots/benchmark_results_test_train.csv`

---

## Análise de robustez

A pasta `robustness_analysis/` mostra como os modelos se comportam quando o conjunto de teste sofre perturbações visuais.

O arquivo `robustness_metrics_levels.csv` resume as métricas de cada modelo em:

- cenário limpo (`Clean`)
- ruído leve, moderado e extremo
- blur leve, moderado e extremo
- contraste leve, moderado e extremo

### Destaques da robustez

- `convnext_tiny` foi o melhor no cenário limpo e também manteve a liderança em várias perturbações, especialmente em **ruído leve** e **blur**.
- `tiny_vit_11m_224` apresentou o melhor comportamento sob **redução de contraste**, inclusive nos níveis moderado e extremo.
- `vit_base_patch32_224` foi o melhor entre os modelos avaliados sob **ruído moderado** e **ruído extremo**.

Isso reforça uma conclusão importante do projeto: **o desempenho na base limpa não é suficiente para escolher o melhor modelo**. A robustez muda bastante conforme o tipo de degradação.

### Gráficos de robustez

- `robustness_analysis/robustness_noise.png`
- `robustness_analysis/robustness_blur.png`
- `robustness_analysis/robustness_contrast.png`
- `robustness_analysis/perturbation_samples.png`

---

## Requisitos

O projeto foi configurado para:

- **Python 3.13+**
- `torch`
- `torchvision`
- `timm`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tqdm`

As dependências estão declaradas em `pyproject.toml`.

---

## Como reproduzir

### 1. Preparar o ambiente

```bash
uv sync
```

Ou, se preferir `pip`:

```bash
pip install -e .
```

### 2. Preparar o split do dataset MRI

```bash
python split_data.py
```

### 3. Opcional: balancear o treino com augmentation

```bash
python augment_data.py
```

### 4. Treinar e comparar modelos

```bash
python benchmark.py
```

### 5. Avaliar robustez dos pesos salvos

```bash
python evaluate_models.py
```

---

## Leitura dos resultados

Se você quiser começar pelos arquivos mais importantes, siga esta ordem:

1. `plots/benchmark_results_test_train.csv`
2. `plots/ranking_f1_test.png`
3. `robustness_analysis/robustness_metrics_levels.csv`
4. `robustness_analysis/robustness_noise.png`
5. `robustness_analysis/robustness_blur.png`
6. `robustness_analysis/robustness_contrast.png`

---

## Conclusão

Este repositório implementa um pipeline sólido de classificação de imagens médicas, com comparação de arquiteturas modernas e análise de robustez em condições adversas. Os resultados já salvos mostram desempenho excelente no conjunto de teste limpo, com destaque para `convnext_tiny`, `swin_s3_tiny_224` e `mobilevit_s`.

O diferencial do projeto está em não parar na acurácia/F1 da base limpa: a análise de robustez revela que o comportamento dos modelos muda bastante sob ruído, blur e contraste, o que é essencial em aplicações médicas reais.
