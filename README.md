# cv_framework

Framework modular e funcional para benchmarking de modelos de visão computacional em classificação de imagens médicas (leucemia — dataset multi-classe `all_benign`, `all_early`, `all_pre`, `all_pro`), com suporte nativo a:

- Treinamento single-split (train/val/test) **e** validação cruzada estratificada (K-Fold);
- Comparação sistemática entre múltiplas arquiteturas (CNNs, Transformers e híbridos DINOv2/v3);
- Interpretabilidade (Grad-CAM, atenção de Transformers, projeção UMAP do espaço latente);
- Auditoria de robustez sob perturbações (ruído, blur, contraste);
- Estatística inter-modelo (matriz de concordância, teste de McNemar);
- Relatórios e gráficos comparativos automatizados.

O núcleo (`cv_framework/`) é **agnóstico de projeto**: não conhece caminhos de dataset nem hiperparâmetros fixos. Toda configuração é injetada via `BenchmarkContext` ou `config.py`, que vivem na raiz do projeto que consome o framework.

---

## Índice

- [Arquitetura geral](#arquitetura-geral)
- [Instalação](#instalação)
- [Estrutura de dataset esperada](#estrutura-de-dataset-esperada)
- [Início rápido](#início-rápido)
- [Módulos do framework](#módulos-do-framework)
- [Pipelines de entrada (scripts da raiz)](#pipelines-de-entrada-scripts-da-raiz)
- [Modelos suportados](#modelos-suportados)
- [Saídas geradas](#saídas-geradas)
- [Estendendo o framework](#estendendo-o-framework)

---

## Arquitetura geral

```
projeto/
├── config.py                      # Configuração central (caminhos, hiperparâmetros, lista de modelos)
├── benchmark_kfold.py              # Entry point: benchmark comparativo via K-Fold
├── evaluate_robustness_kfold.py    # Entry point: auditoria de robustez com ensembles K-Fold
├── datasets/
│   └── MC_ALL_split_70_20_10/
│       ├── train/<classe>/*.png
│       ├── val/<classe>/*.png
│       └── test/<classe>/*.png
└── cv_framework/                   # Biblioteca reutilizável (paradigma funcional)
    ├── __init__.py
    ├── context.py
    ├── reproducibility.py
    ├── data.py
    ├── kfold.py
    ├── transforms.py
    ├── models.py
    ├── training.py
    ├── metrics.py
    ├── explainability.py
    ├── plots.py
    ├── inter_model.py
    └── io.py
```

O design segue um estilo predominantemente **funcional**: a maioria das funções recebe dados/estado explicitamente e retorna novos valores (ex.: `step_early_stopping` retorna um novo `EarlyStoppingState` em vez de mutar o antigo), em vez de depender de classes com estado interno oculto.

Fluxo conceitual:

```
config.py ──► BenchmarkContext ──► data.py / kfold.py ──► training.py ──► metrics.py
                                                              │
                                                              ▼
                                          explainability.py + plots.py + io.py + inter_model.py
```

---

## Instalação

Dependências principais (não há `requirements.txt` incluído — instale conforme necessário):

```bash
pip install torch torchvision timm transformers
pip install scikit-learn scikit-image statsmodels
pip install polars pandas matplotlib seaborn
pip install opencv-python pytorch-grad-cam umap-learn tqdm
```

> `models.py` usa `transformers.AutoModel` para carregar DINOv2/DINOv3 (`facebook/dinov2-base`, `facebook/dinov3-vit-base`) e `timm` para as demais arquiteturas.

---

## Estrutura de dataset esperada

O framework assume um `ImageFolder` do torchvision já dividido em três pastas (`train`, `val`, `test`), cada uma contendo uma subpasta por classe:

```
MC_ALL_split_70_20_10/
├── train/{all_benign,all_early,all_pre,all_pro}/*.png
├── val/{all_benign,all_early,all_pre,all_pro}/*.png
└── test/{all_benign,all_early,all_pre,all_pro}/*.png
```

O `dataset_summary.json` de referência descreve a distribuição atual: 14.000 imagens de treino, 4.000 de validação e 2.000 de teste, balanceadas entre 4 classes (3.500 / 1.000 / 500 por classe respectivamente).

No modo **K-Fold**, `train` e `val` são mesclados em memória (`ConcatDataset`) para formar o pool estratificado; `test` permanece intocado e é usado separadamente no script de robustez.

---

## Início rápido

### 1. Treinamento simples (single-split)

```python
from pathlib import Path
from cv_framework import (
    BenchmarkContext, set_seed, setup_data_loaders, train_model_pipeline,
)

ctx = BenchmarkContext(
    root_dir=Path("datasets/MC_ALL_split_70_20_10"),
    plot_dir=Path("plots"),
    results_dir=Path("results"),
    num_epochs=25,
    batch_size=32,
)

set_seed(ctx.seed)
(train_loader, val_loader, test_loader), class_names = setup_data_loaders(ctx, ctx.batch_size)

resultado = train_model_pipeline(
    model_name="resnet50",
    context=ctx,
    loaders=(train_loader, val_loader, test_loader),
    class_names=class_names,
)
```

### 2. Benchmark comparativo com K-Fold

```bash
python benchmark_kfold.py
```

Isso executa `run_kfold_benchmark(k=5)`, que treina cada modelo em `config.BENCHMARK_MODELS` em todos os folds, agrega métricas Out-Of-Fold (OOF) e gera relatórios/gráficos comparativos.

### 3. Auditoria de robustez dos ensembles treinados

```bash
python evaluate_robustness_kfold.py
```

Carrega os K modelos salvos por `benchmark_kfold.py`, monta um ensemble por arquitetura (média das probabilidades softmax) e avalia sob 10 cenários de perturbação (ruído, blur e contraste, em 3 intensidades cada + `Clean`).

---

## Módulos do framework

### `context.py`
Define `BenchmarkContext`, um `dataclass(frozen=True)` que centraliza toda configuração injetável: diretórios, batch size, épocas, learning rate, paciência de early stopping, número de workers, seed e overrides de batch size por modelo. É o objeto que percorre praticamente todas as funções do framework.

### `reproducibility.py`
`set_seed(seed)` fixa seeds do Python (`random`), NumPy, PyTorch (CPU e CUDA) e força `cudnn.deterministic=True` / `cudnn.benchmark=False`.

### `data.py`
Camada de dados para o fluxo **single-split**:
- `build_imagefolder_datasets` — cria os três `ImageFolder` (train/val/test);
- `get_class_names` — lê nomes de classes de um split;
- `build_class_weights` — calcula pesos inversamente proporcionais à frequência de classe, para `CrossEntropyLoss` ponderada;
- `setup_data_loaders` — orquestrador que retorna os três `DataLoader` prontos + nomes das classes.

### `kfold.py`
Camada de dados para o fluxo **K-Fold**:
- `FoldDataset` — wrapper de `Dataset` que aplica transform dinamicamente sobre índices de um dataset base (`ImageFolder` ou `ConcatDataset`), preservando os targets corretos do fold;
- `get_stratified_kfold_datasets` — usa `StratifiedKFold` do scikit-learn para gerar `k` pares `(FoldDataset treino, FoldDataset validação)`. Por padrão (`merge_val=True`), mescla `train` + `val` em um único pool antes de particionar, deixando `test` de fora (reservado para avaliação de robustez).

### `transforms.py`
Pipelines de transformação com `torchvision.transforms.v2`, construídos de forma funcional (composição de listas de transforms):
- `build_train_transform` — pipeline de treino com augmentation pesado (rotação até 360°, crop aleatório, flips, blur, jitter de cor, grayscale, `RandomErasing`);
- `build_eval_transform` — pipeline limpo (apenas padding quadrado + resize + normalização) para validação/teste;
- `build_perturbation_transforms` — dicionário com 10 pipelines de estresse (`Clean`, `Noise_{Leve,Moderada,Extrema}`, `Blur_{...}`, `Contrast_{...}`) usados na auditoria de robustez;
- `build_visualization_perturbation_transforms` — variante sem normalização matemática, para gerar imagens visualmente interpretáveis;
- Funções puras auxiliares: `square_pad`, `apply_contrast`, `add_gaussian_noise`.

### `models.py`
Fábrica central de arquiteturas via `build_model(model_name, num_classes, ...)`. Suporta:
- Qualquer arquitetura disponível no `timm` (passagem direta);
- `resnet50_radimagenet` — baixa e carrega pesos pré-treinados do RadImageNet (via `urllib`) e substitui a camada final;
- `dinov2` / `dinov3` — `DinoVisionTransformer`, wrapper sobre `transformers.AutoModel` que expõe `forward_features` (embedding do token `[CLS]`) e `get_last_self_attention` (mapa de atenção da última camada), compatíveis com o pipeline de explicabilidade;
- `dino_hybrid` — `DinoSpatialHybrid`, arquitetura híbrida que projeta os tokens espaciais do DINOv2 para os canais de um `ConvNeXt` pré-treinado (via `timm`) e refina com os últimos blocos convolucionais antes do pooling global;
- `multicancernet_attention` — CNN customizada com blocos `CBAM` (atenção de canal + espacial) em 4 estágios;
- `multicancernet_attention_hybrid` — variante híbrida CNN + `TransformerEncoder` sobre um grid espacial 14×14 tokenizado.

### `training.py`
Motores de treinamento e validação (estilo funcional com estado explícito):
- `EarlyStoppingState` (`TypedDict`) + `init_early_stopping` / `step_early_stopping` — early stopping baseado em F1-Macro de validação, com efeito colateral de salvar o melhor `state_dict` em disco;
- `train_one_epoch` / `validate_one_epoch` — uma época de treino (com `OneCycleLR`) e uma época de validação, respectivamente;
- `generate_post_training_visualizations` — roteia automaticamente para Grad-CAM (arquiteturas CNN) ou mapas de atenção de Transformer (arquiteturas com atributo `transformer`);
- `train_model_pipeline` — orquestrador completo **single-split**: monta modelo, otimizador, scheduler, roda o loop de épocas com early stopping, avalia em teste e treino finais, gera todos os gráficos e retorna um dicionário rico de métricas prontas para `polars.DataFrame`;
- `train_kfold_fold` — orquestrador **enxuto** para um único fold do K-Fold: mesmo ciclo de treino/early stopping, mas sem geração de gráficos, sem UMAP/Grad-CAM e sem reavaliação do split de treino (foco em performance, já que roda `k × n_modelos` vezes). Retorna métricas Out-Of-Fold (`oof_preds`, `oof_labels` incluídos para análise inter-modelo posterior).

### `metrics.py`
Hub funcional de métricas de classificação:
- `run_inference` — roda um `DataLoader` completo no modelo e retorna predições, labels, matriz de probabilidades e tempo total;
- `calculate_auc` — trata automaticamente o caso binário (`roc_auc_score` na classe positiva) vs. multi-classe (`ovr`, média macro);
- `calculate_specificity` — especificidade macro (TNR) calculada a partir da matriz de confusão;
- `calculate_all_metrics` — agrega acurácia, precisão, recall, F1-macro, MCC, AUC-macro, especificidade e MSE (contra one-hot) em um único dicionário;
- `evaluate_split` — combina `run_inference` + `calculate_all_metrics` + tempo de inferência por imagem, retornando um dicionário completo pronto para uso (gráficos, CSV, comparação inter-modelo).

### `explainability.py`
Módulo de interpretabilidade (XAI):
- `extract_attention_map` — extrai atenção da última camada de um `TransformerEncoder` genérico via forward pre-hook (usado como fallback para modelos sem atenção nativa exposta);
- `create_heatmap_overlay` — desnormaliza a imagem (usando `IMAGENET_MEAN/STD`), redimensiona o mapa de atenção e monta o overlay heatmap (60%) + imagem original (40%);
- `generate_transformer_samples` — orquestrador que percorre um `DataLoader` de teste, extrai atenção (via `get_last_self_attention` quando disponível, ou hooks como fallback) e salva `N` amostras por classe;
- `get_target_layer_for_cam` — heurística que identifica automaticamente a camada convolucional alvo para Grad-CAM, com regras específicas por família de arquitetura (`resnet`, `densenet`, `efficientnet`, `mobilenetv3`, `dino_hybrid`, `multicancernet_attention`) e fallback genérico (última `Conv2d`);
- `generate_gradcam_samples` — orquestrador equivalente ao anterior, usando `pytorch_grad_cam.GradCAM`;
- `plot_latent_space` — projeta o espaço de características (via `forward_features`, com fallback para saída direta do modelo) em 2D com UMAP e plota colorido por classe real.

### `plots.py`
Módulo puramente de visualização (Matplotlib/Seaborn/Polars), sem lógica de negócio:
- Atômicas por modelo: `plot_learning_curves`, `plot_f1_per_class`, `plot_roc_auc_curve`, `plot_confusion_matrix`, `plot_confidence_distribution`;
- Orquestrador por modelo: `generate_model_plots` (curvas de aprendizado + F1 por classe + ROC + matriz de confusão);
- Orquestrador global (multi-modelo): `generate_global_reports` — ranking de F1 de teste, eficiência (parâmetros × F1), trade-off velocidade × performance, análise de overfitting (treino vs. teste) e custo de treinamento acumulado;
- `plot_inter_model_agreement` — heatmap da matriz de concordância entre modelos.

### `inter_model.py`
`calculate_inter_model_metrics(results)` — recebe uma lista de resultados por modelo e calcula:
1. Matriz de concordância (% de predições idênticas entre cada par de modelos);
2. Teste de McNemar (exato) do melhor modelo (maior F1-teste) contra todos os demais, sobre uma tabela de contingência de acertos/erros;
3. Índices das amostras em que **todos** os modelos erraram simultaneamente ("falhas genuínas" — casos provavelmente difíceis/ambíguos no dataset).

### `io.py`
`update_results_csv(df_new, csv_path)` — faz upsert funcional de um CSV de resultados usando Polars: remove do CSV antigo as linhas de modelos que estão sendo reescritos, concatena com os novos resultados, ordena por F1-teste decrescente e salva.

### `__init__.py`
Superfície pública do pacote — reexporta todas as funções/classes acima organizadas por domínio conceitual (contexto/reprodutibilidade, dados/K-Fold, modelos, treinamento, explicabilidade, gráficos, I/O), permitindo `from cv_framework import build_model, train_model_pipeline, ...`.

---

## Pipelines de entrada (scripts da raiz)

Estes scripts **não fazem parte do pacote `cv_framework`** — vivem na raiz do projeto e o consomem, injetando configuração de `config.py`.

### `config.py`
Fonte única de verdade do projeto: diretórios derivados (`plots/`, `results/`, `robustness_analysis/` e variantes `_standard`), hiperparâmetros padrão (seed, épocas, LR, batch size, early stopping, workers), a tupla `BENCHMARK_MODELS` com as 14 arquiteturas avaliadas, overrides de batch size por modelo (para arquiteturas mais pesadas em VRAM) e a URL dos pesos do RadImageNet.

### `benchmark_kfold.py`
`run_kfold_benchmark(models, k, context)`:
1. Monta o `BenchmarkContext` (ou usa um customizado);
2. Gera os `k` folds estratificados uma única vez (`get_stratified_kfold_datasets`);
3. Para cada arquitetura em `models`: treina os `k` folds via `train_kfold_fold`, isolando falhas por modelo (um modelo com erro não interrompe o benchmark inteiro);
4. Agrega métricas OOF por modelo (média/desvio padrão via Polars) e concatena as predições OOF de todos os folds para análise estatística;
5. Roda `calculate_inter_model_metrics` (concordância + McNemar) sobre as predições OOF agregadas;
6. Faz upsert no CSV de resultados (`update_results_csv`) e gera todos os relatórios globais (`generate_global_reports`, `plot_inter_model_agreement`).

### `evaluate_robustness_kfold.py`
`run_kfold_robustness_benchmark(models, k, context)`:
1. Para cada arquitetura, carrega o **ensemble dos k modelos** já treinados por `benchmark_kfold.py` (`load_kfold_ensemble`, buscando os pesos em `results_dir/model_name/fold_N/best_*.pth`);
2. Roda inferência em consenso (`run_ensemble_inference` — média das probabilidades softmax dos k modelos) sobre o split de `test`, sob os 10 cenários de `build_perturbation_transforms`;
3. Registra métricas completas por (modelo, perturbação) em um `polars.DataFrame`, exporta CSV e gera gráficos comparativos de degradação (`plot_comparative_degradation` — barras agrupadas + curvas de queda de performance).

---

## Modelos suportados

Definidos em `config.BENCHMARK_MODELS` e roteados por `models.build_model`:

| Categoria | Modelos |
|---|---|
| CNNs (timm) | `resnet50`, `resnet50_radimagenet`, `densenet201`, `mobilenetv3_large_100`, `efficientnet_b3`, `convnext_base` |
| Transformers (timm) | `vit_base_patch16_224`, `swin_tiny_patch4_window7_224`, `swin_base_patch4_window7_224`, `maxvit_tiny_tf_224`, `mobilevit_s` |
| Backbone genérico timm | `convformer_s18` (padrão em `config.DEFAULT_MODEL_NAME`) |
| DINOv2/v3 | `dinov2`, `dino_hybrid` (DINOv2 + ConvNeXt) |
| Customizados | `multicancernet_attention` (CNN + CBAM), `all_attention_hybrid` / `all_attention` (roteados como `multicancernet_attention_hybrid` / `multicancernet_attention`) |

Qualquer outro nome de modelo válido do `timm` funciona automaticamente sem alterações no framework.

---

## Saídas geradas

Por modelo (`plot_dir/model_name/`):
- `learning_curves.png`, `f1_per_class.png`, `roc_auc_curve_val.png`, `confusion_matrix_val.png`, `confidence_distribution.png`
- `umap_latent_space.png`
- `gradcam/` ou `transformer_attention/` com amostras por classe

Globais (`plot_dir/`):
- `ranking_f1_test.png`, `eficiencia_test.png`, `velocidade_vs_performance.png`, `overfitting_analysis.png`, `tempo_treinamento.png`, `inter_model_agreement.png`

Robustez (`plot_dir/robustness/`):
- `kfold_{k}f_comparative_robustness_f1.png`, `kfold_{k}f_degradation_curves.png`, `benchmark_kfold_{k}f_robustness_summary.csv`

CSVs de resultados: `benchmark_kfold_{k}f_results.csv` (ranking agregado) e `{model_name}_granular_folds.csv` (log por fold).

---

## Estendendo o framework

- **Nova arquitetura**: adicionar um `elif` em `models.build_model` (ou apenas usar um nome válido do `timm`, sem código extra) e incluir o nome em `config.BENCHMARK_MODELS`.
- **Nova perturbação de robustez**: adicionar uma entrada ao dicionário retornado por `build_perturbation_transforms` (e, se desejar visualização, também em `build_visualization_perturbation_transforms`).
- **Nova métrica**: adicionar o cálculo em `calculate_all_metrics` (`metrics.py`) — ela se propaga automaticamente para `evaluate_split`, `train_model_pipeline`, `train_kfold_fold` e os CSVs de resultado.
- **Novo gráfico global**: adicionar uma função atômica em `plots.py` e chamá-la em `generate_global_reports`.
