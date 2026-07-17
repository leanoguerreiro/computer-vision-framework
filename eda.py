import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import math
import random


def realizar_eda(dataset_path):
    data = []
    splits = ['train', 'val', 'test']

    print(f"Analisando o diretório: {dataset_path}")

    # 1. Descobrir as classes dinamicamente a partir dos subdiretórios
    classes_set = set()
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            # Adiciona apenas se for um diretório (ignora arquivos soltos como .DS_Store)
            for item in os.listdir(split_path):
                if os.path.isdir(os.path.join(split_path, item)):
                    classes_set.add(item)

    classes = sorted(list(classes_set))

    if not classes:
        print("Nenhuma classe encontrada. Verifique se o caminho tem subdiretórios válidos.")
        return pd.DataFrame()

    print(f"Classes detectadas: {classes}")
    print("Coletando metadados das imagens...")

    # Passar as classes dinâmicas para a função de amostra
    exibir_amostras(dataset_path, classes, target_split='train')

    # 2. Coletar os dados
    for split in splits:
        for cls in classes:
            path = os.path.join(dataset_path, split, cls)

            if not os.path.exists(path):
                continue

            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                # Ignorar diretórios internos ou arquivos ocultos
                if not os.path.isfile(img_path) or img_name.startswith('.'):
                    continue

                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        # Estatísticas básicas de cor (brilho médio)
                        img_array = np.array(img.convert('L'))  # Converte para cinza
                        mean_brightness = img_array.mean()

                        data.append({
                            'filename': img_name,
                            'split': split,
                            'class': cls,
                            'width': width,
                            'height': height,
                            'brightness': mean_brightness,
                            'aspect_ratio': width / height
                        })
                except Exception as e:
                    pass  # Ignora arquivos que não são imagens válidas

    df = pd.DataFrame(data)

    if df.empty:
        print("Nenhuma imagem válida foi processada.")
        return df

    # --- VISUALIZAÇÃO ---
    plt.figure(figsize=(18, 12))

    # 1. Distribuição de Classes (agrupado por split)
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='class', hue='split', palette='viridis', order=classes)
    plt.title('Distribuição de Classes por Split (Train/Val/Test)')
    plt.xticks(rotation=45)

    # 2. Distribuição de Brilho
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='brightness', hue='class', kde=True, element="step", palette='tab10', hue_order=classes)
    plt.title('Distribuição de Brilho Médio')

    # 3. Dispersão de Resolução (Width vs Height)
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='width', y='height', hue='class', alpha=0.5, palette='tab10', hue_order=classes)
    plt.title('Dimensões das Imagens (Resolução)')

    # 4. Aspect Ratio
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='class', y='aspect_ratio', hue='class', palette='tab10', order=classes, legend=False)
    plt.title('Proporção da Imagem (Aspect Ratio)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()



    return df


def exibir_amostras(dataset_path, classes, target_split='train'):
    num_classes = len(classes)

    if num_classes == 0:
        print("Nenhuma classe encontrada para exibir amostras.")
        return

    # Define o número de colunas de forma dinâmica
    MAX_COLS = 5
    if num_classes <= MAX_COLS:
        cols = num_classes
        rows = 1
    else:
        # Se houver mais classes, use o número máximo de colunas e calcule as linhas
        cols = MAX_COLS
        rows = math.ceil(num_classes / cols)

    # Cria a figura com um tamanho proporcional
    # Aumentamos um pouco o multiplicador de largura para as imagens ficarem maiores e o layout mais apertado
    width_per_col = 4.0
    height_per_row = 3.5
    fig, axes = plt.subplots(rows, cols, figsize=(width_per_col * cols, height_per_row * rows))

    # Converte 'axes' para um array unidimensional para facilitar o loop
    if num_classes == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        # Se for uma única linha ou coluna, não precisa de flatten (depende da versão do matplotlib, mas essa é uma boa prática)
        axes = axes
    else:
        axes = axes.flatten()

    for i, cls in enumerate(classes):
        ax = axes[i]
        path = os.path.join(dataset_path, target_split, cls)

        if not os.path.exists(path):
            ax.set_title(f"Classe: {cls}\n(Pasta ausente)", fontsize=10)
            ax.axis('off')
            continue

        imgs = [
            f
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and not f.startswith('.')
        ]

        if not imgs:
            ax.set_title(
                f"Classe: {cls}\n(Sem imagens em '{target_split}')", fontsize=10
            )
            ax.axis('off')
            continue

        img_name = random.choice(imgs)
        img_path = os.path.join(path, img_name)

        try:
            img = Image.open(img_path)
            # Removemos o cmap='gray' para mostrar as cores originais da imagem
            # Adicionamos interpolation='none' para evitar suavização artificial na exibição
            ax.imshow(img, aspect='equal', interpolation='none')
            ax.set_title(
                f"Classe: {cls}\nTam: {img.size[0]}x{img.size[1]} px",
                fontsize=11,
                fontweight='bold',
            )
        except Exception as e:
            ax.set_title(f"Classe: {cls}\n(Erro de leitura)", fontsize=10)

        ax.axis('off')  # Esconde os eixos X e Y

    # Oculta os subplots que sobrarem vazios no final da grade
    for j in range(num_classes, len(axes)):
        axes[j].axis('off')

    plt.suptitle(
        f"Amostra Aleatória por Classe",
        fontsize=16,
        fontweight='bold',
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    # Em alguns casos, pode ser necessário forçar um layout mais apertado após o preenchimento
    # plt.tight_layout(pad=0) # Tente isso se ainda houver problemas
    plt.show()


if __name__ == "__main__":
    # Substitua pelo caminho do seu dataset atual
    caminho_do_dataset = "datasets/MC_ALL_split_70_20_10"  # Pode ser o
    # mri_split_70_20_10 também!

    df_meta = realizar_eda(caminho_do_dataset)

    if not df_meta.empty:
        print("\nResumo Estatístico do Brilho (por classe):")
        print(df_meta.groupby('class')['brightness'].describe())

        print("\nContagem de Imagens por Split e Classe:")
        print(df_meta.groupby(['split', 'class']).size().unstack(fill_value=0))