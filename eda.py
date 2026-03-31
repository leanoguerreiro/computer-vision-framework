import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np


def realizar_eda(dataset_path):
    data = []
    # Novas classes baseadas na estrutura da imagem
    classes = ['glioma', 'healthy', 'meningioma', 'pituitary']
    # Os splits de divisão de dados
    splits = ['train', 'val', 'test']

    print("Coletando metadados das imagens...")

    for split in splits:
        for cls in classes:
            path = os.path.join(dataset_path, split, cls)

            # Verifica se o caminho existe para evitar erros
            if not os.path.exists(path):
                continue

            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
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
                    # Ignora arquivos que não são imagens válidas
                    continue

    df = pd.DataFrame(data)

    if df.empty:
        print("Nenhuma imagem foi encontrada. Verifique o caminho do dataset.")
        return df

    # --- VISUALIZAÇÃO ---
    plt.figure(figsize=(18, 12))

    # 1. Distribuição de Classes (agrupado por split)
    plt.subplot(2, 2, 1)
    # Mostra a distribuição de classes considerando train/val/test
    sns.countplot(data=df, x='class', hue='split', palette='viridis')
    plt.title('Distribuição de Classes por Split (Train/Val/Test)')
    plt.xticks(rotation=45)

    # 2. Distribuição de Brilho
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='brightness', hue='class', kde=True, element="step", palette='tab10')
    plt.title('Distribuição de Brilho Médio')

    # 3. Dispersão de Resolução (Width vs Height)
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='width', y='height', hue='class', alpha=0.5, palette='tab10')
    plt.title('Dimensões das Imagens (Resolução)')

    # 4. Aspect Ratio
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='class', y='aspect_ratio', palette='tab10')
    plt.title('Proporção da Imagem (Aspect Ratio)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Exibir amostras das imagens (usando a pasta de treino como referência)
    exibir_amostras(dataset_path, target_split='train')

    return df


def exibir_amostras(dataset_path, target_split='train'):
    classes = ['glioma', 'healthy', 'meningioma', 'pituitary']
    # Atualizado para 4 linhas (uma por classe) e 5 colunas
    fig, axes = plt.subplots(len(classes), 5, figsize=(20, 15))

    for i, cls in enumerate(classes):
        path = os.path.join(dataset_path, target_split, cls)

        if not os.path.exists(path):
            continue

        # Pega apenas arquivos reais para não bugar com pastas escondidas
        imgs = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))][:5]

        for j, img_name in enumerate(imgs):
            img = Image.open(os.path.join(path, img_name))
            axes[i, j].imshow(img, cmap='gray')  # Adicionado cmap='gray' caso as MRI já sejam cinzas
            axes[i, j].set_title(f"{cls}\n{img.size}")
            axes[i, j].axis('off')

        # Limpa os eixos vazios caso a pasta tenha menos de 5 imagens
        for j in range(len(imgs), 5):
            axes[i, j].axis('off')

    plt.suptitle(f"Amostras do Dataset (Split: {target_split})", fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


if __name__ == "__main__":
    # Substitua pelo caminho correto onde a pasta 'mri_split_70_20_10' está localizada
    caminho_do_dataset = "mri_split_70_20_10"

    df_meta = realizar_eda(caminho_do_dataset)

    if not df_meta.empty:
        print("\nResumo Estatístico do Brilho (por classe):")
        print(df_meta.groupby('class')['brightness'].describe())

        print("\nContagem de Imagens por Split e Classe:")
        print(df_meta.groupby(['split', 'class']).size().unstack(fill_value=0))