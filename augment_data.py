import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path


def balancear_dataset_treino(pasta_train, seed=42):
    random.seed(seed)
    path_train = Path(pasta_train)

    # 1. Definir as transformações (Data Augmentation)
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])

    # 2. Mapear as classes e contar quantas imagens cada uma tem
    stats = {}
    for classe_dir in path_train.iterdir():
        if classe_dir.is_dir():
            imagens = [f for f in classe_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            stats[classe_dir.name] = {
                'path': classe_dir,
                'imagens': imagens,
                'total': len(imagens)
            }

    # 3. Descobrir qual o objetivo (o total da maior classe)
    max_imagens = max(info['total'] for info in stats.values())
    print(f"Alvo de balanceamento: {max_imagens} imagens por classe.\n")

    # 4. Processar cada classe para atingir o alvo
    for nome_classe, info in stats.items():
        total_atual = info['total']
        faltam = max_imagens - total_atual

        if faltam <= 0:
            print(f" -> Classe '{nome_classe}' já está no máximo. Pulando...")
            continue

        print(f" -> Classe '{nome_classe}': Gerando {faltam} imagens sintéticas...")

        # Seleciona aleatoriamente imagens da própria classe para servirem de base
        imagens_base = random.choices(info['imagens'], k=faltam)

        for i, caminho_img in enumerate(tqdm(imagens_base, desc=f"Augmenting {nome_classe}")):
            try:
                img_original = Image.open(caminho_img).convert('RGB')

                # Aplica transformação
                img_aug = augmentations(img_original)

                # Salva com nome único
                nome_novo = f"aug_{i}_{caminho_img.name}"
                img_aug.save(info['path'] / nome_novo)
            except Exception as e:
                print(f"Erro em {caminho_img.name}: {e}")

    print("\n✅ Balanceamento concluído! Todas as pastas de treino agora têm o mesmo tamanho.")


if __name__ == "__main__":
    # Caminho para a pasta 'train' dentro do seu split de 70/20/10
    PASTA_TREINO = "mri_split_70_20_10/train"

    balancear_dataset_treino(PASTA_TREINO)