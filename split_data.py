import shutil
import random
from pathlib import Path


def dividir_treino_val_teste(origem, destino, p_treino=0.7, p_val=0.2, p_teste=0.1, seed=42):
    """
    Divide o dataset em Treino (70%), Validação (20%) e Teste (10%).
    """
    # Garantir que a soma seja 1.0 (100%)
    assert round(p_treino + p_val + p_teste, 1) == 1.0, "As proporções devem somar 1.0"

    random.seed(seed)
    path_origem = Path(origem)
    path_destino = Path(destino)

    # 1. Identificar as classes automaticamente (glioma, healthy, etc.)
    classes = [f.name for f in path_origem.iterdir() if f.is_dir()]

    # 2. Criar estrutura de pastas: train, val, test
    splits = ['train', 'val', 'test']
    for split in splits:
        for cls in classes:
            (path_destino / split / cls).mkdir(parents=True, exist_ok=True)

    print(f"Dividindo dataset: {p_treino * 100}% / {p_val * 100}% / {p_teste * 100}%")

    # 3. Processar cada classe
    for cls in classes:
        pasta_classe = path_origem / cls
        imagens = [f for f in pasta_classe.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        random.shuffle(imagens)

        # Calcular pontos de corte
        total = len(imagens)
        fim_treino = int(total * p_treino)
        fim_val = int(total * (p_treino + p_val))

        # Divisão das listas
        treino_imgs = imagens[:fim_treino]
        val_imgs = imagens[fim_treino:fim_val]
        teste_imgs = imagens[fim_val:]

        print(f" -> Classe '{cls}': {len(treino_imgs)} Train | {len(val_imgs)} Val | {len(teste_imgs)} Test")

        # 4. Função auxiliar para copiar
        def copiar_arquivos(lista, pasta_split):
            for img in lista:
                shutil.copy2(img, path_destino / pasta_split / cls / img.name)

        copiar_arquivos(treino_imgs, 'train')
        copiar_arquivos(val_imgs, 'val')
        copiar_arquivos(teste_imgs, 'test')

    print(f"\n✅ Concluído! Dataset organizado em: {destino}")


if __name__ == "__main__":
    # Ajuste para o caminho da sua pasta MRI
    PASTA_ORIGEM = "./data_raw/Multi Cancer/Multi Cancer/ALL"
    PASTA_DESTINO = "datasets/MC_ALL_70_20_10"

    dividir_treino_val_teste(PASTA_ORIGEM, PASTA_DESTINO)