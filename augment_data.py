import random
import multiprocessing
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# 1. Definir as transformações globalmente para que os processos "workers" tenham acesso sem recriar
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])


# 2. Função isolada que processará uma única imagem (necessário para o multiprocessamento)
def processar_uma_imagem(args):
    i, caminho_img, path_destino = args
    try:
        # Abre, transforma e salva
        img_original = Image.open(caminho_img).convert('RGB')
        img_aug = augmentations(img_original)
        nome_novo = f"aug_{i}_{caminho_img.name}"
        img_aug.save(path_destino / nome_novo)
        return None  # Retorna None se deu tudo certo
    except Exception as e:
        return f"Erro em {caminho_img.name}: {e}"


def balancear_dataset_treino(pasta_train, seed=42):
    random.seed(seed)
    path_train = Path(pasta_train)

    # Mapear as classes e contar quantas imagens cada uma tem
    stats = {}
    for classe_dir in path_train.iterdir():
        if classe_dir.is_dir():
            imagens = [f for f in classe_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            stats[classe_dir.name] = {
                'path': classe_dir,
                'imagens': imagens,
                'total': len(imagens)
            }

    if not stats:
        print("Nenhuma pasta de classe encontrada no diretório.")
        return

    # Descobrir qual o objetivo (o total da maior classe)
    max_imagens = max(info['total'] for info in stats.values())
    print(f"Alvo de balanceamento: {max_imagens} imagens por classe.\n")

    # Preparar a lista de tarefas para o multiprocessamento
    tarefas = []
    for nome_classe, info in stats.items():
        total_atual = info['total']
        faltam = max_imagens - total_atual

        if faltam <= 0:
            print(f" -> Classe '{nome_classe}' já está no máximo. Pulando...")
            continue

        print(f" -> Adicionando {faltam} tarefas para a classe '{nome_classe}'...")
        imagens_base = random.choices(info['imagens'], k=faltam)

        for i, caminho_img in enumerate(imagens_base):
            tarefas.append((i, caminho_img, info['path']))

    if not tarefas:
        print("\nDataset já está totalmente balanceado!")
        return

    # 3. Executar as tarefas em paralelo usando todos os núcleos da CPU
    nucleos_disponiveis = multiprocessing.cpu_count()
    print(f"\nIniciando processamento paralelo usando {nucleos_disponiveis} núcleos...")

    with ProcessPoolExecutor(max_workers=nucleos_disponiveis) as executor:
        # O map aplica a função processar_uma_imagem em todas as tarefas paralelamente
        # O list() em volta do tqdm garante que a barra de progresso atualize conforme as tarefas terminam
        resultados = list(tqdm(executor.map(processar_uma_imagem, tarefas), total=len(tarefas), desc="Gerando imagens"))

    # Checar e imprimir possíveis erros que ocorreram nos workers
    erros = [r for r in resultados if r is not None]
    if erros:
        print(f"\nOcorreram {len(erros)} erros durante o processamento:")
        for erro in erros[:10]:
            print(erro)

    print("\n✅ Balanceamento concluído! Todas as pastas de treino agora têm o mesmo tamanho.")


if __name__ == "__main__":
    PASTA_TREINO = "datasets/deepterrain_70_20_10/train"
    balancear_dataset_treino(PASTA_TREINO)