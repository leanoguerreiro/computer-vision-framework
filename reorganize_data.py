import os
import shutil
from pathlib import Path


def organizar_para_classificacao(origem, destino):
    path_origem = Path(origem)
    path_destino = Path(destino)

    # Criar pastas das classes no destino
    classes = ['Anomaly', 'Normal']
    for classe in classes:
        (path_destino / classe).mkdir(parents=True, exist_ok=True)

    print(f"Organizando para CLASSIFICAÇÃO em: {path_destino}")

    # Percorrer pastas de PCBs
    for pcb_folder in path_origem.iterdir():
        # Pular arquivos soltos ou pastas de sistema
        if not pcb_folder.is_dir() or pcb_folder.name.startswith('.'):
            continue

        # Ignorar pastas que não sejam de dados (como 'dataset_final' ou 'split_csv')
        if pcb_folder.name in ['dataset_final', 'split_csv', 'dataset_pytorch']:
            continue

        print(f"Processando {pcb_folder.name}...")
        img_base = pcb_folder / "Data" / "Images"

        for classe in classes:
            classe_path = img_base / classe

            if not classe_path.exists():
                continue

            for img_path in classe_path.glob("*"):
                # Aceita JPG, jpeg e PNG (case-insensitive)
                if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Nome único para evitar sobrescrever (ex: pcb4_042.JPG)
                    new_filename = f"{pcb_folder.name}_{img_path.name}"
                    dest_path = path_destino / classe / new_filename

                    shutil.copy2(img_path, dest_path)

    print("\nConcluído! Imagens organizadas por classe.")


if __name__ == "__main__":
    # Ajuste 'data' para a sua pasta raiz onde estão pcb1, pcb2...
    organizar_para_classificacao("./data", "dataset_classificacao")