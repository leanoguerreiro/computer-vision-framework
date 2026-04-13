import csv
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def gerar_metadata(diretorio_raiz, output_csv="dataset_metadata.csv", output_json="dataset_summary.json"):
    path_raiz = Path(diretorio_raiz)

    if not path_raiz.exists():
        print(f"Erro: O diretório {diretorio_raiz} não existe.")
        return

    dados_imagens = []
    resumo = {}

    # Cabeçalho do CSV
    cabecalho = ["caminho_relativo", "split", "classe", "largura", "altura", "formato"]

    # Identifica as pastas de split (ex: train, val, test)
    splits = [d for d in path_raiz.iterdir() if d.is_dir()]

    print(f"🔍 Analisando o dataset em: {path_raiz.absolute()}")

    for pasta_split in splits:
        nome_split = pasta_split.name
        resumo[nome_split] = {"total_imagens": 0, "classes": {}}

        # Identifica as pastas de classes (0, 1, 2, 3...)
        classes = [d for d in pasta_split.iterdir() if d.is_dir()]

        for pasta_classe in classes:
            nome_classe = pasta_classe.name
            imagens = [f for f in pasta_classe.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

            resumo[nome_split]["classes"][nome_classe] = len(imagens)
            resumo[nome_split]["total_imagens"] += len(imagens)

            # Extrai info de cada imagem
            for img_path in tqdm(imagens, desc=f"Lendo {nome_split}/Classe {nome_classe}"):
                try:
                    # O PIL abre apenas o cabeçalho do arquivo aqui (super rápido)
                    with Image.open(img_path) as img:
                        largura, altura = img.size
                        formato = img.format

                    # Salva o caminho relativo (ex: train/9/aug_0_img.jpg) para manter portabilidade
                    caminho_relativo = img_path.relative_to(path_raiz)

                    dados_imagens.append({
                        "caminho_relativo": str(caminho_relativo),
                        "split": nome_split,
                        "classe": nome_classe,
                        "largura": largura,
                        "altura": altura,
                        "formato": formato
                    })
                except Exception as e:
                    print(f"⚠️ Erro ao ler {img_path.name}: {e}")

    # 1. Salvar o CSV com todos os detalhes
    print("\n💾 Salvando metadados no CSV...")
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cabecalho)
        writer.writeheader()
        writer.writerows(dados_imagens)

    # 2. Salvar o JSON com o resumo do dataset
    print("💾 Salvando resumo no JSON...")
    with open(output_json, mode='w', encoding='utf-8') as f:
        json.dump(resumo, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Concluído! ")
    print(f" -> Detalhes de {len(dados_imagens)} imagens salvos em: {output_csv}")
    print(f" -> Resumo salvo em: {output_json}")


if __name__ == "__main__":
    # Aponte para a raiz do seu dataset (a pasta que contém train, val, test)
    RAIZ_DATASET = "datasets/mri_split_70_20_10"
    OUTPUT_CSV = f"{RAIZ_DATASET}/dataset_metadata.csv"
    OUTPUT_JSON = f"{RAIZ_DATASET}/dataset_summary.json"

    gerar_metadata(RAIZ_DATASET, output_csv=OUTPUT_CSV, output_json=OUTPUT_JSON)