import glob
import polars as pl

# 1. Mapear e carregar todos os arquivos CSV do diretório
# Se preferir, você pode passar a lista explícita com os nomes dos seus arquivos
arquivos_csv = glob.glob("*_granular_folds.csv")

# Ler todos os CSVs e concatenar em um único DataFrame do Polars
dfs = [pl.read_csv(arquivo) for arquivo in arquivos_csv]
df = pl.concat(dfs)

# 2. Definir as métricas numéricas que serão calculadas
colunas_metricas = [
    "Test_Accuracy",
    "Test_Precision",
    "Test_Recall",
    "Test_F1-Macro",
    "Test_Specificity",
    "Test_MCC",
    "Test_AUC-Macro",
    "Test_MSE",
    "Train_Time_s",
    "Inference_ms_per_img",
]

# ==============================================================================
# OPÇÃO 1: Colunas Separadas para Média e Desvio Padrão
# ==============================================================================
expr_separadas = []
for col in colunas_metricas:
    expr_separadas.append(pl.col(col).mean().alias(f"{col}_mean"))
    expr_separadas.append(pl.col(col).std().alias(f"{col}_std"))

resumo_analitico = (
    df.group_by("Modelo").agg(expr_separadas).sort("Modelo")
)

print("--- Resumo Analítico (Colunas Separadas) ---")
print(resumo_analitico)
resumo_analitico.write_csv("resumo_analitico_polars.csv")

# ==============================================================================
# OPÇÃO 2: Formato Científico ("Média ± Desvio Padrão")
# ==============================================================================
expr_formatadas = []
for col in colunas_metricas:
    # Arredondamos para 4 casas decimais e combinamos texto
    expr_formatadas.append(
        pl.concat_str(
            [
                pl.col(col).mean().round(4).cast(pl.String),
                pl.lit(" ± "),
                pl.col(col).std().round(4).cast(pl.String),
            ]
        ).alias(col)
    )

resumo_formatado = (
    df.group_by("Modelo").agg(expr_formatadas).sort("Modelo")
)

print("\n--- Resumo Formatado (Média ± Desvio) ---")
print(resumo_formatado)
resumo_formatado.write_csv("resumo_formatado_polars.csv")

print("\nArquivos 'resumo_analitico_polars.csv' e 'resumo_formatado_polars.csv' salvos com sucesso!")