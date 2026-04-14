import torch
import torch.nn as nn
import timm
import gc


def testar_limites_vram(lista_modelos, batch_size=32, image_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != 'cuda':
        print("⚠️ CUDA não detectado. O teste de VRAM precisa de uma GPU da Nvidia.")
        return

    print(f"{'=' * 50}")
    print(f" INICIANDO TESTE DE ESTRESSE NA VRAM")
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" Lote (Batch Size): {batch_size} imagens")
    print(f"{'=' * 50}\n")

    resultados = []

    for nome_modelo in lista_modelos:
        print(f"Testando: {nome_modelo:<30} ...", end=" ")

        try:
            # 1. Cria o modelo (pretrained=False para não perder tempo baixando pesos)
            model = timm.create_model(nome_modelo, pretrained=False, num_classes=2).to(device)
            model.train()  # Coloca em modo de treino (gasta mais memória)

            # 2. Cria imagens e rótulos falsos (Ruído aleatório)
            dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
            dummy_target = torch.randint(0, 2, (batch_size,), device=device)

            criterion = nn.CrossEntropyLoss()

            # 3. Forward Pass (Previsão)
            outputs = model(dummy_input)
            loss = criterion(outputs, dummy_target)

            # 4. Backward Pass (Cálculo de Gradientes - Onde a VRAM geralmente estoura)
            loss.backward()

            print("✅ APROVADO")
            resultados.append((nome_modelo, "Passou"))

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("❌ REPROVADO (CUDA Out of Memory)")
                resultados.append((nome_modelo, "Faltou VRAM"))
            else:
                print(f"⚠️ ERRO: {str(e)[:50]}...")
                resultados.append((nome_modelo, "Erro Desconhecido"))

        finally:
            # 5. Limpeza Agressiva da GPU para o próximo modelo
            if 'model' in locals(): del model
            if 'dummy_input' in locals(): del dummy_input
            if 'dummy_target' in locals(): del dummy_target
            if 'outputs' in locals(): del outputs
            if 'loss' in locals(): del loss

            # Força o coletor de lixo do Python e esvazia o cache do PyTorch
            gc.collect()
            torch.cuda.empty_cache()

    # --- Resumo Final ---
    print("\n" + "=" * 50)
    print(" RESUMO DO TESTE DE HARDWARE")
    print("=" * 50)
    for nome, status in resultados:
        icone = "🟢" if status == "Passou" else "🔴"
        print(f"{icone} {nome:<30}: {status}")


if __name__ == "__main__":
    # A sua lista de arquiteturas
    modelos_candidatos = [
    # # ── originais ──────────────────────────────────────────────────────────────
    # "mobilenetv3_large_100",
    # "efficientnet_b0",
    # "resnet18",
    # "resnet50",
    # "efficientnet_b3",
    # "convnext_small",
    # "mobilevit_s",
    # "fastvit_t8",
    # "tiny_vit_11m_224",
    # "vit_small_patch16_224",
    # "swin_tiny_patch4_window7_224",
    # "vit_base_patch16_224",
    # # ── leves ──────────────────────────────────────────────────────────────────
    # "efficientnet_b1",
    # "efficientnet_b2",
    # "mobilenetv3_small_100",
    # "ghostnet_100",
    # # ── médios ─────────────────────────────────────────────────────────────────
    # "resnet34",
    # "densenet121",
    # "convnext_tiny",
    # "swin_s3_tiny_224",
    # "xcit_small_12_p16_224",
    # "vit_base_patch32_224",
    # # ── médico: densenet169 com pesos ImageNet ─────────────────────────────────
    # "densenet169",
    # "convnext_base",
    # "efficientnetv2_m"
    "mobilenetv3_large_100",
    "mobilevit_s",
    "efficientnet_b3",
    "resnet50",
    "swin_base_patch4_window7_224",
    "convformer_s18",
    "convnext_base",
    "vit_base_patch16_224",
    "maxvit_tiny_tf_224"
]

    # Você pode alterar o batch_size aqui para testar o limite da RTX 3060
    testar_limites_vram(modelos_candidatos, batch_size=32)