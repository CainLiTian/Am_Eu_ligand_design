import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger, Chem
from jtnn_vae import JTNNVAE
from vocab import Vocab
import os

RDLogger.DisableLog("rdApp.*")
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 配置参数
LATENT_DIM = 48
CJTVAE_CKPT = "/home/dc/data_new/23/best_cjtvae_cond.pth"
VOCAB_PATH = "/home/dc/data_new/new_vocab.txt"
N_LATENT_VECTORS = 1000  # 隐向量数量
DECODES_PER_VECTOR = 10  # 每个向量解码次数
BATCH_SIZE = 5
SAVE_DIR = "/home/dc/data_new/VAE_test"
SAVE_INTERVAL = 500  # 每100个分子保存一次中间文件

os.makedirs(SAVE_DIR, exist_ok=True)


def load_models():
    vocab = Vocab([x.strip() for x in open(VOCAB_PATH)])

    cjtvae = JTNNVAE(
        vocab=vocab,
        hidden_size=256,
        latent_size=LATENT_DIM,
        depthT=15,
        depthG=3,
        cond_dim=0,
    )

    ckpt = torch.load(CJTVAE_CKPT, map_location=DEVICE)
    cjtvae.load_state_dict(ckpt, strict=False)
    cjtvae = cjtvae.to(DEVICE).eval()

    return cjtvae, vocab


def validity():
    print(f"Using device: {DEVICE}")
    print("Loading VAE model...")

    cjtvae, vocab = load_models()

    valid_count = 0
    total_count = 0
    all_results = []

    total_samples = N_LATENT_VECTORS * DECODES_PER_VECTOR

    latent_vector_counter = 1  # 从1开始计数的潜向量序号

    for i in tqdm(range(0, N_LATENT_VECTORS, BATCH_SIZE)):
        current_batch_size = min(BATCH_SIZE, N_LATENT_VECTORS - i)

        # 从标准正态分布采样
        z = torch.randn(current_batch_size, LATENT_DIM).to(DEVICE)

        # 解码
        for a in range(0, DECODES_PER_VECTOR):
            with torch.no_grad():
                smiles_list = cjtvae.decode_from_latent(z, prob_decode=False)

            # 检查有效性
            for j, smi in enumerate(smiles_list):
                total_count += 1
                if smi is not None:
                    is_valid = Chem.MolFromSmiles(smi) is not None
                else:
                    is_valid = False

                if is_valid:
                    valid_count += 1

                # 保存结果，latent_vector_id从1开始按顺序递增
                latent_vector_id = latent_vector_counter + j
                all_results.append({
                    'latent_vector_id': latent_vector_id,
                    'smiles': smi,
                    'valid': is_valid
                })

            # 每SAVE_INTERVAL个分子保存一次
            if total_count % SAVE_INTERVAL == 0:
                df_temp = pd.DataFrame(all_results)
                df_temp.to_excel(os.path.join(SAVE_DIR, f'intermediate_results_{total_count}.xlsx'), index=False)

        # 更新潜向量计数器
        latent_vector_counter += current_batch_size

    # 保存最终结果
    df = pd.DataFrame(all_results)
    df.to_excel(os.path.join(SAVE_DIR, 'validity_test_results.xlsx'), index=False)

    validity_rate = valid_count / total_count * 100

    print(f'分子总数{len(all_results)}')
    print(f'有效率{validity_rate}')

    return validity_rate


if __name__ == "__main__":
    validity()