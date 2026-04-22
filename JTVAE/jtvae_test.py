import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger, Chem
from rdkit.Contrib.SA_Score import sascorer
from jtnn_vae import JTNNVAE
from vocab import Vocab
import os

RDLogger.DisableLog("rdApp.*")
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 配置参数
LATENT_DIM = 48
CJTVAE_CKPT = "/home/dc/data_new/23/best_cjtvae_cond.pth"
VOCAB_PATH = "/home/dc/data_new/new_vocab.txt"
N_SAMPLES = 1000  # 测试样本数
BATCH_SIZE = 10
SAVE_DIR = "/home/dc/data_new/VAE_test"
SAVE_INTERVAL = 100  # 每100个分子保存一次中间文件

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


def calculate_sa_score(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return sascorer.calculateScore(mol)
    except:
        return None


def validity():
    print(f"Using device: {DEVICE}")
    print("Loading VAE model...")

    cjtvae, vocab = load_models()

    valid_count = 0
    total_count = 0
    all_results = []

    print(f"\nTesting {N_SAMPLES} random latent vectors...")

    for i in tqdm(range(0, N_SAMPLES, BATCH_SIZE)):
        current_batch_size = min(BATCH_SIZE, N_SAMPLES - i)

        # 从标准正态分布采样
        z = torch.randn(current_batch_size, LATENT_DIM).to(DEVICE)

        # 解码
        with torch.no_grad():
            smiles_list = cjtvae.decode_from_latent(z, prob_decode=False)

        # 检查有效性
        for smi in smiles_list:
            total_count += 1

            # 计算SAscore
            sa_score = calculate_sa_score(smi) if smi else None
            is_valid = sa_score is not None

            if is_valid:
                valid_count += 1

            # 保存结果
            all_results.append({
                'smiles': smi,
                'valid': is_valid,
                'sa_score': sa_score
            })

            # 每SAVE_INTERVAL个分子保存一次
            if total_count % SAVE_INTERVAL == 0:
                df_temp = pd.DataFrame(all_results)
                df_temp.to_excel(os.path.join(SAVE_DIR, f'intermediate_results_{total_count}.xlsx'), index=False)

    # 保存最终结果
    df = pd.DataFrame(all_results)
    df.to_excel(os.path.join(SAVE_DIR, 'pre_vae_test_results.xlsx'), index=False)

    validity_rate = valid_count / total_count * 100

    print("\n" + "=" * 50)
    print("VAE有效性测试结果")
    print("=" * 50)
    print(f"测试样本数: {total_count}")
    print(f"有效分子数: {valid_count}")
    print(f"有效率: {validity_rate:.2f}%")

    if valid_count > 0:
        valid_df = df[df['valid'] == True]
        print(f"\nSAscore统计:")
        print(f"  平均SAscore: {valid_df['sa_score'].mean():.3f}")
        print(f"  最小SAscore: {valid_df['sa_score'].min():.3f}")
        print(f"  最大SAscore: {valid_df['sa_score'].max():.3f}")

    return validity_rate


if __name__ == "__main__":
    validity()