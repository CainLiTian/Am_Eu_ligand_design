import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger, Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt
from jtnn_vae import JTNNVAE
from vocab import Vocab
import os

RDLogger.DisableLog("rdApp.*")

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

N_SAMPLES = 100
LATENT_DIM = 48

CJTVAE_CKPT = "/home/dc/data_new/23/best_cjtvae_cond.pth"
VOCAB_PATH = "/home/dc/data_new/new_vocab.txt"

SAVE_DIR = "/home/dc/data_new/VAE_test"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_models():
    vocab = Vocab([x.strip() for x in open(VOCAB_PATH)])

    model = JTNNVAE(
        vocab=vocab,
        hidden_size=256,
        latent_size=LATENT_DIM,
        depthT=15,
        depthG=3,
        cond_dim=0,
    )

    ckpt = torch.load(CJTVAE_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt, strict=False)

    model = model.to(DEVICE).eval()

    return model


def tanimoto_similarity(sm1, sm2):

    if sm1 is None or sm2 is None:
        return None

    mol1 = Chem.MolFromSmiles(sm1)
    mol2 = Chem.MolFromSmiles(sm2)

    if mol1 is None or mol2 is None:
        return None

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def latent_smoothness_test():

    print("Loading model...")
    model = load_models()

    # σ 从0到1
    sigma_list = np.linspace(0, 0.6, 16)

    similarity_dict = {s: [] for s in sigma_list}

    print("Running smoothness test...")

    for i in tqdm(range(N_SAMPLES)):

        # 随机采样latent
        z = torch.randn(1, LATENT_DIM).to(DEVICE)

        # 原始分子
        base_smiles = model.decode_from_latent(z)[0]

        if base_smiles is None:
            continue

        for sigma in sigma_list:

            noise = torch.randn_like(z) * sigma
            z_perturbed = z + noise

            new_smiles = model.decode_from_latent(z_perturbed)[0]

            sim = tanimoto_similarity(base_smiles, new_smiles)

            if sim is not None:
                similarity_dict[sigma].append(sim)

    # 计算平均相似度
    mean_similarity = []

    for sigma in sigma_list:

        sims = similarity_dict[sigma]

        if len(sims) > 0:
            mean_similarity.append(np.mean(sims))
        else:
            mean_similarity.append(0)

    # 保存结果
    df = pd.DataFrame({
        "sigma": sigma_list,
        "mean_similarity": mean_similarity
    })

    df.to_excel(os.path.join(SAVE_DIR, "latent_smoothness.xlsx"), index=False)

    # 画图
    plt.figure(figsize=(6,4))
    plt.plot(sigma_list, mean_similarity, marker='o')
    plt.xlabel("Noise σ")
    plt.ylabel("Average Tanimoto Similarity")
    plt.title("Latent Space Smoothness Test")
    plt.grid()

    plt.savefig(os.path.join(SAVE_DIR, "latent_smoothness.png"), dpi=300)
    plt.show()

    print("Finished!")


if __name__ == "__main__":
    latent_smoothness_test()
