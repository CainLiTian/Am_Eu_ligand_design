import torch
import pandas as pd
import numpy as np
from rdkit import RDLogger
from tqdm import tqdm
from env import CJTVAE_XGB_Env
from agent import MolecularRLAgent
from jtnn_vae import JTNNVAE
from vocab import Vocab
import pickle
import os

RDLogger.DisableLog("rdApp.*")

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 48
BATCH_SIZE = 4
PPO_STEPS = 3
MAX_RETRY = 10  # 分子无效时的最大重试次数
NUM_SAMPLES = 3  # 采样次数，选最好的

SMILE_FILE = "/home/dc/data_new/S.xlsx"
PPO_CHECKPOINT = "/home/dc/data_new/ppo_results/ppo/checkpoint_step100.pth"
CJTVAE_CKPT = "/home/dc/data_new/finetune/best_cjtvae_cond_finetuned_final.pth"
VOCAB_PATH = "/home/dc/data_new/new_vocab.txt"
XGB_MODEL_PATH = "/home/dc/data_new/XGB/XGB.pkl"
OUTPUT_DIR = "/home/dc/data_new/ppo_results/generation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_models():
    vocab = Vocab([x.strip() for x in open(VOCAB_PATH)])
    cjtvae = JTNNVAE(vocab=vocab, hidden_size=256, latent_size=LATENT_DIM, depthT=15, depthG=3, cond_dim=0)
    ckpt = torch.load(CJTVAE_CKPT, map_location=DEVICE)
    cjtvae.load_state_dict(ckpt, strict=False)
    cjtvae = cjtvae.to(DEVICE).eval()
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    return cjtvae, xgb_model, vocab


def is_valid_smiles(smiles):
    return smiles is not None and isinstance(smiles, str) and len(smiles) > 3


def try_generate_valid(agent, env, current_z_i, target_smiles):
    """尝试生成一个有效分子，最多重试MAX_RETRY次，返回有效分子和对应的z"""
    for retry in range(MAX_RETRY):
        action, _ = agent.act(current_z_i)
        temp_z = current_z_i + action
        smi, valid = env.decode(temp_z)

        if is_valid_smiles(smi[0]):
            return temp_z, smi[0], valid[0]

    # 重试失败，返回最后一次的结果
    return temp_z, smi[0], valid[0]


def generate_valid_step(agent, env, current_z, target_smiles_list, step_num):
    """对每个分子：采样NUM_SAMPLES次，每次内部重试MAX_RETRY次，选SF最好的"""
    batch_size = current_z.shape[0]
    next_z = current_z.clone()
    smiles_list = [None] * batch_size
    valid_mask_list = [False] * batch_size

    for i in range(batch_size):
        best_sf = -np.inf
        best_z = None
        best_smi = None
        best_valid = False

        # 采样NUM_SAMPLES次
        for sample_idx in range(NUM_SAMPLES):
            # 内部重试MAX_RETRY次，直到得到有效分子
            temp_z, smi, valid = try_generate_valid(agent, env, current_z[i:i+1], target_smiles_list[i])

            if is_valid_smiles(smi):
                # 修改：使用精确匹配模式计算reward
                _, info = env.compute_reward(
                    temp_z, [smi], [valid],
                    target_literature_smiles=target_smiles_list[i],
                    use_exact_match=True
                )
                sf = info[0].get("best_sf", -np.inf) if info else -np.inf

                if sf > best_sf:
                    best_sf = sf
                    best_z = temp_z.clone()
                    best_smi = smi
                    best_valid = valid

        if best_z is not None:
            next_z[i:i+1] = best_z
            smiles_list[i] = best_smi
            valid_mask_list[i] = best_valid
            print(f"       Sample {i}: best SF = {best_sf:.4f}")
        else:
            print(f"       Sample {i}: failed to generate valid molecule")

    valid_count = sum(1 for s in smiles_list if is_valid_smiles(s))
    print(f"     ✅ Step {step_num}: {valid_count}/{batch_size} valid")

    return next_z, smiles_list, valid_mask_list


def save_results(all_results, output_file):
    df = pd.DataFrame(all_results)
    df.to_excel(output_file, index=False)
    print(f"  💾 Saved to {output_file}")


def main():
    print(f"Using device: {DEVICE}")

    print("Loading models...")
    cjtvae, xgb_model, vocab = load_models()
    env = CJTVAE_XGB_Env(cjtvae_model=cjtvae, xgb_model=xgb_model, device=DEVICE)

    agent = MolecularRLAgent(z_dim=LATENT_DIM).to(DEVICE)
    checkpoint = torch.load(PPO_CHECKPOINT, map_location=DEVICE)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    print("Models loaded\n")

    print(f"Reading {SMILE_FILE}...")
    df = pd.read_excel(SMILE_FILE)
    sf_col = 'SF' if 'SF' in df.columns else 'soft_sf' if 'soft_sf' in df.columns else df.columns[1]
    df_grouped = df.groupby('SMILES')[sf_col].mean().reset_index()
    df_grouped.columns = ['SMILES', 'SF']
    unique_smiles = df_grouped['SMILES'].tolist()
    original_sf_dict = dict(zip(df_grouped['SMILES'], df_grouped['SF']))
    print(f"Unique SMILES: {len(unique_smiles)}\n")

    all_results = []
    n_batches = (len(unique_smiles) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(unique_smiles))
        batch_smiles = unique_smiles[start_idx:end_idx]
        batch_original_sf = [original_sf_dict[smi] for smi in batch_smiles]
        current_batch_size = len(batch_smiles)

        print(f"\n{'=' * 60}")
        print(f"📦 Batch {batch_idx + 1}/{n_batches} ({current_batch_size} molecules)")
        print(f"{'=' * 60}")

        print("  🔄 Encoding SMILES to latent vectors...")
        _, _, z_combined = cjtvae.get_sampled_latent_vector(batch_smiles)
        z_combined = z_combined.to(DEVICE)
        print("  ✅ Encoding done")

        current_z = z_combined.clone()
        trajectory_smiles = []
        trajectory_sf = []

        for step in range(PPO_STEPS):
            print(f"\n  🚀 Step {step + 1}/{PPO_STEPS}")
            # 修改：传入batch_smiles作为目标文献SMILES
            current_z, smiles, valid_mask = generate_valid_step(agent, env, current_z, batch_smiles, step + 1)

            sf_list = []
            for j, (smi, valid) in enumerate(zip(smiles, valid_mask)):
                if valid and smi and is_valid_smiles(smi):
                    # 修改：使用精确匹配模式计算reward
                    _, info = env.compute_reward(
                        current_z[j:j+1], [smi], [valid],
                        target_literature_smiles=batch_smiles[j],
                        use_exact_match=True
                    )
                    sf = info[0].get("best_sf", np.nan) if info else np.nan
                else:
                    sf = np.nan
                sf_list.append(sf)

            trajectory_smiles.append(smiles)
            trajectory_sf.append(sf_list)
            print(f"     📊 Avg SF: {np.nanmean(sf_list):.4f}")

        for j in range(current_batch_size):
            result = {
                'original_smiles': batch_smiles[j],
                'original_sf': batch_original_sf[j],
            }
            for step in range(PPO_STEPS):
                result[f'step_{step + 1}_smiles'] = trajectory_smiles[step][j]
                result[f'step_{step + 1}_sf'] = trajectory_sf[step][j]
            result['sf_improvement'] = result['step_3_sf'] - result['original_sf'] if not pd.isna(
                result['step_3_sf']) else np.nan
            all_results.append(result)

        temp_file = os.path.join(OUTPUT_DIR, f"batch_{batch_idx + 1}_temp.xlsx")
        save_results(all_results, temp_file)

        full_file = os.path.join(OUTPUT_DIR, "generation_results.xlsx")
        save_results(all_results, full_file)

        batch_improved = sum(1 for r in all_results[-current_batch_size:] if
                             not pd.isna(r['sf_improvement']) and r['sf_improvement'] > 0)
        print(f"\n  📈 Batch {batch_idx + 1} summary: {batch_improved}/{current_batch_size} improved")

    valid_improvements = [r['sf_improvement'] for r in all_results if not pd.isna(r['sf_improvement'])]
    print(f"\n{'=' * 60}")
    print(f"✅ ALL BATCHES COMPLETED!")
    print(f"{'=' * 60}")
    print(f"📊 Total molecules: {len(all_results)}")
    print(f"📈 Improved: {sum(1 for r in all_results if not pd.isna(r['sf_improvement']) and r['sf_improvement'] > 0)}")
    print(f"📉 Degraded: {sum(1 for r in all_results if not pd.isna(r['sf_improvement']) and r['sf_improvement'] < 0)}")
    print(f"📊 Avg improvement: {np.nanmean(valid_improvements):.4f}")
    print(f"📁 Results saved to {OUTPUT_DIR}/generation_results.xlsx")


if __name__ == "__main__":
    main()