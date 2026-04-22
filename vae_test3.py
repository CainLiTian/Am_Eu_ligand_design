import torch
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
import os
from tqdm import tqdm

from jtnn_vae import JTNNVAE
from vocab import Vocab

RDLogger.DisableLog("rdApp.*")

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 48

CJTVAE_CKPT = "/home/dc/data_new/23/best_cjtvae_cond.pth"
VOCAB_PATH = "/home/dc/data_new/new_vocab.txt"

SAVE_DIR = "/home/dc/data_new/VAE_test/latent"
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_SIMS = [1.0, 0.7, 0.55, 0.4, 0.25]
SIGMA_LIST = [0, 0.1, 0.2, 0.3, 0.4]
TOLERANCE = 0.05
NUM_SUCCESS_CASES = 10


def load_model():
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


def latent_perturbation_visualization():
    print("Loading model...")
    model = load_model()

    success_count = 0
    attempt_count = 0

    while success_count < NUM_SUCCESS_CASES:
        attempt_count += 1
        print(f"\n{'=' * 50}")
        print(f"Starting sampling attempt #{attempt_count} (Successes so far: {success_count}/{NUM_SUCCESS_CASES})")
        print(f"{'=' * 50}")

        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        mols = []
        legends = []

        base_smiles = model.decode_from_latent(z)[0]
        if base_smiles is None:
            print("Base molecule invalid, retrying...")
            continue

        base_mol = Chem.MolFromSmiles(base_smiles)
        base_fp = AllChem.GetMorganFingerprintAsBitVect(base_mol, 2, nBits=2048)

        mols.append(base_mol)
        legends.append(f"sigma = {SIGMA_LIST[0]}\nTanimoto: 1.00")
        print(f"Base molecule (sigma=0): {base_smiles}")

        current_z = z.clone()
        success = True

        for idx in tqdm(range(1, len(SIGMA_LIST)), desc=f"Generating molecules for case #{attempt_count}"):
            sigma = SIGMA_LIST[idx]
            target_sim = TARGET_SIMS[idx]
            print(f"\nTrying to generate molecule with sigma={sigma}, target similarity={target_sim}±{TOLERANCE}")

            found_valid = False
            for attempt in range(10):
                noise = torch.randn_like(current_z) * sigma
                z_new = current_z + noise
                new_smiles = model.decode_from_latent(z_new)[0]

                if new_smiles is not None:
                    mol = Chem.MolFromSmiles(new_smiles)
                    if mol is not None:
                        new_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        sim = DataStructs.TanimotoSimilarity(base_fp, new_fp)
                        print(f"  Attempt {attempt + 1}: similarity={sim:.3f}")

                        if abs(sim - target_sim) <= TOLERANCE:
                            mols.append(mol)
                            legends.append(f"sigma = {sigma}\nTanimoto: {sim:.2f}")
                            current_z = z_new
                            found_valid = True
                            break

            if not found_valid:
                print(f"  Failed to find molecule with required similarity for sigma={sigma} after 10 attempts")
                success = False
                break

        if success:
            success_count += 1
            print(f"\n✓ Successfully generated case #{success_count}!")

            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=len(mols),
                subImgSize=(300, 300),
                legends=legends
            )

            save_path = os.path.join(SAVE_DIR, f"latent_perturbation_case2_{success_count}.png")
            img.save(save_path)
            print(f"  Image saved to: {save_path}")
        else:
            print(f"\n✗ Failed to generate all molecules for this attempt")

    print(f"\n{'=' * 50}")
    print(f"Completed! Generated {NUM_SUCCESS_CASES} successful cases after {attempt_count} attempts")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    latent_perturbation_visualization()