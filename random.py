import pickle
import torch
import pandas as pd
from tqdm import trange
from rdkit import RDLogger
from collections import defaultdict
from envv import CJTVAE_XGB_Env
from agent import MolecularRLAgent, MultiStepPPOTrainer
from jtnn_vae import JTNNVAE
from vocab import Vocab
import os
import numpy as np
import signal
import time
from plot import create_fingerprint_umap,plot_training_metrics, plot_training_dashboard, plot_reward_comparison, plot_baseline_results
from control_group import (
    RandomSearchBaseline,
    GreedySearchBaseline,
    CEMBaseline,
    BaselineConfig,
)


RDLogger.DisableLog("rdApp.*")
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 48
BATCH_SIZE = 16

TOTAL_UPDATES = 100
LOG_INTERVAL = 50
SAVE_INTERVAL = 10


CJTVAE_CKPT = "/home/dc/data_new/finetune/best_cjtvae_cond_finetuned_final.pth"
VOCAB_PATH = "/home/dc/data_new/new_vocab.txt"
XGB_MODEL_PATH = "/home/dc/data_new/XGB/trained_pipeline_best_experiment_rebuilt.pkl"

SAVE_DIR = "/home/dc/data_new/ppo_results"
os.makedirs(SAVE_DIR, exist_ok=True)

baseline_config = BaselineConfig()

baseline_config.LATENT_DIM = LATENT_DIM
baseline_config.BATCH_SIZE = BATCH_SIZE
baseline_config.TOTAL_STEPS = TOTAL_UPDATES
baseline_config.ROLLOUT_STEPS = ROLLOUT_STEPS
baseline_config.SAVE_STEPS = SAVE_INTERVAL


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

    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)

    return cjtvae, xgb_model, vocab



def main():
    print("Using device:", DEVICE)

    cjtvae, xgb_model, vocab = load_models()

    env = CJTVAE_XGB_Env(
        cjtvae_model=cjtvae,
        xgb_model=xgb_model,
        device=DEVICE,
    )

    print("Env ready.")


    # ---------------- Random ----------------

    random_trainer = RandomSearchBaseline(
        env,
        DEVICE,
        baseline_config
    )

    random_df, random_reward_means = random_trainer.run()

    print("Random baseline finished.")




if __name__ == "__main__":
    main()