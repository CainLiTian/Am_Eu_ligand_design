import os
import time
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm.auto import tqdm
from pandas import DataFrame
import warnings
import multiprocessing as mp
import matplotlib.pyplot as plt
from datautils import tensorize
from jtnn_vae import JTNNVAE   # 已改为支持 cond_dim, forward(mol_batch, beta, cond_batch=...)
from vocab import Vocab
from collections import defaultdict
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from datautils import MolTreeFolder
import torch.nn as nn

writer = SummaryWriter("runs/jtnn_cond")
warnings.filterwarnings("ignore")

config = {
    "processed_data": "/home/dc/data_new/data_pca.xlsx",
    "mol_pkl": "/home/dc/data_new/new_moltree.pkl",
    "vocab_path": "/home/dc/data_new/new_vocab.txt",

    "output_path": "/home/dc/data_new/24/cjtvae_history.xlsx",
    "model_save": "/home/dc/data_new/24/best_cjtvae_cond.pth",
    "plot_path": "/home/dc/data_new/24/cjtvae_loss.png",

    "hidden_size": 256,
    "latent_size": 48,
    "depthT": 15,
    "depthG": 3,

    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "beta_max": 0.1,  # KL weight
    "warm_up": 20,

    "use_cosine": True,
    "cosine_T_max": 60,
    "use_plateau": True,
    "plateau_patience": 5,
    "plateau_factor": 0.5,

    "early_stop_patience": 10,
    "min_delta": 1e-4,

    "num_workers": 0,
    "pin_memory": True
}


def _decode_worker(model, z_tree, z_mol, cond, queue):
    try:
        smiles = model.decode(z_tree, z_mol, cond)
        queue.put(smiles)
    except:
        queue.put(None)


def safe_decode(model, z_tree, z_mol, cond=None, timeout=5):
    queue = mp.Queue()
    p = mp.Process(
        target=_decode_worker,
        args=(model, z_tree, z_mol, cond, queue)
    )

    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None

    if queue.empty():
        return None

    return queue.get()


def plot_metric(history,key,title,ylabel,path):
    epochs = range(1,len(history[key])+1)
    plt.figure()
    plt.plot(epochs,history[key],linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.savefig(path,dpi=300)
    plt.show()


def evaluate_validity(model, device, latent_size=48, cond_dim=8, n_sample=100):
    valid = 0
    total = 0

    for _ in range(n_sample):
        z_tree = torch.randn(1, latent_size).to(device)
        z_mol = torch.randn(1, latent_size).to(device)
        cond = torch.zeros(1, cond_dim).to(device) if cond_dim > 0 else None
        smiles = safe_decode(model, z_tree, z_mol, cond, timeout=5)

        if smiles is None:
            continue
        mol = Chem.MolFromSmiles(smiles)
        total += 1
        if mol is not None:
            valid += 1
    if total == 0:
        return 0

    return valid / total


def tanimoto(sm1, sm2):

    m1 = Chem.MolFromSmiles(sm1)
    m2 = Chem.MolFromSmiles(sm2)
    if m1 is None or m2 is None:
        return 0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 2048)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def evaluate_reconstruction(model, val_loader, device):
    sims = []
    for batch in val_loader:
        smiles_list = batch["smiles"]
        jtenc_holder, mpn_holder = batch["tensor"]
        tree_vec, mol_vec = model.encode(jtenc_holder, mpn_holder)
        for i, sm in enumerate(smiles_list):
            z_tree = tree_vec[i].unsqueeze(0)
            z_mol = mol_vec[i].unsqueeze(0)
            recon = safe_decode(model, z_tree, z_mol, None, timeout=5)
            if recon is None:
                continue
            sim = tanimoto(sm, recon)
            sims.append(sim)
    if len(sims) == 0:
        return 0

    return np.mean(sims)


def plot_latent_space(model, loader, device, save_path):
    model.eval()
    latents = []

    with torch.no_grad():
        for mol_batch, cond_batch, jtenc_holder, mpn_holder, jtmpn_holder in loader:
            jtenc_holder = [x.to(device) for x in jtenc_holder]
            mpn_holder = [x.to(device) for x in mpn_holder]
            mean, _ = model.encode_latent(jtenc_holder, mpn_holder)
            latents.append(mean.cpu().numpy())

    latents = np.concatenate(latents,axis=0)
    pca = PCA(n_components=2)
    z = pca.fit_transform(latents)
    plt.figure(figsize=(6,6))
    plt.scatter(z[:,0],z[:,1],s=5,alpha=0.6)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title("Latent Space Clustering")
    plt.grid(alpha=0.3)
    plt.savefig(save_path,dpi=300)
    plt.show()



class MolTreeCondDataset(Dataset):

    def __init__(self, moltrees, cond_matrix):
        assert len(moltrees) == len(cond_matrix), "moltrees length must equal cond rows"
        self.moltrees = moltrees
        self.cond = cond_matrix.astype(np.float32)

    def __len__(self):
        return len(self.moltrees)

    def __getitem__(self, idx):
        return self.moltrees[idx], self.cond[idx]

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

def collate_moltree_cond(batch):
    mols = [item[0] for item in batch]
    conds = np.stack([item[1] for item in batch], axis=0)
    conds = torch.from_numpy(conds).float()
    return mols, conds

def beta_schedule_linear(ep):
    return min(config["beta_max"], config["beta_max"] * ep / config["warm_up"])


def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    # 监控BN层的运行统计
    if ep == 1:  # 只在第一轮记录
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                module.track_running_stats = True
    total_graphs = 0
    total_loss = total_tree_acc = total_graph_acc = total_stereo_acc = 0
    total_kl = total_word_loss = total_topo_loss = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for mol_batch, cond_batch, jtenc_holder, mpn_holder, jtmpn_holder in pbar:

        # --- Move cond + holders to GPU ---
        cond_batch = cond_batch.to(device)
        jtenc_holder = [x.to(device) for x in jtenc_holder]
        mpn_holder = [x.to(device) for x in mpn_holder]

        if jtmpn_holder is not None:
            jtmpn_holder = (
                [x.to(device) for x in jtmpn_holder[0]],
                jtmpn_holder[1].to(device)
            )

        batch_input = (mol_batch, jtenc_holder, mpn_holder, jtmpn_holder)

        # ========== forward ==========
        loss, kl, tacc, gacc, sacc, word_loss, topo_loss = model(
            batch_input,
            beta=beta,
            cond_batch=cond_batch
        )

        # ======= backward =======
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # ======= accumulate =======
        n = len(mol_batch)
        total_graphs += n

        total_loss += loss.item() * n
        total_tree_acc += tacc * n
        total_graph_acc += gacc * n
        total_stereo_acc += sacc * n
        total_kl += kl * n
        total_word_loss += word_loss.item() * n
        total_topo_loss += topo_loss.item() * n

    return (
        total_loss / total_graphs,
        total_tree_acc / total_graphs,
        total_graph_acc / total_graphs,
        total_stereo_acc / total_graphs,
        total_kl / total_graphs,
        total_word_loss / total_graphs,
        total_topo_loss / total_graphs
    )




def valid_epoch(model, loader, device, beta):
    model.eval()

    total_graphs = 0
    total_loss = total_tree_acc = total_graph_acc = total_stereo_acc = 0
    total_kl = total_word_loss = total_topo_loss = 0

    with torch.no_grad():
        for mol_batch, cond_batch, jtenc_holder, mpn_holder, jtmpn_holder in loader:

            cond_batch = cond_batch.to(device)
            jtenc_holder = [x.to(device) for x in jtenc_holder]
            mpn_holder = [x.to(device) for x in mpn_holder]

            if jtmpn_holder is not None:
                jtmpn_holder = (
                    [x.to(device) for x in jtmpn_holder[0]],
                    jtmpn_holder[1].to(device)
                )

            batch_input = (mol_batch, jtenc_holder, mpn_holder, jtmpn_holder)

            # --------- forward ---------
            loss, kl, tacc, gacc, sacc, word_loss, topo_loss = model(
                batch_input,
                beta=beta,
                cond_batch=cond_batch
            )

            n = len(mol_batch)
            total_graphs += n

            total_loss += loss.item() * n
            total_tree_acc += tacc * n
            total_graph_acc += gacc * n
            total_stereo_acc += sacc * n
            total_kl += kl * n
            total_word_loss += word_loss.item() * n
            total_topo_loss += topo_loss.item() * n

    return (
        total_loss / total_graphs,
        total_tree_acc / total_graphs,
        total_graph_acc / total_graphs,
        total_stereo_acc / total_graphs,
        total_kl / total_graphs,
        total_word_loss / total_graphs,
        total_topo_loss / total_graphs
    )





def plot_training_curves(history, save_path=None):
    epochs = range(1, len(history['tr_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # --- subplot 1: Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['tr_loss'], label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['tr_tacc'], label='Tree Acc', linewidth=2)
    plt.plot(epochs, history['val_tacc'], label='Val Tree Acc', linewidth=2)
    plt.plot(epochs, history['tr_gacc'], label='Graph Acc', linewidth=2)
    plt.plot(epochs, history['val_gacc'], label='Val Graph Acc', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved training curves to {save_path}")
    plt.show()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    print("Loading moltrees and vocab...")
    with open(config["mol_pkl"], "rb") as f:
        moltrees = pickle.load(f)
    vocab = Vocab([x.strip() for x in open(config["vocab_path"])])
    print(f"Loaded moltrees: {len(moltrees)}, vocab size: {len(vocab.vocab)}")

    df = pd.read_excel(config["processed_data"],sheet_name="Sheet1")
    if "Ligand_SMILES" not in df.columns:
        raise RuntimeError("processed_data must contain 'Ligand_SMILES' column")
    feat_cols = [c for c in df.columns if c.startswith("solv_pca_")]
    if len(feat_cols) == 0:
        raise RuntimeError("No feat_* columns found in processed_data")
    cond_matrix = df[feat_cols].to_numpy(dtype=np.float32)
    print(f"Loaded processed_data rows: {len(df)}, cond_dim: {cond_matrix.shape[1]}")

    if len(moltrees) != len(cond_matrix):
        raise RuntimeError(f"Length mismatch: moltrees={len(moltrees)} vs cond rows={len(cond_matrix)}. Use processed_data_filtered.xlsx that aligns with cond_moltrees.pkl")

    from sklearn.model_selection import train_test_split

    grouped = {}  # {smiles: [(moltree, cond_vec), ...]}

    for mol, cond_vec in zip(moltrees, cond_matrix):
        smi = mol.smiles
        if smi not in grouped:
            grouped[smi] = []
        grouped[smi].append((mol, cond_vec))

    all_smiles = np.array(list(grouped.keys()))

    train_smi, test_smi = train_test_split(all_smiles, test_size=0.1, random_state=42)
    train_smi, val_smi = train_test_split(train_smi, test_size=0.1, random_state=42)

    # 转成集合方便查询
    train_smi = set(train_smi)
    val_smi = set(val_smi)
    test_smi = set(test_smi)
    def collect_samples(smiles_set):
        samples = []  # (moltree, cond_vec)
        for smi in smiles_set:
            for mol, cond_vec in grouped[smi]:
                samples.append((mol, cond_vec))
        return samples


    train_data = collect_samples(train_smi)
    val_data = collect_samples(val_smi)
    test_data = collect_samples(test_smi)

    train_loader = MolTreeFolder(
        data=train_data,
        vocab=vocab,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        assm=False
    )

    val_loader = MolTreeFolder(
        data=val_data,
        vocab=vocab,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        assm=False
    )

    test_loader = MolTreeFolder(
        data=test_data,
        vocab=vocab,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        assm=False
    )

    cond_dim = cond_matrix.shape[1]
    model = JTNNVAE(vocab,
                    config['hidden_size'],
                    config['latent_size'],
                    config['depthT'],
                    config['depthG'],
                    cond_dim=cond_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    schedulers = []
    if config['use_cosine']:
        schedulers.append(CosineAnnealingLR(optimizer, T_max=config['cosine_T_max']))
    if config['use_plateau']:
        schedulers.append(ReduceLROnPlateau(optimizer, mode='min',
                                            patience=config['plateau_patience'],
                                            factor=config['plateau_factor']))

    history = {'tr_loss': [], 'tr_tacc': [], 'tr_gacc': [], 'tr_sacc': [], 'tr_g_loss': [], 'tr_t_loss': [],
               'val_loss': [], 'val_tacc': [], 'val_gacc': [], 'val_sacc': [], 'val_g_loss': [], 'val_t_loss': [],
               'tr_kl': [], 'val_kl': [], 'lr': [], 'validity': [], 'reconstruction': []}
    best_val_loss = float('inf')
    no_improve = 0

    print("-" * 60)
    for ep in range(1, config['epochs'] + 1):
        start = time.time()

        # === 训练 & 验证 ===
        tr_loss, tr_tacc, tr_gacc, tr_sacc, tr_kl, tr_g_loss, tr_t_loss = train_epoch(
            model, train_loader, optimizer, device, beta_schedule_linear(ep)
        )
        val_loss, val_tacc, val_gacc, val_sacc, val_kl, val_g_loss, val_t_loss = valid_epoch(
            model, val_loader, device, beta_schedule_linear(ep)
        )
        lr = optimizer.param_groups[0]['lr']

        validity = evaluate_validity(model, device)
        recon = evaluate_reconstruction(model, val_loader, device)
        end = time.time()

        # === 保存历史 ===
        history['tr_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['tr_tacc'].append(tr_tacc)
        history['val_tacc'].append(val_tacc)
        history['tr_gacc'].append(tr_gacc)
        history['val_gacc'].append(val_gacc)
        history['tr_sacc'].append(tr_sacc)
        history['val_sacc'].append(val_sacc)
        history['tr_g_loss'].append(tr_g_loss)
        history['tr_t_loss'].append(tr_t_loss)
        history['val_g_loss'].append(val_g_loss)
        history['val_t_loss'].append(val_t_loss)
        history['tr_kl'].append(tr_kl)
        history['val_kl'].append(val_kl)
        history['lr'].append(lr)
        history['validity'].append(validity)
        history['reconstruction'].append(recon)

        print(f"Reconstruction: {recon:.4f}")

        # === 学习率调度 ===
        for sch in schedulers:
            if isinstance(sch, ReduceLROnPlateau):
                sch.step(val_loss)
            else:
                sch.step()

        # === 输出信息 ===
        print(f"\nEpoch {ep} | Time: {end - start:.1f}s")
        print(f"Train Loss: {tr_loss:.4f}  | KL: {tr_kl:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | KL: {val_kl:.4f}")
        print(f"Train Acc: {tr_tacc:.4f}   | Val Acc: {val_tacc:.4f}")
        print(f"Validity: {validity:.4f}   | Reconstruction: {recon:.4f}")
        print(f"Learning Rate: {lr:.2e}")
        print("-" * 60)

        # === 早停机制 ===
        if val_loss + config['min_delta'] < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), config['model_save'])
            print(f"Saved improved model to {config['model_save']}")
        else:
            no_improve += 1
            if no_improve >= config['early_stop_patience']:
                print(f"Early stopping triggered at epoch {ep}")
                break

    DataFrame(history).to_excel(config['output_path'])
    plot_training_curves(history, save_path=config["plot_path"])
    plot_metric(history, "validity", "Validity Curve", "Validity",
                "/home/dc/data_new/24/validity.png")

    plot_metric(history, "reconstruction", "Reconstruction Curve",
                "Tanimoto Similarity",
                "/home/dc/data_new/24/reconstruction.png")
    plot_latent_space(
        model,
        val_loader,
        device,
        "/home/dc/data_new/24/latent_pca.png"
    )

    print("Training finished.")