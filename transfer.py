import os
import time
import pickle
import torch
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm.auto import tqdm
from pandas import DataFrame
import warnings
import multiprocessing as mp
from jtnn_vae import JTNNVAE
from vocab import Vocab
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from datautils import MolTreeFolder
from train_jtnn import plot_training_curves

writer = SummaryWriter("runs/jtnn_cond_finetune")
warnings.filterwarnings("ignore")

config = {
    # 小数据集路径
    "processed_data": "/home/dc/data_new/small_data/small_data2.xlsx",
    "mol_pkl": "/home/dc/data_new/small_data/small_moltree.pkl",
    "vocab_path": "/home/dc/data_new/new_vocab.txt",
    "output_path": "/home/dc/data_new/finetune/cjtvae_finetune_history.xlsx",
    "model_save": "/home/dc/data_new/finetune/best_cjtvae_cond_finetuned.pth",
    "plot_path": "/home/dc/data_new/finetune/cjtvae_finetune_loss.png",

    # 模型结构
    "hidden_size": 256,
    "latent_size": 48,
    "depthT": 15,
    "depthG": 3,

    # 动态微调策略配置
    "finetune_strategy": "progressive",  # "progressive" 或 "fixed"

    # 阶段定义（基于epoch）
    "phase_epochs": [50, 120, 200],  # 各阶段结束的epoch
    "phase_lrs": [1e-4, 5e-4, 1e-4],  # 各阶段初始学习率
    "phase_l2sp_factors": [1.0, 0.5, 0.1],  # L2-SP正则化强度因子

    # 阶段1：
    "phase1_modules": ["decoder", "G_mean", "G_var", "T_mean", "T_var"],

    # 阶段2：
    "phase2_modules": ["decoder", "G_mean", "G_var", "T_mean", "T_var"],

    # 阶段3：
    "phase3_modules": ["decoder", "G_mean", "G_var", "T_mean", "T_var"],

    # 基础超参数
    "batch_size": 8,
    "epochs": 200,
    "initial_lr": 1e-4,
    "weight_decay": 0.0,
    "beta": 0.01,

    # 基础L2-SP权重
    "base_l2sp_lambda": 2,

    # 调度器
    "use_cosine": False,
    "cosine_T_max": 60,
    "use_plateau": True,
    "plateau_patience": 5,
    "plateau_factor": 0.5,

    # 早停
    "early_stop_patience": 10,  # 增加耐心值，因为策略会变化
    "min_delta": 1e-4,

    # 系统
    "num_workers": 0,
    "pin_memory": True,

    # 预训练模型
    "pretrained_ckpt": "/home/dc/data_new/23/best_cjtvae_cond.pth",

    # 数据集分割方式
    "split_method": "by_molecule",  # "by_sample" 或 "by_molecule"
    "train_ratio": 0.70,  # 小数据集建议增加训练比例
    "val_ratio": 0.20,
    "test_ratio": 0.10
}

# 基础L2-SP权重
BASE_L2SP_LAMBDAS = {
    "encoder": 5.0,
    "decoder_rnn": 2.0,
    "decoder_head": 0.5,
    "latent": 1.0,
}


def which_group(name):
    """确定参数所属的分组"""
    if name.startswith(("jtnn", "mpn", "jtmpn")):
        return "encoder"
    elif name.startswith(("T_mean", "T_var", "G_mean", "G_var")):
        return "latent"
    elif "decoder" in name:
        if any(k in name for k in ["W_o", "U_o", "cond_proj"]):
            return "decoder_head"
        else:
            return "decoder_rnn"
    else:
        return None


def get_current_phase(epoch, phase_epochs):
    """根据当前epoch确定所处的阶段"""
    for i, end_epoch in enumerate(phase_epochs):
        if epoch <= end_epoch:
            return i
    return len(phase_epochs) - 1


def apply_freeze_strategy(model, epoch, config, verbose=True):
    """
    根据当前epoch动态应用冻结策略

    Args:
        model: JTNNVAE模型
        epoch: 当前epoch（从1开始）
        config: 配置字典
        verbose: 是否打印详细信息
    """
    current_phase = get_current_phase(epoch, config["phase_epochs"])

    # 确定当前阶段要解冻的模块
    if current_phase == 0:
        train_modules = config["phase1_modules"]
        phase_name = "Phase 1 (Top Layers Only)"
    elif current_phase == 1:
        train_modules = config["phase2_modules"]
        phase_name = "Phase 2 (Decoder + Latent)"
    else:
        train_modules = config["phase3_modules"]
        phase_name = "Phase 3 (Partial Encoder)"

    # 首先冻结所有参数
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 解冻当前阶段指定的模块
    for name, param in model.named_parameters():
        # 检查是否属于当前阶段要解冻的模块
        should_train = False
        for module in train_modules:
            if module in name:
                should_train = True
                break

        # 处理层级解冻（如 jtnn.decoder 表示 jtnn 中的 decoder 层）
        if "." in module:
            module_parts = module.split(".")
            if len(module_parts) == 2:
                parent, child = module_parts
                if parent in name and child in name:
                    should_train = True

        if should_train:
            param.requires_grad = True

    if verbose:
        print(f"\n=== Epoch {epoch}: {phase_name} ===")
        print(f"Current Phase: {current_phase + 1}")
        print(f"Trainable modules: {train_modules}")

        # 统计可训练参数
        total_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)")

        # 显示可训练参数示例
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"\nSample trainable parameters (max 10):")
        for name in trainable_names[:10]:
            print(f"  - {name}")
        if len(trainable_names) > 10:
            print(f"  ... and {len(trainable_names) - 10} more")


def get_current_l2sp_lambdas(epoch, config):
    """根据当前epoch获取L2-SP正则化权重"""
    current_phase = get_current_phase(epoch, config["phase_epochs"])
    factor = config["phase_l2sp_factors"][current_phase]

    # 应用阶段因子
    lambdas = {}
    for key, value in BASE_L2SP_LAMBDAS.items():
        lambdas[key] = value * factor

    return lambdas


def adjust_learning_rate(optimizer, epoch, config):
    """根据阶段调整学习率"""
    current_phase = get_current_phase(epoch, config["phase_epochs"])
    target_lr = config["phase_lrs"][current_phase]

    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']

    # 如果学习率需要调整
    if abs(current_lr - target_lr) > 1e-8:
        for param_group in optimizer.param_groups:
            param_group['lr'] = target_lr

        print(f"Learning rate adjusted: {current_lr:.2e} -> {target_lr:.2e}")
        return True

    return False


def train_epoch(model, loader, optimizer, device, epoch, theta_pretrained):
    """
    训练一个epoch，包含动态L2-SP正则化

    Args:
        epoch: 当前epoch（用于动态策略）
    """
    model.train()

    # 获取当前阶段的L2-SP权重
    l2sp_lambdas = get_current_l2sp_lambdas(epoch, config)

    total_graphs = 0
    total_loss = total_tree_acc = total_graph_acc = total_stereo_acc = 0
    total_kl = total_word_loss = total_topo_loss = 0
    total_l2sp = 0

    # 计算批次数量
    batch_count = 0

    pbar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)

    for mol_batch, cond_batch, jtenc_holder, mpn_holder, jtmpn_holder in pbar:
        batch_count += 1

        # 移动数据到设备
        cond_batch = cond_batch.to(device)
        jtenc_holder = [x.to(device) for x in jtenc_holder]
        mpn_holder = [x.to(device) for x in mpn_holder]

        if jtmpn_holder is not None:
            jtmpn_holder = (
                [x.to(device) for x in jtmpn_holder[0]],
                jtmpn_holder[1].to(device)
            )

        batch_input = (mol_batch, jtenc_holder, mpn_holder, jtmpn_holder)

        # ========== 前向传播 ==========
        loss, kl, tacc, gacc, sacc, word_loss, topo_loss = model(
            batch_input,
            beta=config['beta'],
            cond_batch=cond_batch
        )

        # ===== 动态L2-SP正则化 =====
        l2sp_loss = 0.0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            group = which_group(name)
            if group is None or group not in l2sp_lambdas:
                continue
            lam = l2sp_lambdas[group]
            if name in theta_pretrained:
                l2sp_loss = l2sp_loss + lam * torch.sum((p - theta_pretrained[name]) ** 2)

        total_l2sp += l2sp_loss.item()
        loss = loss + l2sp_loss

        # ======= 反向传播 =======
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（根据阶段调整阈值）
        current_phase = get_current_phase(epoch, config["phase_epochs"])
        grad_clip = 5.0 if current_phase < 2 else 1.0  # 后期使用更严格的裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        # ======= 累积统计 =======
        n = len(mol_batch)
        total_graphs += n

        total_loss += loss.item() * n
        total_tree_acc += tacc * n
        total_graph_acc += gacc * n
        total_stereo_acc += sacc * n
        total_kl += kl * n
        total_word_loss += word_loss.item() * n
        total_topo_loss += topo_loss.item() * n

        # 更新进度条
        pbar.set_postfix({
            'loss': loss.item(),
            'kl': kl,
            'l2sp': l2sp_loss.item()
        })

    # 计算平均L2-SP损失
    avg_l2sp = total_l2sp / batch_count if batch_count > 0 else 0.0

    return (
        total_loss / total_graphs,
        total_tree_acc / total_graphs,
        total_graph_acc / total_graphs,
        total_stereo_acc / total_graphs,
        total_kl / total_graphs,
        total_word_loss / total_graphs,
        total_topo_loss / total_graphs,
        avg_l2sp  # 平均L2-SP损失
    )


def valid_epoch(model, loader, device, epoch):
    """验证一个epoch"""
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

            # --------- 前向传播 ---------
            loss, kl, tacc, gacc, sacc, word_loss, topo_loss = model(
                batch_input,
                beta=config['beta'],
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


def prepare_data(moltrees, cond_matrix, config):
    """准备数据集，支持不同的分割方式"""
    split_method = config.get("split_method", "by_sample")

    print(f"\nPreparing data with method: {split_method}")
    print(f"Total samples: {len(moltrees)}")

    if split_method == "by_sample":
        # 按样本分割
        all_samples = list(zip(moltrees, cond_matrix))

        # 计算分割比例
        test_ratio = config.get("test_ratio", 0.10)
        val_ratio = config.get("val_ratio", 0.20)

        # 先分割测试集
        train_val_data, test_data = train_test_split(
            all_samples,
            test_size=test_ratio,
            random_state=42,
            shuffle=True
        )

        # 再分割验证集（从剩余数据中）
        remaining_ratio = 1 - test_ratio
        val_ratio_adjusted = val_ratio / remaining_ratio

        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_ratio_adjusted,
            random_state=42,
            shuffle=True
        )

    elif split_method == "by_molecule":
        # 按分子分割（原始方式）
        grouped = {}
        for mol, cond_vec in zip(moltrees, cond_matrix):
            smi = mol.smiles
            if smi not in grouped:
                grouped[smi] = []
            grouped[smi].append((mol, cond_vec))

        all_smiles = np.array(list(grouped.keys()))

        # 计算分割比例
        test_ratio = config.get("test_ratio", 0.10)
        val_ratio = config.get("val_ratio", 0.20)

        # 先分割测试集
        train_val_smiles, test_smiles = train_test_split(
            all_smiles,
            test_size=test_ratio,
            random_state=42
        )

        # 再分割验证集
        remaining_ratio = 1 - test_ratio
        val_ratio_adjusted = val_ratio / remaining_ratio

        train_smiles, val_smiles = train_test_split(
            train_val_smiles,
            test_size=val_ratio_adjusted,
            random_state=42
        )

        def collect_samples(smiles_set):
            samples = []
            for smi in smiles_set:
                for mol, cond_vec in grouped[smi]:
                    samples.append((mol, cond_vec))
            return samples

        train_data = collect_samples(set(train_smiles))
        val_data = collect_samples(set(val_smiles))
        test_data = collect_samples(set(test_smiles))

    else:
        raise ValueError(f"Unknown split method: {split_method}")

    # 统计信息
    print(f"\nData Split ({split_method}):")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # 检查分子重叠情况
    def get_smiles_set(data):
        return set(mol.smiles for mol, _ in data)

    train_smiles = get_smiles_set(train_data)
    val_smiles = get_smiles_set(val_data)
    test_smiles = get_smiles_set(test_data)

    print(f"\nMolecular Overlap Analysis:")
    print(f"  Unique in Train: {len(train_smiles)}")
    print(f"  Unique in Val:   {len(val_smiles)}")
    print(f"  Unique in Test:  {len(test_smiles)}")
    print(f"  Train-Val overlap: {len(train_smiles & val_smiles)} molecules")
    print(f"  Train-Test overlap: {len(train_smiles & test_smiles)} molecules")
    print(f"  Val-Test overlap: {len(val_smiles & test_smiles)} molecules")

    return train_data, val_data, test_data


def check_condition_distribution(data, name):
    """检查溶剂条件的分布"""
    if len(data) == 0:
        return

    conds = np.array([cond_vec for _, cond_vec in data])
    print(f"\n{name} Condition Statistics:")
    print(f"  Shape: {conds.shape}")

    if conds.shape[1] > 0:
        for i in range(min(3, conds.shape[1])):  # 只检查前3个特征
            col = conds[:, i]
            print(f"  Feature {i}: min={col.min():.3f}, max={col.max():.3f}, "
                  f"mean={col.mean():.3f}, std={col.std():.3f}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    print("Loading moltrees and vocab...")
    with open(config["mol_pkl"], "rb") as f:
        moltrees = pickle.load(f)
    vocab = Vocab([x.strip() for x in open(config["vocab_path"])])
    print(f"Loaded moltrees: {len(moltrees)}, vocab size: {len(vocab.vocab)}")

    df = pd.read_excel(config["processed_data"])
    if "Ligand_SMILES" not in df.columns:
        raise RuntimeError("processed_data must contain 'Ligand_SMILES' column")
    feat_cols = [c for c in df.columns if c.startswith("solv_pca_")]
    if len(feat_cols) == 0:
        raise RuntimeError("No feat_* columns found in processed_data")
    cond_matrix = df[feat_cols].to_numpy(dtype=np.float32)
    print(f"Loaded processed_data rows: {len(df)}, cond_dim: {cond_matrix.shape[1]}")

    if len(moltrees) != len(cond_matrix):
        raise RuntimeError(f"Length mismatch: moltrees={len(moltrees)} vs cond rows={len(cond_matrix)}")

    # 准备数据
    train_data, val_data, test_data = prepare_data(moltrees, cond_matrix, config)

    # 检查条件分布
    check_condition_distribution(train_data, "Train")
    check_condition_distribution(val_data, "Validation")
    check_condition_distribution(test_data, "Test")

    # 创建数据加载器
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

    # 初始化模型
    cond_dim = cond_matrix.shape[1]
    model = JTNNVAE(vocab,
                    config['hidden_size'],
                    config['latent_size'],
                    config['depthT'],
                    config['depthG'],
                    cond_dim=cond_dim).to(device)

    # 加载预训练权重
    if config.get("pretrained_ckpt", None) is not None and os.path.exists(config["pretrained_ckpt"]):
        print(f"Loading pretrained checkpoint from {config['pretrained_ckpt']}")
        ckpt = torch.load(config["pretrained_ckpt"], map_location=device)
        missing = model.load_state_dict(ckpt, strict=False)

        print("模型加载结果:")
        if len(missing[0]) == 0 and len(missing[1]) == 0:
            print("✅ 没有缺失的键")
        else:
            print("Missing keys:", missing[0])
            print("Unexpected keys:", missing[1])
    else:
        print("Warning: pretrained_ckpt not found. Training from scratch.")

    # 保存预训练参数作为L2-SP的起点
    theta_pretrained = {}
    for name, p in model.named_parameters():
        theta_pretrained[name] = p.detach().clone()

    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['initial_lr'],
                                  weight_decay=config['weight_decay'])

    # 学习率调度器
    schedulers = []
    if config['use_cosine']:
        schedulers.append(CosineAnnealingLR(optimizer, T_max=config['cosine_T_max']))
    if config['use_plateau']:
        schedulers.append(ReduceLROnPlateau(optimizer, mode='min',
                                            patience=config['plateau_patience'],
                                            factor=config['plateau_factor']))

    # 训练历史记录
    history = {
        'epoch': [],
        'phase': [],
        'tr_loss': [], 'tr_tacc': [], 'tr_gacc': [], 'tr_sacc': [],
        'tr_kl': [], 'tr_l2sp': [],
        'val_loss': [], 'val_tacc': [], 'val_gacc': [], 'val_sacc': [],
        'val_kl': [], 'lr': [],
    }

    best_val_loss = float('inf')
    no_improve = 0

    print("\n" + "=" * 60)
    print("开始动态渐进式微调训练")
    print("=" * 60)

    for epoch in range(1, config['epochs'] + 1):
        start = time.time()

        # ===== 动态策略应用 =====
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'=' * 40}")

        # 1. 应用冻结策略
        apply_freeze_strategy(model, epoch, config, verbose=True)

        # 2. 调整学习率
        lr_adjusted = adjust_learning_rate(optimizer, epoch, config)

        # 3. 获取当前阶段的L2-SP权重
        current_l2sp = get_current_l2sp_lambdas(epoch, config)
        print(f"Current L2-SP factors: {current_l2sp}")

        # ===== 训练与验证 =====
        tr_results = train_epoch(model, train_loader, optimizer, device, epoch, theta_pretrained)
        tr_loss, tr_tacc, tr_gacc, tr_sacc, tr_kl, tr_g_loss, tr_t_loss, tr_l2sp = tr_results

        val_results = valid_epoch(model, val_loader, device, epoch)
        val_loss, val_tacc, val_gacc, val_sacc, val_kl, val_g_loss, val_t_loss = val_results

        lr = optimizer.param_groups[0]['lr']
        end = time.time()

        # ===== 记录历史 =====
        current_phase = get_current_phase(epoch, config["phase_epochs"])
        history['epoch'].append(epoch)
        history['phase'].append(current_phase + 1)
        history['tr_loss'].append(tr_loss)
        history['tr_tacc'].append(tr_tacc)
        history['tr_gacc'].append(tr_gacc)
        history['tr_sacc'].append(tr_sacc)
        history['tr_kl'].append(tr_kl)
        history['tr_l2sp'].append(tr_l2sp)
        history['val_loss'].append(val_loss)
        history['val_tacc'].append(val_tacc)
        history['val_gacc'].append(val_gacc)
        history['val_sacc'].append(val_sacc)
        history['val_kl'].append(val_kl)
        history['lr'].append(lr)

        # ===== 学习率调度 =====
        for sch in schedulers:
            if isinstance(sch, ReduceLROnPlateau):
                sch.step(val_loss)
            else:
                sch.step()

        # ===== 输出信息 =====
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Time: {end - start:.1f}s | Phase: {current_phase + 1}")
        print(f"  Train Loss: {tr_loss:.4f} | KL: {tr_kl:.4f} | L2-SP: {tr_l2sp:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | KL: {val_kl:.4f}")
        print(f"  Train Acc: Tree={tr_tacc:.4f}, Graph={tr_gacc:.4f}")
        print(f"  Val   Acc: Tree={val_tacc:.4f}, Graph={val_gacc:.4f}")
        print(f"  Learning Rate: {lr:.2e}")

        # ===== 保存最佳模型 =====
        if val_loss + config['min_delta'] < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), config['model_save'])
            print(f"  ✅ Saved improved model to {config['model_save']}")
        else:
            no_improve += 1
            print(f"  ⏳ No improvement for {no_improve} epoch(s)")

            # 检查是否触发阶段内早停
            if no_improve >= config['early_stop_patience']:
                # 如果不在最后一个阶段，可以考虑提前进入下一阶段
                if current_phase < len(config["phase_epochs"]) - 1:
                    print(f"  ⚡ Early phase transition triggered")
                    # 跳过当前阶段的剩余epoch
                    next_phase_start = config["phase_epochs"][current_phase] + 1
                    if epoch < next_phase_start:
                        print(f"  ⚡ Jumping to next phase at epoch {next_phase_start}")
                        # 这里可以调整epoch计数器，但更简单的是继续训练
                        # 让自然的过程进入下一阶段
                else:
                    print(f"  🛑 Early stopping triggered at epoch {epoch}")
                    break

        # TensorBoard记录
        writer.add_scalar('Loss/train', tr_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('KL/train', tr_kl, epoch)
        writer.add_scalar('KL/val', val_kl, epoch)
        writer.add_scalar('L2-SP/train', tr_l2sp, epoch)
        writer.add_scalar('Accuracy/tree_train', tr_tacc, epoch)
        writer.add_scalar('Accuracy/tree_val', val_tacc, epoch)
        writer.add_scalar('LR', lr, epoch)
        writer.add_scalar('Phase', current_phase + 1, epoch)

        print(f"{'=' * 40}")

    # ===== 训练结束 =====
    writer.close()

    # 保存训练历史
    df_history = DataFrame(history)
    df_history.to_excel(config['output_path'], index=False)
    print(f"\nTraining history saved to {config['output_path']}")

    # 绘制训练曲线
    plot_training_curves(history, save_path=config["plot_path"])

    # 最终模型评估
    print("\n" + "=" * 60)
    print("训练完成！最终评估：")
    print("=" * 60)

    # 在测试集上评估
    test_results = valid_epoch(model, test_loader, device, epoch)
    test_loss, test_tacc, test_gacc, test_sacc, test_kl, _, _ = test_results

    print(f"测试集结果:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Tree Accuracy: {test_tacc:.4f}")
    print(f"  Graph Accuracy: {test_gacc:.4f}")
    print(f"  Stereo Accuracy: {test_sacc:.4f}")
    print(f"  KL Divergence: {test_kl:.4f}")

    # 保存最终模型
    final_model_path = config['model_save'].replace('.pth', '_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n最终模型保存到: {final_model_path}")

    print("\n动态渐进式微调训练完成！")