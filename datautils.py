import torch
from torch.utils.data import Dataset, DataLoader
from mol_tree import MolTree
import numpy as np
from jtnn_enc import JTNNEncoder
from mpn import MPN
from jtmpn import JTMPN
import pickle
import os, random
from nnutils import create_var


def simple_collate(batch):
    return batch[0]


class MolTreeFolder(object):
    def __init__(self, data, vocab, batch_size, num_workers=4, shuffle=True, assm=True):

        # data: list of (moltree, cond_vec)
        self.data = data
        self.vocab = vocab
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

    def __iter__(self):
        data = self.data.copy()
        if self.shuffle:
            random.shuffle(data)

        mol_batches = []
        cond_batches = []

        # batch 划分
        for i in range(0, len(data), self.batch_size):
            chunk = data[i:i+self.batch_size]
            if len(chunk) < self.batch_size:
                break  # 保持和原 JTVAE 一致

            mols = [x[0] for x in chunk]
            conds = [x[1] for x in chunk]

            mol_batches.append(mols)
            cond_batches.append(conds)

        # 创建 Dataset
        dataset = MolTreeDataset(mol_batches, cond_batches, self.vocab, self.assm)

        # 调用 DataLoader（与原版同步）
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=simple_collate
        )

        # DataLoader 的输出已经包含 cond_batch，无需再重新查 cond_dict
        for mol_batch, cond_batch, jtenc_holder, mpn_holder, jtmpn_holder in dataloader:
            yield mol_batch, torch.tensor(cond_batch, dtype=torch.float32), jtenc_holder, mpn_holder, jtmpn_holder



class MolTreeDataset(Dataset):
    def __init__(self, mol_batches, cond_batches, vocab, assm=True):
        """
        mol_batches: list of moltree lists, each is a batch
        cond_batches: list of cond lists, aligned with mol_batches
        """
        self.mol_batches = mol_batches
        self.cond_batches = cond_batches
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.mol_batches)

    def __getitem__(self, idx):
        mol_batch = self.mol_batches[idx]  # list of MolTrees
        cond_batch = self.cond_batches[idx]  # list of cond vectors

        jtenc_holder, mpn_holder, jtmpn_holder = None, None, None

        # 原版 tensorize，是对 mol_batch 做批处理的
        res = tensorize(mol_batch, self.vocab, assm=self.assm)
        if res is None:
            return None

        mol_batch, jtenc_holder, mpn_holder = res
        return mol_batch, cond_batch, jtenc_holder, mpn_holder, jtmpn_holder


def tensorize(tree_batch, vocab, assm=True, device=None):
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]

    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    mpn_holder = MPN.tensorize(smiles_batch)

    # --------- no-assm 模式不改 ---------
    if not assm:
        jtenc_holder = tuple([create_var(x, device=device) for x in jtenc_holder])
        mpn_holder = tuple([create_var(x, device=device) for x in mpn_holder])
        return tree_batch, jtenc_holder, mpn_holder

    # ========== 下面是仅对 assm=True 的修复 ==========
    cands = []
    batch_idx = []

    for i, mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:

            if node.is_leaf:
                continue
            if not hasattr(node, 'cands') or not hasattr(node, 'label_mol'):
                continue
            if len(node.cands) <= 1:
                continue

            # 过滤掉没有 bond 的 candidate（避免 fbonds 维度为 0）
            valid_cands = []
            for cand in node.cands:
                mol = cand.mol
                if mol is None or mol.GetNumAtoms() == 0:
                    continue
                if mol.GetNumBonds() == 0:     # <- 关键修复
                    continue
                valid_cands.append(cand)

            # 过滤后为空，则跳过该节点
            if len(valid_cands) == 0:
                continue

            # 添加 candidate
            cands.extend([(cand, mol_tree.nodes, node) for cand in valid_cands])
            batch_idx.extend([i] * len(valid_cands))

    # -------- 如果所有候选为空，返回 None，训练函数跳过该 batch --------
    if len(cands) == 0:
        return None

    # ========== 正常 JTMPN tensorize ==========
    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)

    # JTMPN 仍可能返回空（极小概率）
    if jtmpn_holder is None or len(jtmpn_holder) == 0:
        return None

    # ========= 把所有内容转为 tensor =========
    jtenc_holder = tuple([create_var(x, device=device) for x in jtenc_holder])
    mpn_holder = tuple([create_var(x, device=device) for x in mpn_holder])
    jtmpn_holder = tuple([create_var(x, device=device) for x in jtmpn_holder])

    batch_idx = create_var(torch.LongTensor(batch_idx), device=device)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx)



def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
