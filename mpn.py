import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from nnutils import *
from chemutils import get_mol

# ======================== 原子/化学键特征编码 ========================
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
             'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']  # 确保最后一个元素是 'unknown'

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1  # 原子特征总维度
BOND_FDIM = 5 + 6  # 化学键特征总维度
MAX_NB = 6  # 最大邻接化学键数

def onek_encoding_unk(x, allowable_set):
    """将值 x 转换为 one-hot 编码，如果 x 不在允许集合中则编码为 'unknown'"""
    if x not in allowable_set:
        x = allowable_set[-1]  # 最后一个元素是 'unknown'
    # 返回列表而非 map 对象
    return [int(s == x) for s in allowable_set]  # 修复点：使用列表推导式替代 map

def atom_features(atom):
    """提取原子特征"""
    features = (
        onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) +
        onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +
        onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3]) +
        [int(atom.GetIsAromatic())]  # 修复点：所有拼接项都是列表
    )
    return torch.Tensor(features)

def bond_features(bond):
    """提取化学键特征"""
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    # 化学键基础特征
    fbond = [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.IsInRing())
    ]
    # 立体化学特征
    fstereo = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])  # 返回列表
    return torch.Tensor(fbond + fstereo)

# ======================== MPN 模型定义 ========================
class MPN(nn.Module):
    def __init__(self, hidden_size, depth, dropout=0.0):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = nn.Dropout(p=dropout)

        # 网络层定义
        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

        # 只添加一个BN层在输出前
        self.bn_o = nn.BatchNorm1d(hidden_size)

    def forward(self, fatoms, fbonds, agraph, bgraph, scope):
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)

        # 初始边的消息
        binput = self.W_i(fbonds)
        message = F.relu(binput)

        # 多轮消息传递
        for _ in range(self.depth - 1):
            nei_message = index_select_ND(message, 0, bgraph).sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = F.relu(binput + nei_message)

        # 聚合到原子
        nei_message = index_select_ND(message, 0, agraph).sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = self.W_o(ainput)
        atom_hiddens = self.bn_o(atom_hiddens)  # BN before activation
        atom_hiddens = F.relu(atom_hiddens)
        atom_hiddens = self.dropout(atom_hiddens)

        # 平均池化获得分子级向量
        batch_vecs = [
            atom_hiddens[st: st + le].mean(dim=0)
            for st, le in scope
        ]
        return torch.stack(batch_vecs, dim=0)


    @staticmethod
    def tensorize(mol_batch):
        """将分子批量转换为 Tensor 格式"""
        padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
        fatoms, fbonds = [], [padding]  # 化学键从 1 开始索引
        in_bonds, all_bonds = [], [(-1, -1)]  # 化学键从 1 开始索引
        scope = []
        total_atoms = 0

        for smiles in mol_batch:
            mol = get_mol(smiles)
            if mol is None:
                continue  # 跳过无效分子

            n_atoms = mol.GetNumAtoms()
            # 提取原子特征
            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))
                in_bonds.append([])

            # 提取化学键特征
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom().GetIdx() + total_atoms
                a2 = bond.GetEndAtom().GetIdx() + total_atoms

                # 双向化学键（正向）
                b = len(all_bonds)
                all_bonds.append((a1, a2))
                fbonds.append(torch.cat([fatoms[a1], bond_features(bond)], 0))
                in_bonds[a2].append(b)

                # 双向化学键（反向）
                b = len(all_bonds)
                all_bonds.append((a2, a1))
                fbonds.append(torch.cat([fatoms[a2], bond_features(bond)], 0))
                in_bonds[a1].append(b)

            scope.append((total_atoms, n_atoms))
            total_atoms += n_atoms

        # 构建邻接矩阵
        total_bonds = len(all_bonds)
        fatoms = torch.stack(fatoms, 0) if fatoms else torch.empty(0)
        fbonds = torch.stack(fbonds, 0) if fbonds else torch.empty(0)
        agraph = torch.zeros(total_atoms, MAX_NB).long()
        bgraph = torch.zeros(total_bonds, MAX_NB).long()

        # 填充邻接索引
        for a in range(total_atoms):
            for i, b in enumerate(in_bonds[a][:MAX_NB]):
                agraph[a, i] = b

        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x][:MAX_NB]):
                if all_bonds[b2][0] != y:
                    bgraph[b1, i] = b2

        return (fatoms, fbonds, agraph, bgraph, scope)