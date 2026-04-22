import torch
import torch.nn as nn
import torch.nn.functional as F
from nnutils import create_var, index_select_ND
from chemutils import get_mol
import rdkit.Chem as Chem

# ======================== 原子/化学键特征编码 ========================
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
             'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1  # 原子特征总维度
BOND_FDIM = 5  # 化学键特征总维度
MAX_NB = 15    # 最大邻接化学键数

def onek_encoding_unk(x, allowable_set):
    """将值 x 转换为 one-hot 编码，如果 x 不在允许集合中则编码为 'unknown'"""
    if x not in allowable_set:
        x = allowable_set[-1]  # 最后一个元素是 'unknown'
    # 返回列表而非 map 对象
    return [int(s == x) for s in allowable_set]  # ✅ 关键修复：使用列表推导式替代 map

def atom_features(atom):
    """原子特征（修复后的拼接逻辑）"""
    features = (
        onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) +
        onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +
        [int(atom.GetIsAromatic())]  # ✅ 确保所有部分都是列表
    )
    return torch.Tensor(features)

def bond_features(bond):
    """化学键特征"""
    bt = bond.GetBondType()
    return torch.Tensor([
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.IsInRing())
    ])

# ======================== JTMPN 模型定义 ========================
import torch.nn as nn
import torch.nn.functional as F


# ======================== JTMPN 模型定义 ========================
# ======================== JTMPN 模型定义 ========================
class JTMPN(nn.Module):
    def __init__(self, hidden_size, depth, dropout=0):
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

        # 只添加一个BN层在输出前
        self.bn_o = nn.BatchNorm1d(hidden_size)

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, fatoms, fbonds, agraph, bgraph, scope, tree_message):
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)

        # 初始化消息
        binput = self.W_i(fbonds)
        graph_message = F.relu(binput)
        graph_message = self.dropout_layer(graph_message)

        for _ in range(self.depth - 1):
            message = torch.cat([tree_message, graph_message], dim=0)
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            nei_message = self.dropout_layer(nei_message)
            graph_message = F.relu(binput + nei_message)
            graph_message = self.dropout_layer(graph_message)

        message = torch.cat([tree_message, graph_message], dim=0)
        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = self.W_o(ainput)
        atom_hiddens = self.bn_o(atom_hiddens)  # BN before activation
        atom_hiddens = F.relu(atom_hiddens)
        atom_hiddens = self.dropout_layer(atom_hiddens)

        mol_vecs = []
        for st, le in scope:
            if le > 0:
                mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
                mol_vecs.append(mol_vec)
            else:
                mol_vecs.append(torch.zeros(self.hidden_size).to(atom_hiddens.device))

        return torch.stack(mol_vecs, dim=0)
    @staticmethod
    def tensorize(cand_batch, mess_dict):
        fatoms, fbonds = [], []
        in_bonds, all_bonds = [], []
        total_atoms = 0
        total_mess = len(mess_dict) + 1  # must include vec(0) padding
        scope = []

        for smiles, all_nodes, ctr_node in cand_batch:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue  # 跳过无效分子
            Chem.Kekulize(mol)
            n_atoms = mol.GetNumAtoms()

            # 提取原子特征
            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))
                in_bonds.append([])

            # 处理化学键
            for bond in mol.GetBonds():
                # 获取 Atom 对象
                a1_obj = bond.GetBeginAtom()
                a2_obj = bond.GetEndAtom()

                # 计算批量偏移后的索引
                a1_idx = a1_obj.GetIdx() + total_atoms
                a2_idx = a2_obj.GetIdx() + total_atoms

                # 获取原子映射编号
                x_nid = a1_obj.GetAtomMapNum()
                y_nid = a2_obj.GetAtomMapNum()

                # 处理正向和反向化学键
                for x, y in [(a1_idx, a2_idx), (a2_idx, a1_idx)]:
                    b = total_mess + len(all_bonds)
                    all_bonds.append((x, y))
                    fbonds.append(torch.cat([fatoms[x], bond_features(bond)], 0))
                    in_bonds[y].append(b)

                # 处理消息传递索引
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1
                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid, y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid, y_bid)]
                        in_bonds[a2_idx].append(mess_idx)
                    if (y_bid, x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid, x_bid)]
                        in_bonds[a1_idx].append(mess_idx)

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

        for b1 in range(total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x][:MAX_NB]):
                if b2 < total_mess or all_bonds[b2 - total_mess][0] != y:
                    bgraph[b1, i] = b2

        return (fatoms, fbonds, agraph, bgraph, scope)