import torch
import torch.nn as nn
import torch.nn.functional as F
from mol_tree import Vocab, MolTree
from nnutils import create_var, flatten_tensor, avg_pool
from jtnn_enc import JTNNEncoder
from jtnn_dec import JTNNDecoder
from mpn import MPN
from jtmpn import JTMPN
from datautils import tensorize

from chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols
import rdkit
import rdkit.Chem as Chem
import copy, math
import signal
device = torch.device("cuda:1" if torch.cuda.is_available() else "")


class TimeoutException(Exception):
    pass


class Timeout:
    def __init__(self, seconds):
        self.seconds = seconds

    def handle_timeout(self, signum, frame):
        raise TimeoutException()

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)


class JTNNVAE(nn.Module):
    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG, cond_dim=0):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size = int(latent_size // 2)  # Tree and Mol have two vectors

        # 添加Dropout层（增加dropout率）
        self.dropout = nn.Dropout(0.3)  # 从0.2增加到0.3

        self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size)).to(device)
        self.decoder = JTNNDecoder(vocab, hidden_size, latent_size, cond_dim=cond_dim).to(device)

        self.jtmpn = JTMPN(hidden_size, depthG, dropout=0.3).to(device)  # 增加dropout
        self.mpn = MPN(hidden_size, depthG, dropout=0.3).to(device)  # 增加dropout

        self.A_assm = nn.Linear(latent_size, hidden_size, bias=False).to(device)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.T_mean = nn.Linear(hidden_size, latent_size).to(device)
        self.T_var = nn.Linear(hidden_size, latent_size).to(device)
        self.G_mean = nn.Linear(hidden_size, latent_size).to(device)
        self.G_var = nn.Linear(hidden_size, latent_size).to(device)

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        tree_vecs = self.dropout(tree_vecs)  # 编码器输出后添加Dropout
        mol_vecs = self.mpn(*mpn_holder)
        mol_vecs = self.dropout(mol_vecs)  # 编码器输出后添加Dropout
        return tree_vecs, tree_mess, mol_vecs

    def encode_from_smiles(self, smiles_list):
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        return torch.cat([tree_vecs, mol_vecs], dim=-1)

    def encode_latent(self, jtenc_holder, mpn_holder):
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))
        mol_var = -torch.abs(self.G_var(mol_vecs))
        return torch.cat([tree_mean, mol_mean], dim=1), torch.cat([tree_var, mol_var], dim=1)

    def get_sampled_latent_vector(self, smiles_list):
        """
        获取重参数化采样后的潜在向量（用于解码）

        返回:
            z_tree: 树潜在向量 [batch_size, 64]
            z_mol: 分子潜在向量 [batch_size, 64]
            z_combined: 拼接潜在向量 [batch_size, 128]
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        # 1. 编码
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)

        # 2. 通过均值方差网络得到分布参数
        tree_mean = self.T_mean(tree_vecs)  # [batch_size, 64]
        tree_log_var = -torch.abs(self.T_var(tree_vecs))  # [batch_size, 64]
        mol_mean = self.G_mean(mol_vecs)  # [batch_size, 64]
        mol_log_var = -torch.abs(self.G_var(mol_vecs))  # [batch_size, 64]

        # 3. 重参数化采样
        epsilon_tree = torch.randn_like(tree_mean)
        epsilon_mol = torch.randn_like(mol_mean)

        z_tree = tree_mean + torch.exp(tree_log_var / 2) * epsilon_tree
        z_mol = mol_mean + torch.exp(mol_log_var / 2) * epsilon_mol

        # 4. 拼接
        z_combined = torch.cat([z_tree, z_mol], dim=1)  # [batch_size, 128]

        return z_tree, z_mol, z_combined

    @torch.no_grad()
    def decode_from_latent(self, z, prob_decode=False, timeout=10):
        """
        z: [B, 2 * latent_size]
        timeout: 单个分子最大解码时间（秒）
        """

        if z.dim() == 1:
            z = z.unsqueeze(0)

        z_tree, z_mol = torch.split(
            z,
            self.latent_size,
            dim=1
        )

        smiles = []
        for i in range(z.size(0)):
            try:
                with Timeout(timeout):
                    smi = self.decode(
                        z_tree[i:i + 1],
                        z_mol[i:i + 1],
                        prob_decode
                    )
            except TimeoutException:
                smi = None
            except Exception:
                smi = None
            smiles.append(smi)
        return smiles

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))  # Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_var(torch.randn_like(z_mean), device=device)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        z_tree = torch.randn(1, self.latent_size).to(device)
        z_mol = torch.randn(1, self.latent_size).to(device)
        return self.decode(z_tree, z_mol, prob_decode)

    def forward(self, x_batch, beta, cond_batch=None):
        # ====== 1. unpack batch ======
        if len(x_batch) == 4:
            x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        elif len(x_batch) == 3:
            x_batch, x_jtenc_holder, x_mpn_holder = x_batch
            x_jtmpn_holder = None
        else:
            raise ValueError(
                f"Invalid batch size: expected 3 or 4 elements, got {len(x_batch)}"
            )

        # ====== 2. Encoder（保持原版，无条件）======
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(
            x_jtenc_holder,
            x_mpn_holder
        )

        # ====== 3. reparameterization ======
        z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

        kl_div = tree_kl + mol_kl

        # ====== 4. Decoder（唯一使用 cond 的地方）======
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(
            x_batch,
            z_tree_vecs,
            cond_vec=cond_batch  # ⭐ 溶剂条件 ONLY HERE
        )

        # ====== 5. Assembly loss（保持原版）======
        if x_jtmpn_holder is not None:
            assm_loss, assm_acc = self.assm(
                x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess
            )
        else:
            assm_loss = torch.tensor(0.0, device=z_mol_vecs.device)
            assm_acc = 0.0

        # ====== 6. Total loss ======
        total_loss = word_loss + topo_loss + assm_loss + beta * kl_div

        return (
            total_loss,
            kl_div.item(),
            word_acc,
            topo_acc,
            assm_acc,
            word_loss,
            topo_loss
        )


    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        jtmpn_holder, batch_idx = jtmpn_holder
        fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
        batch_idx = create_var(batch_idx, device=device)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)

        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs)  # bilinear
        scores = torch.bmm(
            x_mol_vecs.unsqueeze(1),
            cand_vecs.unsqueeze(-1)
        ).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch):
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                # ========== 关键修复开始 ==========

                # 确保 label_mol 存在
                try:
                    # 检查 label_mol 是否存在
                    if not hasattr(node, 'label_mol') or node.label_mol is None:
                        print(f"跳过无效节点，未找到 label_mol，候选数: {len(node.cands)}")
                        cnt -= 1
                        continue

                    canonical_label = Chem.MolToSmiles(node.label_mol, kekuleSmiles=True).replace(':', '')
                    canonical_cands = [Chem.MolToSmiles(Chem.MolFromSmiles(c), kekuleSmiles=True).replace(':', '')
                                       for c in node.cands]
                    label = canonical_cands.index(canonical_label)

                except (ValueError, AttributeError) as e:
                    # 处理无效候选或无效分子
                    print(f"跳过无效节点，真实标签: {canonical_label}，候选数: {len(node.cands)}")
                    cnt -= 1
                    continue
                # ========== 关键修复结束 ==========

                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]), device=device)
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        # 处理所有节点都被跳过的情况
        if cnt == 0:
            return create_var(torch.zeros(1), device=device), 0.0

        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        # currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root, pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0:
            return None
        elif len(pred_nodes) == 1:
            return pred_root.smiles

        # Mark nid & is_leaf & atommap
        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _, tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict)  # Important: tree_mess is a matrix, mess_dict is a python dict

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze()  # bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol, _ = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None,
                                       prob_decode, check_aroma=True)
        if cur_mol is None:
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol, pre_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root,
                                                 None, prob_decode, check_aroma=False)
            if cur_mol is None: cur_mol = pre_mol

        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None

    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node,
                     prob_decode, check_aroma,  depth=0, max_depth=50):
        if depth > max_depth:
            print(f"警告：递归深度超限 ({depth})")
            return None, cur_mol

        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands, aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles, cand_amap = zip(*cands)
        aroma_score = torch.Tensor(aroma_score).to(device)
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
            cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        if prob_decode:
            probs = F.softmax(scores.view(1, -1), dim=1).squeeze() + 1e-7  # prevent prob = 0
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)  # father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue

            has_error = False
            for nei_node in children:
                if nei_node.is_leaf: continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(
                    y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap,
                    pred_amap, nei_node, cur_node, prob_decode, check_aroma,
                    depth + 1, max_depth  # 传递深度
                )
                if tmp_mol is None:
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error: return cur_mol, cur_mol

        return None, pre_mol
