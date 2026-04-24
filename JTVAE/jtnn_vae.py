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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class TimeoutException(Exception):
    """Exception raised when molecular decoding exceeds the time limit."""
    pass


class Timeout:
    """Context manager for setting a time limit on molecular decoding operations."""

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
    """
    Junction Tree Variational Autoencoder (JT-VAE).

    This model encodes molecules into a continuous latent space using a dual
    representation: a tree encoder for substructure-level topology and a graph
    encoder for atom-level connectivity. Decoding proceeds autoregressively
    from the latent vectors to reconstruct valid molecular structures.

    Reference: Jin et al., ICML 2018.
    """

    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG, cond_dim=0):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        # latent_size is split equally between tree and molecular representations
        self.latent_size = latent_size = int(latent_size // 2)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Tree encoder: captures substructure composition and connectivity
        self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size)).to(device)
        # Tree decoder: autoregressive generation from latent vector
        self.decoder = JTNNDecoder(vocab, hidden_size, latent_size, cond_dim=cond_dim).to(device)

        # Message-passing networks for tree and graph representations
        self.jtmpn = JTMPN(hidden_size, depthG, dropout=0.3).to(device)
        self.mpn = MPN(hidden_size, depthG, dropout=0.3).to(device)

        # Bilinear layer for assembly score prediction
        self.A_assm = nn.Linear(latent_size, hidden_size, bias=False).to(device)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        # Mean and log-variance layers for the latent distributions
        self.T_mean = nn.Linear(hidden_size, latent_size).to(device)
        self.T_var = nn.Linear(hidden_size, latent_size).to(device)
        self.G_mean = nn.Linear(hidden_size, latent_size).to(device)
        self.G_var = nn.Linear(hidden_size, latent_size).to(device)

    def encode(self, jtenc_holder, mpn_holder):
        """Encode molecules into tree and graph feature vectors."""
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        tree_vecs = self.dropout(tree_vecs)
        mol_vecs = self.mpn(*mpn_holder)
        mol_vecs = self.dropout(mol_vecs)
        return tree_vecs, tree_mess, mol_vecs

    def encode_from_smiles(self, smiles_list):
        """Encode a list of SMILES strings directly into latent vectors."""
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        return torch.cat([tree_vecs, mol_vecs], dim=-1)

    def encode_latent(self, jtenc_holder, mpn_holder):
        """
        Encode molecules into the parameters of the latent Gaussian distributions.

        Returns:
            means: Concatenated mean vectors [tree_mean | mol_mean].
            log_vars: Concatenated log-variance vectors [tree_log_var | mol_log_var].
        """
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))
        mol_var = -torch.abs(self.G_var(mol_vecs))
        return torch.cat([tree_mean, mol_mean], dim=1), torch.cat([tree_var, mol_var], dim=1)

    def get_sampled_latent_vector(self, smiles_list):
        """
        Sample latent vectors via the reparameterization trick for given SMILES.

        Returns:
            z_tree: Tree latent vector [batch_size, latent_size]
            z_mol: Molecular latent vector [batch_size, latent_size]
            z_combined: Concatenated latent vector [batch_size, 2 * latent_size]
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        # Step 1: Encode molecules into feature vectors
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)

        # Step 2: Compute distribution parameters
        tree_mean = self.T_mean(tree_vecs)
        tree_log_var = -torch.abs(self.T_var(tree_vecs))
        mol_mean = self.G_mean(mol_vecs)
        mol_log_var = -torch.abs(self.G_var(mol_vecs))

        # Step 3: Reparameterization trick
        epsilon_tree = torch.randn_like(tree_mean)
        epsilon_mol = torch.randn_like(mol_mean)

        z_tree = tree_mean + torch.exp(tree_log_var / 2) * epsilon_tree
        z_mol = mol_mean + torch.exp(mol_log_var / 2) * epsilon_mol

        # Step 4: Concatenate tree and molecular latent vectors
        z_combined = torch.cat([z_tree, z_mol], dim=1)

        return z_tree, z_mol, z_combined

    @torch.no_grad()
    def decode_from_latent(self, z, prob_decode=False, timeout=10):
        """
        Decode latent vectors back into SMILES strings.

        Args:
            z: Latent vector of shape [B, 2 * latent_size].
            prob_decode: Whether to use probabilistic sampling during decoding.
            timeout: Maximum decoding time (seconds) per molecule.

        Returns:
            A list of SMILES strings (None for failed decodings).
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
        """
        Reparameterization trick: sample latent vectors and compute KL divergence.

        Args:
            z_vecs: Input feature vectors.
            W_mean: Linear layer for mean prediction.
            W_var: Linear layer for log-variance prediction.

        Returns:
            z_vecs: Sampled latent vectors.
            kl_loss: KL divergence loss for the batch.
        """
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))  # Enforce negative log-variance (see Mueller et al.)
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_var(torch.randn_like(z_mean), device=device)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        """Generate a single molecule by sampling from the prior distribution."""
        z_tree = torch.randn(1, self.latent_size).to(device)
        z_mol = torch.randn(1, self.latent_size).to(device)
        return self.decode(z_tree, z_mol, prob_decode)

    def forward(self, x_batch, beta, cond_batch=None):
        """
        Forward pass of the JT-VAE.

        Args:
            x_batch: Preprocessed batch of molecules (tree and graph tensors).
            beta: Weight for the KL divergence term.
            cond_batch: Optional condition vectors for conditional generation.

        Returns:
            total_loss, kl_div, word_acc, topo_acc, assm_acc, word_loss, topo_loss
        """
        # Step 1: Unpack batch
        if len(x_batch) == 4:
            x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        elif len(x_batch) == 3:
            x_batch, x_jtenc_holder, x_mpn_holder = x_batch
            x_jtmpn_holder = None
        else:
            raise ValueError(
                f"Invalid batch size: expected 3 or 4 elements, got {len(x_batch)}"
            )

        # Step 2: Encode molecules into feature vectors
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(
            x_jtenc_holder,
            x_mpn_holder
        )

        # Step 3: Reparameterization
        z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

        kl_div = tree_kl + mol_kl

        # Step 4: Decode tree structure (condition on solvent if provided)
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(
            x_batch,
            z_tree_vecs,
            cond_vec=cond_batch
        )

        # Step 5: Compute assembly loss
        if x_jtmpn_holder is not None:
            assm_loss, assm_acc = self.assm(
                x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess
            )
        else:
            assm_loss = torch.tensor(0.0, device=z_mol_vecs.device)
            assm_acc = 0.0

        # Step 6: Compute total loss
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
        """
        Compute the assembly loss for predicting correct subgraph connections.

        This module predicts which candidate substructure should be attached
        at each expansion node based on the molecular latent vector and tree messages.
        """
        jtmpn_holder, batch_idx = jtmpn_holder
        fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
        batch_idx = create_var(batch_idx, device=device)

        # Compute candidate embedding vectors
        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)

        # Bilinear scoring between molecular vector and candidate vectors
        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs)
        scores = torch.bmm(
            x_mol_vecs.unsqueeze(1),
            cand_vecs.unsqueeze(-1)
        ).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch):
            # Only consider nodes with multiple candidates that are not leaves
            comp_nodes = [node for node in mol_tree.nodes if len(node.cands) > 1 and not node.is_leaf]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                # Validate that label_mol exists
                try:
                    if not hasattr(node, 'label_mol') or node.label_mol is None:
                        print(f"Skipping invalid node: label_mol not found, num_candidates={len(node.cands)}")
                        cnt -= 1
                        continue

                    # Canonicalize SMILES for robust label matching
                    canonical_label = Chem.MolToSmiles(node.label_mol, kekuleSmiles=True).replace(':', '')
                    canonical_cands = [Chem.MolToSmiles(Chem.MolFromSmiles(c), kekuleSmiles=True).replace(':', '')
                                       for c in node.cands]
                    label = canonical_cands.index(canonical_label)

                except (ValueError, AttributeError) as e:
                    print(f"Skipping invalid node: label={canonical_label}, num_candidates={len(node.cands)}")
                    cnt -= 1
                    continue

                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]), device=device)
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        # Handle edge case where all nodes were skipped
        if cnt == 0:
            return create_var(torch.zeros(1), device=device), 0.0

        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        """
        Decode a single molecule from its tree and molecular latent vectors.

        This method first generates the junction tree structure, then assembles
        the full molecular graph by attaching substructures according to
        predicted attachment scores.
        """
        # Currently supports only batch size 1
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        # Generate the junction tree
        pred_root, pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0:
            return None
        elif len(pred_nodes) == 1:
            return pred_root.smiles

        # Assign node IDs and mark leaf status
        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        # Encode the generated tree for assembly
        scope = [(0, len(pred_nodes))]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _, tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict)

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze()

        # Initialize the molecular graph from the root node
        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        # Depth-first assembly with aromaticity check
        cur_mol, _ = self.dfs_assemble(
            tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [],
            pred_root, None, prob_decode, check_aroma=True
        )

        # Fallback: retry without aromaticity check if the first attempt failed
        if cur_mol is None:
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol, pre_mol = self.dfs_assemble(
                tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [],
                pred_root, None, prob_decode, check_aroma=False
            )
            if cur_mol is None:
                cur_mol = pre_mol

        if cur_mol is None:
            return None

        # Canonicalize the assembled molecule
        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None

    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node,
                     prob_decode, check_aroma, depth=0, max_depth=50):
        """
        Depth-first assembly of the molecular graph from the junction tree.

        This method recursively attaches child substructures to the current
        molecular graph, selecting the best attachment configuration based on
        learned scoring.

        Args:
            depth: Current recursion depth (for overflow protection).
            max_depth: Maximum allowed recursion depth.

        Returns:
            (cur_mol, pre_mol): The assembled molecule and a fallback molecule.
        """
        if depth > max_depth:
            print(f"Warning: recursion depth limit exceeded (depth={depth})")
            return None, cur_mol

        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        # Separate children into regular nodes and singletons (single-atom fragments)
        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        # Enumerate possible attachment configurations
        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands, aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles, cand_amap = zip(*cands)
        aroma_score = torch.Tensor(aroma_score).to(device)
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        # Score candidate configurations
        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
            cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        # Select candidates (probabilistic or greedy)
        if prob_decode:
            probs = F.softmax(scores.view(1, -1), dim=1).squeeze() + 1e-7
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            # Update atom mappings for the newly attached substructure
            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            # Attach child substructures
            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            # Recursively assemble child substructures
            has_error = False
            for nei_node in children:
                if nei_node.is_leaf:
                    continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(
                    y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap,
                    pred_amap, nei_node, cur_node, prob_decode, check_aroma,
                    depth + 1, max_depth
                )
                if tmp_mol is None:
                    has_error = True
                    if i == 0:
                        pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error:
                return cur_mol, cur_mol

        return None, pre_mol
