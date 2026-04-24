import torch
import torch.nn as nn
import torch.nn.functional as F
from mol_tree import Vocab, MolTree, MolTreeNode
from nnutils import create_var, GRU
from chemutils import enum_assemble, set_atommap
import copy

MAX_NB = 15
MAX_DECODE_LEN = 100
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class JTNNDecoder(nn.Module):
    """
    Junction Tree Decoder.

    Generates a junction tree autoregressively from a latent vector.
    At each step, the decoder predicts either a new child node (word prediction)
    or a backtracking action (stop prediction) based on the current tree state
    and the global tree latent vector.
    """

    def __init__(self, vocab, hidden_size, latent_size, cond_dim=0):
        super(JTNNDecoder, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.cond_dim = cond_dim

        # Handle different vocab types for compatibility
        vocab_size = vocab.size() if hasattr(vocab, 'size') else (
            len(vocab.vocab) if hasattr(vocab, 'vocab') else len(vocab))

        # Node embedding and optional condition projection
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.cond_proj = nn.Linear(cond_dim, hidden_size * 2) if (cond_dim and cond_dim > 0) else None

        # GRU gates for message propagation between tree nodes
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        # Input dimension for prediction heads
        in_dim = hidden_size + latent_size

        # Word prediction head: predicts next child node type
        self.W = nn.Linear(in_dim, hidden_size)
        self.U = nn.Linear(in_dim, hidden_size)

        # Stop prediction head: predicts whether to backtrack
        self.U_i = nn.Linear(2 * hidden_size, hidden_size)

        # Output layers
        self.W_o = nn.Linear(hidden_size, vocab_size)  # Word logits
        self.U_o = nn.Linear(hidden_size, 1)            # Stop logit

        # Regularization
        self.dropout = nn.Dropout(0.3)
        self.layernorm = nn.LayerNorm(hidden_size)

        # Loss functions
        self.pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='sum')

        print("[JTNNDecoder] initialized:")
        print(f"  hidden_size={hidden_size}, latent_size={latent_size}, cond_dim={cond_dim}")
        print(f"  cond_proj: {self.cond_proj}")
        print(f"  aggregate input_dim (in_dim) = {in_dim}")
        print(f"  vocab_size = {vocab_size}")

        self._printed_cond_shape_once = False

    def aggregate(self, hiddens, contexts, x_tree_vecs, mode, cond_vec=None):
        """
        Aggregate hidden states with tree-level latent context and produce predictions.

        Args:
            hiddens: Hidden states of nodes being predicted.
            contexts: Indices mapping each node to its tree in the batch.
            x_tree_vecs: Tree-level latent vectors for each molecule in the batch.
            mode: 'word' for child prediction, 'stop' for backtracking prediction.
            cond_vec: Optional condition vectors (unused in current version).

        Returns:
            Prediction scores (vocab_size for word, 1 for stop).
        """
        if mode == 'word':
            V, V_o = self.W, self.W_o
        elif mode == 'stop':
            V, V_o = self.U, self.U_o
        else:
            raise ValueError("aggregate mode must be 'word' or 'stop'")

        # Retrieve tree-level context for each node
        tree_contexts = x_tree_vecs.index_select(0, contexts)

        # Concatenate node hidden state with tree context
        input_vec = torch.cat([hiddens, tree_contexts], dim=-1)

        # Validate input dimension
        expected = V.in_features
        if input_vec.size(1) != expected:
            raise RuntimeError(
                f"Dimension mismatch in aggregate(): "
                f"input_vec dim {input_vec.size(1)} != {expected}, "
                f"tree_contexts: {tree_contexts.size()}, hiddens: {hiddens.size()}"
            )

        # Apply prediction head
        output = V(input_vec)
        output = self.layernorm(output)
        output = F.relu(output)
        output = self.dropout(output)
        return V_o(output)

    def forward(self, mol_batch, x_tree_vecs, cond_vec=None):
        """
        Teacher-forcing forward pass for training.

        Traverses each tree in DFS order and collects prediction targets
        for both word (child node type) and stop (backtracking) decisions.

        Args:
            mol_batch: List of MolTree objects.
            x_tree_vecs: Tree latent vectors for each molecule.
            cond_vec: Optional condition vectors.

        Returns:
            pred_loss, stop_loss, pred_acc, stop_acc
        """
        pred_hiddens, pred_contexts, pred_targets = [], [], []
        stop_hiddens, stop_contexts, stop_targets = [], [], []
        traces = []

        # Build DFS traversal traces and clear neighbors for incremental construction
        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], -1)
            traces.append(s)
            for node in mol_tree.nodes:
                node.neighbors = []

        batch_size = len(mol_batch)

        # Initialize: predict the root node
        pred_hiddens.append(create_var(torch.zeros(batch_size, self.hidden_size)))
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
        pred_contexts.append(create_var(torch.LongTensor(range(batch_size))))

        padding = create_var(torch.zeros(self.hidden_size))
        h = {}

        max_iter = max([len(tr) for tr in traces])
        for t in range(max_iter):
            prop_list, batch_list = [], []
            for i, plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x = []
            cur_h_nei, cur_o_nei = [], []

            for node_x, real_y, _ in prop_list:
                # Neighbors excluding the parent (for GRU update)
                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
                cur_h_nei.extend(cur_nei + [padding] * (MAX_NB - len(cur_nei)))

                # All neighbors (for stop prediction)
                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
                cur_o_nei.extend(cur_nei + [padding] * (MAX_NB - len(cur_nei)))

                cur_x.append(node_x.wid)

            # Embed and reshape
            cur_x = self.embedding(create_var(torch.LongTensor(cur_x)))
            cur_h_nei = torch.stack([torch.stack(cur_h_nei[i * MAX_NB:(i + 1) * MAX_NB]) for i in range(len(cur_x))])
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            cur_o_nei = torch.stack([torch.stack(cur_o_nei[i * MAX_NB:(i + 1) * MAX_NB]) for i in range(len(cur_x))])
            cur_o = cur_o_nei.sum(dim=1)

            pred_target, pred_list, stop_target = [], [], []
            for i, m in enumerate(prop_list):
                node_x, node_y, direction = m
                x, y = node_x.idx, node_y.idx
                h[(x, y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    # Forward edge: predict child node type
                    pred_target.append(node_y.wid)
                    pred_list.append(i)
                stop_target.append(direction)

            cur_batch = create_var(torch.LongTensor(batch_list))

            # Collect stop prediction targets
            stop_hidden = torch.cat([cur_x, cur_o], dim=1)
            stop_hiddens.append(stop_hidden)
            stop_contexts.append(cur_batch)
            stop_targets.extend(stop_target)

            # Collect word prediction targets for forward edges
            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = create_var(torch.LongTensor(batch_list))
                pred_contexts.append(cur_batch)
                cur_pred = create_var(torch.LongTensor(pred_list))
                pred_hiddens.append(new_h.index_select(0, cur_pred))
                pred_targets.extend(pred_target)

        # Final stop prediction for root nodes (always backtrack to terminate)
        cur_x, cur_o_nei = [], []
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            cur_o_nei.extend(cur_nei + [padding] * (MAX_NB - len(cur_nei)))

        cur_x = self.embedding(create_var(torch.LongTensor(cur_x)))
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1, MAX_NB, self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)
        stop_hidden = torch.cat([cur_x, cur_o], dim=1)
        stop_hiddens.append(stop_hidden)
        stop_contexts.append(create_var(torch.LongTensor(range(batch_size))))
        stop_targets.extend([0] * len(mol_batch))

        # ---- Compute losses ----

        # Word prediction loss
        pred_contexts = torch.cat(pred_contexts, dim=0)
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_scores = self.aggregate(pred_hiddens, pred_contexts, x_tree_vecs, 'word', cond_vec)
        pred_targets = create_var(torch.LongTensor(pred_targets))
        pred_loss = self.pred_loss(pred_scores, pred_targets) / batch_size
        _, preds = torch.max(pred_scores, dim=1)
        pred_acc = (preds == pred_targets).float().mean()

        # Stop prediction loss
        stop_contexts = torch.cat(stop_contexts, dim=0)
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_hiddens = F.relu(self.U_i(stop_hiddens))
        stop_scores = self.aggregate(stop_hiddens, stop_contexts, x_tree_vecs, 'stop', cond_vec).squeeze(-1)
        stop_targets = create_var(torch.Tensor(stop_targets))
        stop_loss = self.stop_loss(stop_scores, stop_targets) / batch_size
        stop_acc = (torch.ge(stop_scores, 0).float() == stop_targets).float().mean()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()

    def decode(self, x_tree_vecs, prob_decode, cond_vec=None):
        """
        Autoregressive decoding of a single molecule from its tree latent vector.

        Uses depth-first traversal with backtracking. At each node, the model
        first decides whether to backtrack or expand. If expanding, it predicts
        the next child node type that is both chemically compatible and
        structurally assemblable.

        Args:
            x_tree_vecs: Tree latent vector [1, latent_size].
            prob_decode: Whether to sample from the predicted distribution.
            cond_vec: Optional condition vector.

        Returns:
            root: Root node of the generated junction tree.
            all_nodes: List of all generated tree nodes.
        """
        assert x_tree_vecs.size(0) == 1

        stack = []
        init_hiddens = create_var(torch.zeros(1, self.hidden_size))
        zero_pad = create_var(torch.zeros(1, 1, self.hidden_size))
        contexts = create_var(torch.LongTensor(1).zero_())

        # Predict the root node
        root_score = self.aggregate(init_hiddens, contexts, x_tree_vecs, 'word', cond_vec)
        _, root_wid = torch.max(root_score, dim=1)
        root_wid = root_wid.item()

        root = MolTreeNode(self.vocab.get_smiles(root_wid))
        root.wid = root_wid
        root.idx = 0
        stack.append((root, self.vocab.get_slots(root_wid)))
        all_nodes, h = [root], {}

        for step in range(MAX_DECODE_LEN):
            node_x, fa_slot = stack[-1]

            # Gather hidden states from already-attached neighbors
            cur_h_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1, -1, self.hidden_size) if len(cur_h_nei) > 0 else zero_pad

            cur_x = self.embedding(create_var(torch.LongTensor([node_x.wid])))
            cur_h = cur_h_nei.sum(dim=1)

            # Stop prediction: should we backtrack?
            stop_hiddens = torch.cat([cur_x, cur_h], dim=1)
            stop_hiddens = F.relu(self.U_i(stop_hiddens))
            stop_score = self.aggregate(stop_hiddens, contexts, x_tree_vecs, 'stop', cond_vec)

            if prob_decode:
                backtrack = (torch.bernoulli(torch.sigmoid(stop_score)).item() == 0)
            else:
                backtrack = (stop_score.item() < 0)

            if not backtrack:
                # Word prediction: which child node type to add?
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                pred_score = self.aggregate(new_h, contexts, x_tree_vecs, 'word', cond_vec)

                if prob_decode:
                    sort_wid = torch.multinomial(F.softmax(pred_score, dim=1).squeeze(), 5)
                else:
                    _, sort_wid = torch.sort(pred_score, dim=1, descending=True)
                    sort_wid = sort_wid.data.squeeze()

                # Find the first valid child that is chemically compatible
                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = MolTreeNode(self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True
                else:
                    node_y = MolTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.wid = next_wid
                    node_y.idx = len(all_nodes)
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx, node_y.idx)] = new_h[0]
                    stack.append((node_y, next_slots))
                    all_nodes.append(node_y)

            if backtrack:
                # Cannot expand further: backtrack to parent
                if len(stack) == 1:
                    break  # Reached root; decoding complete

                node_fa, _ = stack[-2]
                cur_h_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx]
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1, -1, self.hidden_size) if len(cur_h_nei) > 0 else zero_pad
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                h[(node_x.idx, node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes


def dfs(stack, x, fa_idx):
    """
    Depth-first traversal of the junction tree.

    Records (parent, child, direction) tuples for teacher-forcing training.
    direction=1 indicates a forward edge (parent→child),
    direction=0 indicates a backtracking edge (child→parent).
    """
    for y in x.neighbors:
        if y.idx == fa_idx:
            continue
        stack.append((x, y, 1))
        dfs(stack, y, x.idx)
        stack.append((y, x, 0))


def have_slots(fa_slots, ch_slots):
    """
    Check whether two substructures have compatible attachment slots.

    Returns True if the child can be attached to the parent based on
    matching atom types, formal charges, and hydrogen counts.
    """
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True

    matches = []
    for i, s1 in enumerate(fa_slots):
        a1, c1, h1 = s1
        for j, s2 in enumerate(ch_slots):
            a2, c2, h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append((i, j))

    if len(matches) == 0:
        return False

    fa_match, ch_match = zip(*matches)
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2:
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2:
        ch_slots.pop(ch_match[0])

    return True


def can_assemble(node_x, node_y):
    """
    Check whether two tree nodes can be structurally assembled.

    Returns True if there exists at least one valid attachment configuration
    between the two substructures without causing steric clashes.
    """
    node_x.nid = 1
    node_x.is_leaf = False
    set_atommap(node_x.mol, node_x.nid)

    neis = node_x.neighbors + [node_y]
    for i, nei in enumerate(neis):
        nei.nid = i + 2
        nei.is_leaf = (len(nei.neighbors) <= 1)
        if nei.is_leaf:
            set_atommap(nei.mol, 0)
        else:
            set_atommap(nei.mol, nei.nid)

    # Separate multi-atom neighbors from single-atom (singleton) neighbors
    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors

    cands, aroma_scores = enum_assemble(node_x, neighbors)
    return len(cands) > 0
