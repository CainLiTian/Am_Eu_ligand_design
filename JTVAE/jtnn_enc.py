import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from mol_tree import Vocab, MolTree
from nnutils import create_var, index_select_ND


class JTNNEncoder(nn.Module):
    """
    Junction Tree Encoder.

    Encodes a batch of junction trees into continuous vector representations.
    The encoder processes nodes through a tree-structured message-passing
    network (GraphGRU) and produces a root representation for each tree.
    """

    def __init__(self, hidden_size, depth, embedding):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.embedding = embedding
        self.outputNN = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )
        self.GRU = GraphGRU(hidden_size, hidden_size, depth=depth)

    def forward(self, fnode, fmess, node_graph, mess_graph, scope):
        """
        Forward pass of the tree encoder.

        Args:
            fnode: Node feature indices.
            fmess: Message source node indices.
            node_graph: Adjacency list mapping each node to its incoming messages.
            mess_graph: Adjacency list mapping each message to its precursor messages.
            scope: List of (start_idx, length) tuples defining each tree in the batch.

        Returns:
            tree_vecs: Root node vectors for each tree in the batch.
            messages: Hidden states of all messages.
        """
        fnode = create_var(fnode)
        fmess = create_var(fmess)
        node_graph = create_var(node_graph)
        mess_graph = create_var(mess_graph)
        messages = create_var(torch.zeros(mess_graph.size(0), self.hidden_size))

        # Embed node features and initialize messages
        fnode = self.embedding(fnode)
        fmess = index_select_ND(fnode, 0, fmess)
        messages = self.GRU(messages, fmess, mess_graph)

        # Aggregate incoming messages for each node
        mess_nei = index_select_ND(messages, 0, node_graph)
        node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=-1)
        node_vecs = self.outputNN(node_vecs)

        # Extract root node representation for each tree in the batch
        max_len = max([x for _, x in scope])
        batch_vecs = []
        for st, le in scope:
            cur_vecs = node_vecs[st]  # Root node is the first node
            batch_vecs.append(cur_vecs)
        tree_vecs = torch.stack(batch_vecs, dim=0)
        return tree_vecs, messages

    @staticmethod
    def tensorize(tree_batch):
        """
        Convert a list of MolTree objects into tensorized format.

        Returns:
            Tuple of (fnode, fmess, node_graph, mess_graph, scope) and mess_dict.
        """
        node_batch = []
        scope = []
        for tree in tree_batch:
            scope.append((len(node_batch), len(tree.nodes)))
            node_batch.extend(tree.nodes)

        return JTNNEncoder.tensorize_nodes(node_batch, scope)

    @staticmethod
    def tensorize_nodes(node_batch, scope):
        """
        Build tensorized representations from a flat list of tree nodes.

        Constructs message-passing structures:
        - node_graph: For each node, a list of incoming message indices.
        - mess_graph: For each message, a list of child message indices.
        - fmess: The source node index for each message.

        Returns:
            (fnode, fmess, node_graph, mess_graph, scope) tuple and mess_dict.
        """
        messages, mess_dict = [None], {}
        fnode = []
        for x in node_batch:
            fnode.append(x.wid)
            for y in x.neighbors:
                mess_dict[(x.idx, y.idx)] = len(messages)
                messages.append((x, y))

        node_graph = [[] for i in range(len(node_batch))]
        mess_graph = [[] for i in range(len(messages))]
        fmess = [0] * len(messages)

        for x, y in messages[1:]:
            mid1 = mess_dict[(x.idx, y.idx)]
            fmess[mid1] = x.idx
            node_graph[y.idx].append(mid1)
            for z in y.neighbors:
                if z.idx == x.idx:
                    continue
                mid2 = mess_dict[(y.idx, z.idx)]
                mess_graph[mid2].append(mid1)

        # Pad adjacency lists to uniform length
        max_len = max([len(t) for t in node_graph] + [1])
        for t in node_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        max_len = max([len(t) for t in mess_graph] + [1])
        for t in mess_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        mess_graph = torch.LongTensor(mess_graph)
        node_graph = torch.LongTensor(node_graph)
        fmess = torch.LongTensor(fmess)
        fnode = torch.LongTensor(fnode)
        return (fnode, fmess, node_graph, mess_graph, scope), mess_dict


class GraphGRU(nn.Module):
    """
    Gated Recurrent Unit for graph-structured message passing.

    This module performs tree-structured message propagation over multiple
    depth iterations, using a GRU-like gating mechanism to control information
    flow between nodes.
    """

    def __init__(self, input_size, hidden_size, depth, dropout=0):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth
        self.dropout_layer = nn.Dropout(p=dropout)

        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        # Reset gate
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        # Candidate hidden state
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h, x, mess_graph):
        """
        Execute depth iterations of graph message passing.

        Args:
            h: Initial hidden states for all messages.
            x: Input features for each message (embedded node features).
            mess_graph: Adjacency list of child-to-parent message connections.

        Returns:
            Updated hidden states after 'depth' rounds of propagation.
        """
        mask = torch.ones(h.size(0), 1, device=x.device)
        mask[0] = 0  # First message is padding
        mask = create_var(mask)

        for _ in range(self.depth):
            # Aggregate incoming messages from neighbors
            h_nei = index_select_ND(h, 0, mess_graph)
            sum_h = h_nei.sum(dim=1)

            # Update gate: controls how much of the previous state to retain
            z_input = torch.cat([x, sum_h], dim=1)
            z = torch.sigmoid(self.W_z(z_input))

            # Reset gate: controls how much neighbor information to incorporate
            r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
            r_2 = self.U_r(h_nei)
            r = torch.sigmoid(r_1 + r_2)

            # Candidate hidden state with gated neighbor information
            gated_h = r * h_nei
            sum_gated_h = gated_h.sum(dim=1)
            h_input = torch.cat([x, sum_gated_h], dim=1)
            pre_h = torch.tanh(self.W_h(h_input))
            pre_h = self.dropout_layer(pre_h)

            # Combine previous state and candidate state via update gate
            h = (1.0 - z) * sum_h + z * pre_h
            h = self.dropout_layer(h)
            h = h * mask  # Zero out the padding message

        return h
