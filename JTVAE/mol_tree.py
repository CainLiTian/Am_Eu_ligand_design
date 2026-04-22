import rdkit
import rdkit.Chem as Chem
from chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo
from vocab import *


class MolTreeNode(object):
    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)
        if self.mol is None:
            raise ValueError(f"[!] MolTreeNode: Invalid SMILES used to build mol: {self.smiles}")

        self.clique = [x for x in clique]
        self.neighbors = []

    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf:
                continue
            for cidx in nei_node.clique:
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        if label_mol is None:
            self.label = None
            return None

        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label

    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands, aroma = enum_assemble(self, neighbors)
        new_cands = [cand for i, cand in enumerate(cands) if aroma[i] >= 0]
        if len(new_cands) > 0:
            cands = new_cands

        if len(cands) > 0:
            self.cands, _ = zip(*cands)
            self.cands = list(self.cands)
        else:
            self.cands = []


class MolTree(object):
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        if self.mol is None:
            self.valid = False
            return

        cliques, edges = tree_decomp(self.mol)
        self.nodes = []
        root = 0

        try:
            for i, c in enumerate(cliques):
                cmol = get_clique_mol(self.mol, c)
                if cmol is None:
                    self.valid = False
                    return
                node = MolTreeNode(get_smiles(cmol), c)
                self.nodes.append(node)
                if min(c) == 0:
                    root = i

            for x, y in edges:
                if x >= len(self.nodes) or y >= len(self.nodes):
                    continue
                self.nodes[x].add_neighbor(self.nodes[y])
                self.nodes[y].add_neighbor(self.nodes[x])

            if root > 0 and root < len(self.nodes):
                self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

            for i, node in enumerate(self.nodes):
                node.nid = i + 1
                if len(node.neighbors) > 1:
                    set_atommap(node.mol, node.nid)
                node.is_leaf = (len(node.neighbors) == 1)

            self.recover()
            self.assemble()

            # 只要求 label 存在；不再强制要求 node.cands 非空（方案B）
            for node in self.nodes:
                if node.label is None:
                    # 如果你希望更宽松，可以把这段注释掉（即使 label 缺失也算 valid）
                    self.valid = False
                    return

            self.valid = True

            self.valid = True
        except Exception:
            self.valid = False

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()
