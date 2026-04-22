import rdkit
import rdkit.Chem as Chem
import copy

def get_slots(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
    except:
        print(f"[!] Invalid SMILES during slot extraction: {smiles}")
        return []

    slots = []
    for atom in mol.GetAtoms():
        try:
            max_val = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
            valence = atom.GetExplicitValence()
            if atom.GetSymbol() == 'C' and valence > max_val:
                slots.append((atom.GetSymbol(), atom.GetFormalCharge(), 0))
            else:
                slots.append((atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()))
        except:
            slots.append((atom.GetSymbol(), atom.GetFormalCharge(), 0))
    return slots

class Vocab(object):
    benzynes = ['C1=CC=CC=C1', 'C1=CC=NC=C1', 'C1=CC=NN=C1', 'C1=CN=CC=N1', 'C1=CN=CN=C1', 'C1=CN=NC=N1', 'C1=CN=NN=C1', 'C1=NC=NC=N1', 'C1=NN=CN=N1']
    penzynes = ['C1=C[NH]C=C1', 'C1=C[NH]C=N1', 'C1=C[NH]N=C1', 'C1=C[NH]N=N1', 'C1=COC=C1', 'C1=COC=N1', 'C1=CON=C1', 'C1=CSC=C1', 'C1=CSC=N1',
                'C1=CSN=C1', 'C1=CSN=N1', 'C1=NN=C[NH]1', 'C1=NN=CO1', 'C1=NN=CS1', 'C1=N[NH]C=N1', 'C1=N[NH]N=C1', 'C1=N[NH]N=N1',
                'C1=NN=N[NH]1', 'C1=NN=NS1', 'C1=NOC=N1', 'C1=NON=C1', 'C1=NSC=N1', 'C1=NSN=C1']

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]

        Vocab.benzynes = [s for s in smiles_list if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 6] + ['C1=CCNCC1']
        Vocab.penzynes = [s for s in smiles_list if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 5] + ['C1=NCCN1', 'C1=NNCC1']

    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)