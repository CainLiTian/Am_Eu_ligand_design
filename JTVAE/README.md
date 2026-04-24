# JT-VAE: Junction Tree Variational Autoencoder

This module implements the Junction Tree Variational Autoencoder (JT-VAE) for molecular generation, based on the architecture proposed by Jin et al. (ICML 2018).

## Core Files

| File | Description |
|:------|:------------|
| `mol_tree.py` | MolTree and MolTreeNode data structures for junction tree representation |
| `vocab.py` | Chemical substructure vocabulary constructed from training molecules |
| `mpn.py` | Message Passing Network for atom-level graph encoding |
| `jtmpn.py` | Message Passing Network for tree-level substructure encoding |
| `jtnn_enc.py` | Tree-structured encoder (GraphGRU) for junction tree nodes |
| `jtnn_dec.py` | Autoregressive decoder for tree-structured molecular generation |
| `jtnn_vae.py` | Main JT-VAE model integrating encoder, decoder, and assembly prediction |
| `nnutils.py` | Utility functions for tensor manipulation and GRU operations |
| `chemutils.py` | Chemical utility functions (assembly enumeration, atom mapping) |
| `datautils.py` | Data loading, batching, and tensorization utilities |

## Training

```bash
python trainjtnn.py
```

Training configuration and hyperparameters are defined at the top of `trainjtnn.py`. The script supports:

- **Pretraining** on CSD-derived metal–organic complexes
- **Transfer learning** (fine-tuning) on Am/Eu extraction ligands
- **Conditional generation** with solvent feature vectors
- KL annealing with linear warm-up schedule
- Cosine annealing and plateau-based learning rate scheduling
- Early stopping based on validation loss

## Key Modifications from Original JT-VAE

- Added **conditional generation** support (`cond_dim`) for solvent-conditioned decoding
- Increased **dropout regularization** (0.3) to mitigate overfitting on small datasets
- Added **LayerNorm** in the decoder aggregation module
- **Transfer learning mode**: encoder weights can be frozen while fine-tuning decoder and latent mapping layers
- **Timeout-protected decoding** via multiprocessing to prevent hanging on invalid latent points

## Reference

Jin, W., Barzilay, R., & Jaakkola, T. (2018). Junction Tree Variational Autoencoder for Molecular Graph Generation. *Proceedings of the 35th International Conference on Machine Learning (ICML)*.

## Usage

```python
from jtnn_vae import JTNNVAE
from vocab import Vocab

# Load vocabulary
vocab = Vocab(["vocab.txt"])

# Initialize model
model = JTNNVAE(vocab, hidden_size=256, latent_size=48, depthT=15, depthG=3, cond_dim=8)

# Encode molecules
z_tree, z_mol, z_combined = model.get_sampled_latent_vector(["c1ccccc1"])

# Decode from latent vector
smiles = model.decode_from_latent(z_combined)
```
