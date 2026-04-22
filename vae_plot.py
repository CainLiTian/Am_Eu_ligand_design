import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from rdkit import Chem
from rdkit.Chem import AllChem
import os

# 读取数据
df = pd.read_excel('/home/dc/data_new/VAE_test/pre_vae_test_results.xlsx')

# 只保留有效分子
valid_df = df[df['valid'] == True].copy()
print(f"有效分子数量: {len(valid_df)}")

# 生成Morgan指纹
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp)
        else:
            return None
    except:
        return None

# 为所有有效分子生成指纹
fingerprints = []
valid_indices = []

for idx, row in valid_df.iterrows():
    fp = smiles_to_fingerprint(row['smiles'])
    if fp is not None:
        fingerprints.append(fp)
        valid_indices.append(idx)

# 创建指纹矩阵
X = np.array(fingerprints)
y_sa = valid_df.loc[valid_indices, 'sa_score'].values

print(f"指纹矩阵形状: {X.shape}")

# 1. t-SNE降维
print("正在计算t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# 绘制并保存t-SNE图
plt.figure(figsize=(10, 8))
sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                 c=y_sa, cmap='viridis',
                 s=20, alpha=0.7)
plt.colorbar(sc, label='SAscore')
plt.title('t-SNE Visualization (SAscore)', fontsize=14)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.savefig('/home/dc/data_new/VAE_test/pre_tsne_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("t-SNE图已保存")

# 2. UMAP降维
print("正在计算UMAP...")
reducer = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.1)
X_umap = reducer.fit_transform(X)

# 绘制并保存UMAP图
plt.figure(figsize=(10, 8))
sc = plt.scatter(X_umap[:, 0], X_umap[:, 1],
                 c=y_sa, cmap='viridis',
                 s=20, alpha=0.7)
plt.colorbar(sc, label='SAscore')
plt.title('UMAP Visualization (SAscore)', fontsize=14)
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.savefig('/home/dc/data_new/VAE_test/pre_umap_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("UMAP图已保存")

print(f"\n文件已保存到: /home/dc/data_new/VAE_test/")
print("  - tsne_plot.png")
print("  - umap_plot.png")