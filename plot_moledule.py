import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.patches import Rectangle
import os

# 读取Excel文件
file_path = '/home/dc/data_new/ppo_results/training_log.xlsx'
df = pd.read_excel(file_path)

# ========== 先全局去重：基于SMILES去重，保留soft_sf最高的 ==========
df_unique = df.sort_values('soft_sf', ascending=False).drop_duplicates(subset=['smiles'], keep='first').reset_index(drop=True)
print(f"原始数据: {len(df)} 行, 去重后: {len(df_unique)} 个唯一分子")
# =================================================================

# 取前20个（去重后）
top_20 = df_unique.head(20).reset_index(drop=True)

# 提取smiles列表和对应的soft_sf值
smiles_list = top_20['smiles'].tolist()
sf_values = top_20['soft_sf'].tolist()

# 将smiles转换为RDKit分子对象
mols = []
valid_indices = []
valid_sf_values = []
valid_smiles = []

for i, (smiles, sf_val) in enumerate(zip(smiles_list, sf_values)):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)
            valid_sf_values.append(sf_val)
            valid_smiles.append(smiles)
        else:
            print(f"无效的SMILES (索引{i}): {smiles}")
    except Exception as e:
        print(f"解析SMILES失败 (索引{i}): {smiles}, 错误: {e}")

# 创建5x4的网格图，调整整体尺寸让分子比例更好
fig, axes = plt.subplots(5, 4, figsize=(18, 22))
axes = axes.flatten()

# 为每个分子绘制图像
for idx, (mol, sf_val, smiles) in enumerate(zip(mols, valid_sf_values, valid_smiles)):
    # 绘制分子结构，增大尺寸让分子更清晰
    img = Draw.MolToImage(mol, size=(400, 400))
    axes[idx].imshow(img)
    axes[idx].axis('off')

    # 在图像下方添加SF值标注
    axes[idx].text(0.5, -0.05, f'SF: {sf_val:.4f}',
                   transform=axes[idx].transAxes,
                   fontsize=10, ha='center', va='top',
                   fontweight='bold')

# 隐藏多余的子图
for idx in range(len(mols), 20):
    axes[idx].axis('off')
    axes[idx].set_visible(False)

# 调整布局，让分子图比例更协调
plt.tight_layout(pad=0.8)
plt.subplots_adjust(wspace=0.1, hspace=0.15, top=0.98, bottom=0.02)

# 保存图像
output_path = '/home/dc/data_new/ppo_results/baseline/top20_molecules_with_sf_only.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图像已保存到: {output_path}")

# 显示图像
plt.show()

# 打印详细信息表格
print(f"\n=== Top {len(valid_smiles)} Unique Molecules (by soft_sf) ===")
print(f"{'Rank':<6} {'soft_sf':<15} {'SMILES':<70}")
print("-" * 95)
for i, (sf_val, smiles) in enumerate(zip(valid_sf_values, valid_smiles)):
    # 截断过长的SMILES
    smiles_display = smiles if len(smiles) <= 70 else smiles[:67] + "..."
    print(f"{i + 1:<6} {sf_val:<15.6f} {smiles_display:<70}")

# 保存详细数据到CSV（去重后的前20个）
output_csv = '/home/dc/data_new/ppo_results/baseline/top20_molecules_details.csv'
top_20[['smiles', 'soft_sf']].to_csv(output_csv, index=False)
print(f"\n详细数据已保存到: {output_csv}")