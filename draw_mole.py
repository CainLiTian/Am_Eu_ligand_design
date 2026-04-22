import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)

df = pd.read_excel('/home/dc/data_new/ppo_results/22/training_log.xlsx')
smiles_list = df['smiles'].dropna().head(100).tolist()
reward_list = df['reward'].dropna().head(100).tolist()
sf_list = df['soft_sf'].dropna().head(100).tolist()

mols = []
legends = []
for smi, r, sf in zip(smiles_list, reward_list, sf_list):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mols.append(mol)
        # 修改点：在图例末尾增加一个换行符 \n，拉开间距
        legends.append(f"R:{r:.2f} SF:{sf:.2f}")

n_per_page = 25
for i in range(0, len(mols), n_per_page):
    batch = mols[i:i + n_per_page]
    batch_legends = legends[i:i + n_per_page]
    img = Draw.MolsToGridImage(
        batch,
        molsPerRow=5,
        subImgSize=(400, 400),
        legends=batch_legends,
        returnPNG=False
    )
    page = i // n_per_page + 1
    filename = f"/home/dc/data_new/ppo_results/22/molecules_page_{page}.png"
    img.save(filename)
    print(f"已保存: {filename}")