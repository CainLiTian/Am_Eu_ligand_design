import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
import os

# ==============================
# 文件路径
# ==============================

input_file = "/home/dc/data_new/ppo_results/22/training_log.xlsx"

output_dir = "/home/dc/data_new/ppo_results/22/scaffold/"

os.makedirs(output_dir, exist_ok=True)

# ==============================
# 读取数据
# ==============================

df = pd.read_excel(input_file)
df = df.drop_duplicates(subset=['smiles'], keep='first')

# ==============================
# 计算 scaffold
# ==============================

def get_scaffold(smi):
    if pd.isna(smi):
        return None

    mol = Chem.MolFromSmiles(str(smi))

    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold.GetNumAtoms() == 0:
        return None

    return Chem.MolToSmiles(scaffold)

df["scaffold"] = df["smiles"].apply(get_scaffold)

# 删除空 scaffold
df = df[df["scaffold"].notna()]

# ==============================
# 删除单苯环 scaffold
# ==============================

phenyl = "c1ccccc1"

df = df[df["scaffold"] != phenyl]

# ==============================
# scaffold统计
# ==============================

scaffold_stats = (
    df.groupby("scaffold")
    .agg(
        count=("scaffold","size"),
        avg_sf=("soft_sf","mean"),
        avg_reward=("reward","mean")
    )
    .sort_values("count",ascending=False)
)

# 保存统计表
stats_file = os.path.join(output_dir,"scaffold_statistics_TEST.xlsx")

scaffold_stats.to_excel(stats_file)

print("scaffold statistics saved")

# ==============================
# top10 scaffold
# ==============================

top = scaffold_stats.head(10).reset_index()

top["label"] = [f"scaffold{i+1}" for i in range(len(top))]

# ==============================
# 颜色渐变
# ==============================

counts = top["count"].values

norm = plt.Normalize(min(counts),max(counts))

cmap = cm.get_cmap("viridis")

colors = cmap(norm(counts))

# ==============================
# 画图
# ==============================

fig = plt.figure(figsize=(12,6))

gs = fig.add_gridspec(1,2,width_ratios=[1.5,1])

ax = fig.add_subplot(gs[0])

y = np.arange(len(top))

bars = ax.barh(
    y,
    top["count"],
    color=colors
)

ax.set_yticks(y)

ax.set_yticklabels(top["label"])

ax.set_xlabel("Count")

ax.invert_yaxis()

# 添加标题
ax.set_title("Top 10 Common Molecule Scaffold", fontsize=14, fontweight='bold')

# 标注 SF reward
max_count = top["count"].max()
x_limit = max_count * 1.3  # 将横坐标范围扩大30%
ax.set_xlim(0, x_limit)

for i,bar in enumerate(bars):

    width = bar.get_width()

    sf = top.loc[i,"avg_sf"]

    reward = top.loc[i,"avg_reward"]

    text = f"SF={sf:.2f}\nR={reward:.2f}"

    ax.text(
        width + max_count * 0.05,  # 在条形图宽度基础上加上最大值的5%作为偏移
        bar.get_y()+bar.get_height()/2,
        text,
        va="center",
        fontsize=9
    )

# ==============================
# 结构图 - 只画scaffold 1,4,8,10
# ==============================

ax2 = fig.add_subplot(gs[1])

ax2.axis("off")

# 选择索引0,3,7,9对应的scaffold (scaffold1, scaffold4, scaffold8, scaffold10)
#selected_indices = [0, 3, 7, 9]
selected_indices = [0,2,3,4,5,8]
selected_mols = [Chem.MolFromSmiles(top.iloc[i]["scaffold"]) for i in selected_indices]
selected_labels = [top.iloc[i]["label"] for i in selected_indices]

img = Draw.MolsToGridImage(
    selected_mols,
    molsPerRow=2,
    subImgSize=(250,250),
    legends=selected_labels
)

ax2.imshow(img)

# ==============================
# 保存
# ==============================

fig_path = os.path.join(output_dir,"scaffold_barplot_TEST.png")

plt.tight_layout()

plt.savefig(fig_path,dpi=300)

plt.show()

print("figure saved")