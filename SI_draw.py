import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Read data
df = pd.read_excel('/home/dc/data_new/output.xlsx')

# Count metal frequencies
metal_counts = df['metal'].value_counts().to_dict()

# Define periodic table grid (periods 1-7, groups 1-18)
# Dictionary recording each element's position in periodic table (period, group)
periodic_table_positions = {
    # Period 1
    'H': (1, 1), 'He': (1, 18),
    # Period 2
    'Li': (2, 1), 'Be': (2, 2), 'B': (2, 13), 'C': (2, 14), 'N': (2, 15), 'O': (2, 16), 'F': (2, 17), 'Ne': (2, 18),
    # Period 3
    'Na': (3, 1), 'Mg': (3, 2), 'Al': (3, 13), 'Si': (3, 14), 'P': (3, 15), 'S': (3, 16), 'Cl': (3, 17), 'Ar': (3, 18),
    # Period 4
    'K': (4, 1), 'Ca': (4, 2), 'Sc': (4, 3), 'Ti': (4, 4), 'V': (4, 5), 'Cr': (4, 6), 'Mn': (4, 7), 'Fe': (4, 8),
    'Co': (4, 9), 'Ni': (4, 10), 'Cu': (4, 11), 'Zn': (4, 12), 'Ga': (4, 13), 'Ge': (4, 14), 'As': (4, 15),
    'Se': (4, 16), 'Br': (4, 17), 'Kr': (4, 18),
    # Period 5
    'Rb': (5, 1), 'Sr': (5, 2), 'Y': (5, 3), 'Zr': (5, 4), 'Nb': (5, 5), 'Mo': (5, 6), 'Tc': (5, 7), 'Ru': (5, 8),
    'Rh': (5, 9), 'Pd': (5, 10), 'Ag': (5, 11), 'Cd': (5, 12), 'In': (5, 13), 'Sn': (5, 14), 'Sb': (5, 15),
    'Te': (5, 16), 'I': (5, 17), 'Xe': (5, 18),
    # Period 6
    'Cs': (6, 1), 'Ba': (6, 2), 'La': (6, 3), 'Ce': (6, 4), 'Pr': (6, 5), 'Nd': (6, 6), 'Pm': (6, 7), 'Sm': (6, 8),
    'Eu': (6, 9), 'Gd': (6, 10), 'Tb': (6, 11), 'Dy': (6, 12), 'Ho': (6, 13), 'Er': (6, 14), 'Tm': (6, 15),
    'Yb': (6, 16), 'Lu': (6, 17), 'Hf': (6, 4), 'Ta': (6, 5), 'W': (6, 6), 'Re': (6, 7), 'Os': (6, 8),
    'Ir': (6, 9), 'Pt': (6, 10), 'Au': (6, 11), 'Hg': (6, 12), 'Tl': (6, 13), 'Pb': (6, 14), 'Bi': (6, 15),
    'Po': (6, 16), 'At': (6, 17), 'Rn': (6, 18),
    # Period 7
    'Fr': (7, 1), 'Ra': (7, 2), 'Ac': (7, 3), 'Th': (7, 4), 'Pa': (7, 5), 'U': (7, 6), 'Np': (7, 7), 'Pu': (7, 8),
    'Am': (7, 9), 'Cm': (7, 10), 'Bk': (7, 11), 'Cf': (7, 12), 'Es': (7, 13), 'Fm': (7, 14), 'Md': (7, 15),
    'No': (7, 16), 'Lr': (7, 17), 'Rf': (7, 4), 'Db': (7, 5), 'Sg': (7, 6), 'Bh': (7, 7), 'Hs': (7, 8),
    'Mt': (7, 9), 'Ds': (7, 10), 'Rg': (7, 11), 'Cn': (7, 12), 'Nh': (7, 13), 'Fl': (7, 14), 'Mc': (7, 15),
    'Lv': (7, 16), 'Ts': (7, 17), 'Og': (7, 18),
    # Lanthanides (placed in period 6, group offset handled)
    'La': (6, 3), 'Ce': (6, 4), 'Pr': (6, 5), 'Nd': (6, 6), 'Pm': (6, 7), 'Sm': (6, 8), 'Eu': (6, 9),
    'Gd': (6, 10), 'Tb': (6, 11), 'Dy': (6, 12), 'Ho': (6, 13), 'Er': (6, 14), 'Tm': (6, 15), 'Yb': (6, 16),
    'Lu': (6, 17),
    # Actinides (placed in period 7, group offset handled)
    'Ac': (7, 3), 'Th': (7, 4), 'Pa': (7, 5), 'U': (7, 6), 'Np': (7, 7), 'Pu': (7, 8), 'Am': (7, 9),
    'Cm': (7, 10), 'Bk': (7, 11), 'Cf': (7, 12), 'Es': (7, 13), 'Fm': (7, 14), 'Md': (7, 15), 'No': (7, 16),
    'Lr': (7, 17)
}

# Create periodic table grid data
periods = range(1, 8)
groups = range(1, 19)

# Initialize heatmap data matrix
heatmap_data = np.zeros((len(periods), len(groups)))

# Fill data
for metal, count in metal_counts.items():
    if metal in periodic_table_positions:
        period, group = periodic_table_positions[metal]
        # Convert to 0-based index
        period_idx = period - 1
        group_idx = group - 1
        heatmap_data[period_idx, group_idx] = count
    else:
        print(f"Warning: Element {metal} not found in periodic table positions")

# Create custom colormap
colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
cmap = LinearSegmentedColormap.from_list('custom_blues', colors, N=256)

# Plot heatmap
fig, ax = plt.subplots(figsize=(18, 8))

im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=max(metal_counts.values()))

# Add grid lines
for i in range(len(periods) + 1):
    ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
for j in range(len(groups) + 1):
    ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

# Set axes
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([str(g) for g in groups], fontsize=10)
ax.set_xlabel('Group', fontsize=12)

ax.set_yticks(range(len(periods)))
ax.set_yticklabels([str(p) for p in periods], fontsize=10)
ax.set_ylabel('Period', fontsize=12)

# Add element symbols and frequencies in each cell
for period in periods:
    for group in groups:
        period_idx = period - 1
        group_idx = group - 1
        count = heatmap_data[period_idx, group_idx]

        # Find element at this position
        element = None
        for sym, pos in periodic_table_positions.items():
            if pos == (period, group):
                element = sym
                break

        if element and count > 0:
            # For cells with data, display element symbol and frequency
            ax.text(group_idx, period_idx, f'{element}\n{int(count)}',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white' if count > max(metal_counts.values()) / 2 else 'black')
        elif element:
            # For elements that appear in dataset but have zero frequency (theoretically won't happen)
            ax.text(group_idx, period_idx, element,
                    ha='center', va='center', fontsize=8, color='gray', alpha=0.5)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label('Number of Complexes', fontsize=12)

plt.title('Metal Element Distribution Heatmap in CSD Mined Dataset', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/home/dc/data_new/SI/metal_periodic_table_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/dc/data_new/SI/metal_periodic_table_heatmap.pdf', bbox_inches='tight')
plt.show()

# Output statistics
print(f"Number of metal types in dataset: {len(metal_counts)}")
print(f"Total number of complexes: {len(df)}")
print("\nTop 10 most frequent metals:")
for metal, count in sorted(metal_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {metal}: {count}")

# Statistics by element classification
metal_df = pd.DataFrame(list(metal_counts.items()), columns=['metal', 'count'])
metal_df = metal_df.sort_values('count', ascending=False)

# Plot bar chart (top 20 metals)
plt.figure(figsize=(12, 6))
top20 = metal_df.head(20)
sns.barplot(data=top20, x='metal', y='count', palette='Blues_d')
plt.title('Top 20 Most Frequent Metal Elements in CSD Mined Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Metal Element', fontsize=12)
plt.ylabel('Number of Complexes', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/dc/data_new/SI/top20_metals_bar.png', dpi=300)
plt.savefig('/home/dc/data_new/SI/top20_metals_bar.pdf')
plt.show()