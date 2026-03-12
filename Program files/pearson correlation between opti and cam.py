import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Plot style header (same as previous, LaTeX-safe)
plt.rcParams.update({'text.usetex': False})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({
    'font.serif': [
        'Times New Roman', 'Times', 'DejaVu Serif',
        'Computer Modern Roman'
    ]
})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 9})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.close('all')
# ============================================================

# --- Load Excel file ---
file_path = "../data/filtered_Coordinates.xlsx"
df = pd.read_excel(file_path)

opt_vars = ['opt_Z', 'opt_Y', 'opt_X']
force_vars = ['LF_X', 'F_Y', 'RF_Z']

# --- Compute 3x3 Pearson correlation matrix ---
corr_matrix = pd.DataFrame(
    index=opt_vars,
    columns=force_vars,
    dtype=float
)

for opt in opt_vars:
    for force in force_vars:
        corr_matrix.loc[opt, force] = df[opt].corr(df[force], method='pearson')

# --- Plot ---
fig, ax = plt.subplots(figsize=(3, 3))

cax = ax.matshow(corr_matrix, vmin=-1, vmax=1)
fig.colorbar(cax, fraction=0.046, pad=0.04)

# Move x-axis ticks and label to bottom
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')

# Axis ticks and labels
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))

ax.set_xticklabels(['x (px)', 'y (px)', 'z (px)'])
ax.set_yticklabels(['x (mm)', 'y (mm)', 'z (mm)'])

ax.set_xlabel('stereo pixel axis')
ax.set_ylabel('OptiTrack axis')

# Annotate correlation values
for i in range(3):
    for j in range(3):
        ax.text(
            j, i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha='center',
            va='center'
        )

#ax.set_title('Pearson Correlation', pad=8)

plt.tight_layout()
plt.show()
