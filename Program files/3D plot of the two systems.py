import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Plot style header (LaTeX-safe)
plt.rcParams.update({'text.usetex': False})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']
})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.close('all')
# ============================================================

# ================= USER CONTROLS =================
PLOT_PROJECTIONS = True
PROJECTION_STYLE = 'lines'   # 'dots' or 'lines'

proj_color_xy = "k"
proj_color_yz = '0'
proj_color_zx = '0.4'

proj_marker_size = 0.2
proj_line_width = 0.2
proj_alpha = 1
# =================================================

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
file_path = "../data/filtered_Coordinates.xlsx"
df = pd.read_excel(file_path)

x_opti = df["opt_X"].values
y_opti = df["opt_Y"].values
z_opti = df["opt_Z"].values

x_flir = df["LF_X"].values
y_flir = df["F_Y"].values
z_flir = df["RF_Z"].values

# ------------------------------------------------------------
# Coordinate transformation
# ------------------------------------------------------------
def transform_coordinates(x, y, z, rotation_deg, translation):
    rx, ry, rz = np.radians(rotation_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    pts = np.vstack((x, y, z))
    pts_t = R @ pts + np.reshape(translation, (3, 1))
    return pts_t[0], pts_t[1], pts_t[2]

rotation_deg = [0, -45, 0]
translation = [0, 0, 0]
x_flir_t, y_flir_t, z_flir_t = transform_coordinates(
    x_flir, y_flir, z_flir, rotation_deg, translation
)

# Uncomment to disable transform
x_flir_t, y_flir_t, z_flir_t = x_flir, y_flir, z_flir

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(6.5, 3))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# ------------------ OptiTrack ------------------
if PLOT_PROJECTIONS:

    # XY
    ax1.plot(x_opti, y_opti, np.full_like(z_opti, z_opti.min()),
             color=proj_color_xy, lw=proj_line_width, alpha=proj_alpha)

    # YZ
    x_offset = x_opti.min() - 0.03 * (x_opti.max() - x_opti.min())
    ax1.plot(np.full_like(x_opti, x_offset), y_opti, z_opti,
             color=proj_color_yz, lw=proj_line_width, alpha=proj_alpha)

    # ZX
    y_offset = y_opti.min() - 0.03 * (y_opti.max() - y_opti.min())
    ax1.plot(x_opti, np.full_like(y_opti, y_offset), z_opti,
             color=proj_color_zx, lw=proj_line_width, alpha=proj_alpha)

ax1.plot(x_opti, y_opti, z_opti,
         color='green', linewidth=1,
         label='OptiTrack trajectory')

ax1.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')
ax1.set_zlabel('z (mm)')
ax1.legend(loc='upper center')
ax1.grid(True)

# ---------------- Stereo Pixel -----------------
if PLOT_PROJECTIONS:

    # XY
    ax2.plot(x_flir_t, y_flir_t,
             np.full_like(z_flir_t, z_flir_t.min()),
             color=proj_color_xy, lw=proj_line_width, alpha=proj_alpha)

    # YZ
    x_offset = x_flir_t.min() - 0.03 * (x_flir_t.max() - x_flir_t.min())
    ax2.plot(np.full_like(x_flir_t, x_offset), y_flir_t, z_flir_t,
             color=proj_color_yz, lw=proj_line_width, alpha=proj_alpha)

    # ZX
    y_offset = y_flir_t.min() - 0.03 * (y_flir_t.max() - y_flir_t.min())
    ax2.plot(x_flir_t, np.full_like(y_flir_t, y_offset), z_flir_t,
             color=proj_color_zx, lw=proj_line_width, alpha=proj_alpha)

ax2.plot(x_flir_t, y_flir_t, z_flir_t,
         color='red', linewidth=1,
         label='stereo pixel trajectory')

ax2.set_xlabel('x (px)')
ax2.set_ylabel('y (px)')
ax2.set_zlabel('z (px)')
ax2.legend(loc='upper center',)
ax2.grid(True)

# ------------------------------------------------------------
# Axis formatting
# ------------------------------------------------------------
for ax in [ax1, ax2]:
    ax.set_box_aspect([1, 1, 1])
    ax.tick_params(pad=6, direction='in')

ax1.view_init(elev=25, azim=-60)
ax2.view_init(elev=25, azim=-60)

plt.tight_layout(pad = 0)
plt.show()
