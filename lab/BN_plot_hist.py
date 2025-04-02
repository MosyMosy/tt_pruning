import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.cm as cm


conve_layer = ["first_conv", "second_conv"][0]
module = ["", "class_head_"][1]
data = ["clean", "lidar"][0]
adapt = ["", "_updated"][0]
# Load input tensor
input_data = (
    torch.load(f"lab/Batch_Norm_Analyzer/{module}{conve_layer}_input_{data}.pth", weights_only=True)
    .cpu()
    .detach()
)

# Load BN stats
running_means = (
    torch.load(f"lab/Batch_Norm_Analyzer/{module}{conve_layer}_running_mean{adapt}.pth", weights_only=True)
    .cpu()
    .detach()
    .numpy()
)
running_vars = (
    torch.load(f"lab/Batch_Norm_Analyzer/{module}{conve_layer}_running_var{adapt}.pth", weights_only=True)
    .cpu()
    .detach()
    .numpy()
)
running_stds = np.sqrt(running_vars)

# === Settings ===
channels = [0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111]
num_channels = len(channels)

if data == "clean":
    colors = cm.Blues(np.linspace(0.4, 0.9, num_channels))
elif data == "lidar":
    colors = cm.Reds(np.linspace(0.3, 0.8, num_channels))
    
if adapt == "":
    gaussian_color = "black"
else:
    gaussian_color = "green"
# Define global x-axis limits based on input data
x_min = input_data.min().item() - 0.1
x_max = input_data.max().item() + 0.1

ridge_height = 1.0
num_std = 1
alpha_hist = 0.6
fig_height_inches = 6
fig_width_inches = 5

positions = np.arange(num_channels)[::-1] * (fig_height_inches / num_channels)
positions -= positions[-1]

fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))

for idx, i in enumerate(channels):
    values = input_data[:, i].flatten().numpy()
    y_base = positions[idx]

    # Histogram
    hist_vals, bin_edges = np.histogram(values, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_vals *= ridge_height / hist_vals.max()
    hist_vals_offset = hist_vals + y_base

    # Gaussian (per-channel range)
    mean = running_means[i]
    std = running_stds[i]
    x_gauss = np.linspace(mean - num_std * std * 3, mean + num_std * std * 3, 300)
    g_y = norm.pdf(x_gauss, loc=mean, scale=std)
    g_y *= ridge_height / g_y.max()
    g_y_offset = g_y + y_base

    # Plot histogram (middle layer)
    ax.fill_between(
        bin_centers,
        y_base,
        hist_vals_offset,
        color=colors[idx],
        alpha=alpha_hist,
        zorder=i,
    )

    # Plot Gaussian (top layer)
    ax.plot(
        x_gauss,
        g_y_offset,
        color=gaussian_color,
        linewidth=1.0,
        linestyle="--",
        alpha=alpha_hist,
        zorder=i,
    )

    # Ground line (bottom layer)
    ax.plot(
        [x_min, x_max],
        [y_base, y_base],
        color=colors[idx],
        linewidth=1.5,        
        zorder=1,
    )

    # Channel label (optionally topmost)
    ax.text(
        x_min - 0.1 * (x_max - x_min),
        y_base,
        f"ch {i + 1}",
        va="center",
        ha="right",
        fontsize=14,
        zorder=4,
    )

# Styling
ax.set_yticks([])
ax.set_ylim(0, positions[0] + ridge_height * 1.2)
ax.set_xlim(x_min, x_max)
ax.set_xlabel("Value", fontsize=12)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.tight_layout()
plt.savefig(
    f"lab/Batch_Norm_Analyzer/{module}_{conve_layer}_hist_{data}__{adapt}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

