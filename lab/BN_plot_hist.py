import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.cm as cm


def plot_histograms(
    ax,
    input_data,
    means,
    vars,
    colors,
    gaussian_color,
    label_prefix="",
    show_labels=True,
    x_label="Value",
    x_min=None,
    x_max=None,
):
    running_stds = np.sqrt(vars)
    positions = np.arange(num_channels)[::-1] * (fig_height_inches / num_channels)
    positions -= positions[-1]

    for idx, i in enumerate(channels):
        values = input_data[:, i].flatten().numpy()
        y_base = positions[idx]

        # Histogram
        hist_vals, bin_edges = np.histogram(values, bins=100, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist_vals *= ridge_height / hist_vals.max()
        hist_vals_offset = hist_vals + y_base

        # Gaussian - BN stats
        mean_bn = means[i]
        std_bn = running_stds[i]
        x_gauss_bn = np.linspace(mean_bn - num_std * std_bn * 3, mean_bn + num_std * std_bn * 3, 300)
        g_y_bn = norm.pdf(x_gauss_bn, loc=mean_bn, scale=std_bn)
        g_y_bn *= ridge_height / g_y_bn.max()
        g_y_bn_offset = g_y_bn + y_base

        # Gaussian - data stats
        mean_data = values.mean()
        std_data = values.std()
        x_gauss_data = np.linspace(mean_data - num_std * std_data * 3, mean_data + num_std * std_data * 3, 300)
        g_y_data = norm.pdf(x_gauss_data, loc=mean_data, scale=std_data)
        g_y_data *= ridge_height / g_y_data.max()
        g_y_data_offset = g_y_data + y_base

        # Plot
        ax.fill_between(bin_centers, y_base, hist_vals_offset, color=colors[idx], alpha=alpha_hist, zorder=i)
        ax.plot(x_gauss_bn, g_y_bn_offset, color=gaussian_color, linewidth=1.0, linestyle="--", alpha=alpha_hist, zorder=i)
        ax.plot(x_gauss_data, g_y_data_offset, color=colors[idx], linewidth=1.0, linestyle="-", alpha=alpha_hist, zorder=i)
        ax.plot([x_min, x_max], [y_base, y_base], color=colors[idx], linewidth=1.5, zorder=1)

        if show_labels:
            ax.text(
                x_min - 0.0 * (x_max - x_min),
                y_base,
                f"{label_prefix}ch {i + 1}",
                va="center",
                ha="right",
                fontsize=12,
                zorder=4,
            )

    ax.set_yticks([])
    ax.set_ylim(0, positions[0] + ridge_height * 1.2)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(x_label, fontsize=12)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)


# === Parameters ===
channels = [0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111]
num_channels = len(channels)
colors_clean = cm.Blues(np.linspace(0.4, 0.9, num_channels))
colors_corrupt = cm.Reds(np.linspace(0.3, 0.8, num_channels))
gaussian_color = "black"
gaussian_color_adapted = "green"
ridge_height = 1.0
num_std = 1
alpha_hist = 0.6
fig_height_inches = 6
fig_width_inches = 5

# === Load data for both modules ===
def load_data(module):
    input_clean = torch.load(f"lab/Batch_Norm_Analyzer/{module}first_conv_input_clean.pth", weights_only=True).cpu().detach()
    input_corrupt = torch.load(f"lab/Batch_Norm_Analyzer/{module}first_conv_input_lidar.pth", weights_only=True).cpu().detach()
    rm = torch.load(f"lab/Batch_Norm_Analyzer/{module}first_conv_running_mean.pth", weights_only=True).cpu().detach().numpy()
    rv = torch.load(f"lab/Batch_Norm_Analyzer/{module}first_conv_running_var.pth", weights_only=True).cpu().detach().numpy()
    rm_upd = torch.load(f"lab/Batch_Norm_Analyzer/{module}first_conv_running_mean_updated.pth", weights_only=True).cpu().detach().numpy()
    rv_upd = torch.load(f"lab/Batch_Norm_Analyzer/{module}first_conv_running_var_updated.pth", weights_only=True).cpu().detach().numpy()
    return input_clean, input_corrupt, rm, rv, rm_upd, rv_upd

input_clean_main, input_corrupt_main, rm_main, rv_main, rm_upd_main, rv_upd_main = load_data("")
input_clean_head, input_corrupt_head, rm_head, rv_head, rm_upd_head, rv_upd_head = load_data("class_head_")

# === Per-row x-range ===
x_min_main = min(input_clean_main.min().item(), input_corrupt_main.min().item()) - 0.1
x_max_main = max(input_clean_main.max().item(), input_corrupt_main.max().item()) + 0.1
x_min_head = min(input_clean_head.min().item(), input_corrupt_head.min().item()) - 0.1
x_max_head = max(input_clean_head.max().item(), input_corrupt_head.max().item()) + 0.1

# === Create figure ===
fig, axes = plt.subplots(2, 3, figsize=(fig_width_inches * 3, fig_height_inches * 2), sharey=True)

# --- Top row: Main branch ---
plot_histograms(axes[0, 0], input_clean_main, rm_main, rv_main, colors_clean, gaussian_color, show_labels=True, x_label="", x_min=x_min_main, x_max=x_max_main)
plot_histograms(axes[0, 1], input_corrupt_main, rm_main, rv_main, colors_corrupt, gaussian_color, show_labels=False, x_label="", x_min=x_min_main, x_max=x_max_main)
plot_histograms(axes[0, 2], input_corrupt_main, rm_upd_main, rv_upd_main, colors_corrupt, gaussian_color_adapted, show_labels=False, x_label="", x_min=x_min_main, x_max=x_max_main)

# --- Bottom row: Class head ---
plot_histograms(axes[1, 0], input_clean_head, rm_head, rv_head, colors_clean, gaussian_color, show_labels=True, x_min=x_min_head, x_max=x_max_head)
plot_histograms(axes[1, 1], input_corrupt_head, rm_head, rv_head, colors_corrupt, gaussian_color, show_labels=False, x_min=x_min_head, x_max=x_max_head)
plot_histograms(axes[1, 2], input_corrupt_head, rm_upd_head, rv_upd_head, colors_corrupt, gaussian_color_adapted, show_labels=False, x_min=x_min_head, x_max=x_max_head)

# Titles ONLY for the top row
titles = ["Clean with Pretrained Statistics", "Corrupt with Pretrained Statistics", "Corrupt with Adapted Statistics"]
for ax, title in zip(axes[0], titles):
    ax.set_title(title, fontsize=14, pad=10)


# Adjust layout to give the title space
plt.tight_layout()
plt.savefig("lab/Batch_Norm_Analyzer/hist_double_row.png", dpi=300, bbox_inches="tight")
plt.show()
