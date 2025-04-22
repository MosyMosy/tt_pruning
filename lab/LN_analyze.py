# Re-run complete script after kernel reset
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm

# === Simulate Input Data ===
B, N, d = 20, 16, 5
x = torch.randn(B, N, d) * torch.linspace(0.5, 2.0, d)
x_flat = x.reshape(-1, d)

# Apply distribution shift
torch.manual_seed(42)
channel_scale = torch.rand(d) * 1.5 + 0.5
channel_shift = torch.randn(d) * 1.0
x_shifted = x * channel_scale + channel_shift
x_shifted_flat = x_shifted.reshape(-1, d)

# Apply LayerNorm
x_ln = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)
x_ln_flat = x_ln.reshape(-1, d)

x_shifted_ln = (x_shifted - x_shifted.mean(dim=-1, keepdim=True)) / (
    x_shifted.std(dim=-1, keepdim=True) + 1e-5
)
x_shifted_ln_flat = x_shifted_ln.reshape(-1, d)

# Stats
running_mean = x_flat.mean(dim=0).numpy()
running_var = x_flat.var(dim=0).numpy()
shifted_mean = x_shifted_flat.mean(dim=0).numpy()
shifted_var = x_shifted_flat.var(dim=0).numpy()
updated_mean = x_ln_flat.mean(dim=0).numpy()
updated_var = x_ln_flat.var(dim=0).numpy()
shifted_ln_mean = x_shifted_ln_flat.mean(dim=0).numpy()
shifted_ln_var = x_shifted_ln_flat.var(dim=0).numpy()

# Parameters
channels = [0, 1, 2, 3, 4]
num_channels = len(channels)
colors_original = cm.Blues(np.linspace(0.4, 0.9, num_channels))
colors_shifted = cm.Oranges(np.linspace(0.4, 0.9, num_channels))
colors_normed = cm.Greens(np.linspace(0.4, 0.9, num_channels))
gaussian_color = "black"
ridge_height = 1.0
num_std = 1
alpha_hist = 0.6
fig_height_inches = 6
fig_width_inches = 5

# Shared axis limits
all_vals = torch.cat(
    [x_flat, x_shifted_flat, x_ln_flat, x_shifted_ln_flat], dim=0
).numpy()
x_min = all_vals.min() - 0.1
x_max = all_vals.max() + 0.1
y_min = 0
y_max = num_channels * (fig_height_inches / num_channels) + ridge_height * 1.2


# Plotting function
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
    y_min=None,
    y_max=None,
):
    running_stds = np.sqrt(vars)
    positions = np.arange(num_channels)[::-1] * (fig_height_inches / num_channels)
    positions -= positions[-1]

    for idx, i in enumerate(channels):
        values = input_data[:, i].flatten().numpy()
        y_base = positions[idx]

        hist_vals, bin_edges = np.histogram(values, bins=100, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist_vals *= ridge_height / hist_vals.max()
        hist_vals_offset = hist_vals + y_base

        mean_bn = means[i]
        std_bn = running_stds[i]
        x_gauss_bn = np.linspace(
            mean_bn - num_std * std_bn * 3, mean_bn + num_std * std_bn * 3, 300
        )
        g_y_bn = norm.pdf(x_gauss_bn, loc=mean_bn, scale=std_bn)
        g_y_bn *= ridge_height / g_y_bn.max()
        g_y_bn_offset = g_y_bn + y_base

        mean_data = values.mean()
        std_data = values.std()
        x_gauss_data = np.linspace(
            mean_data - num_std * std_data * 3, mean_data + num_std * std_data * 3, 300
        )
        g_y_data = norm.pdf(x_gauss_data, loc=mean_data, scale=std_data)
        g_y_data *= ridge_height / g_y_data.max()
        g_y_data_offset = g_y_data + y_base

        ax.fill_between(
            bin_centers,
            y_base,
            hist_vals_offset,
            color=colors[idx],
            alpha=alpha_hist,
            zorder=i,
        )
        ax.plot(
            x_gauss_bn,
            g_y_bn_offset,
            color=gaussian_color,
            linewidth=1.0,
            linestyle="--",
            alpha=alpha_hist,
            zorder=i,
        )
        ax.plot(
            x_gauss_data,
            g_y_data_offset,
            color=colors[idx],
            linewidth=1.0,
            linestyle="-",
            alpha=alpha_hist,
            zorder=i,
        )
        ax.plot(
            [x_min, x_max], [y_base, y_base], color=colors[idx], linewidth=1.5, zorder=1
        )

        if show_labels:
            ax.text(
                x_min,
                y_base,
                f"{label_prefix}ch {i + 1}",
                va="center",
                ha="right",
                fontsize=12,
                zorder=4,
            )

    ax.set_yticks([])
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(x_label, fontsize=12)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)


# Final plot
fig, axes = plt.subplots(2, 2, figsize=(fig_width_inches * 2.5, fig_height_inches * 2))
plot_histograms(
    axes[0, 0],
    x_flat,
    running_mean,
    running_var,
    colors_original,
    "black",
    show_labels=True,
    x_label="Original",
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
)
axes[0, 0].set_title("Original Data")

plot_histograms(
    axes[0, 1],
    x_shifted_flat,
    shifted_mean,
    shifted_var,
    colors_shifted,
    "orange",
    show_labels=False,
    x_label="Shifted",
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
)
axes[0, 1].set_title("Shifted Data")

plot_histograms(
    axes[1, 0],
    x_ln_flat,
    updated_mean,
    updated_var,
    colors_normed,
    "green",
    show_labels=True,
    x_label="Normed Original",
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
)
axes[1, 0].set_title("LayerNorm on Original")

plot_histograms(
    axes[1, 1],
    x_shifted_ln_flat,
    shifted_ln_mean,
    shifted_ln_var,
    colors_normed,
    "green",
    show_labels=False,
    x_label="Normed Shifted",
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
)
axes[1, 1].set_title("LayerNorm on Shifted")

plt.tight_layout()
plt.show()
