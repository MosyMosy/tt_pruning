# Re-importing required library after execution state reset
import matplotlib.pyplot as plt

# Data

# top1_accuracies = [
#     17.384, 17.556, 18.933, 17.212, 18.761, 19.277, 17.384, 18.589, 20.31, 20.482,
#     20.482, 19.449, 19.966, 19.621, 18.417, 19.277, 22.892, 19.105, 19.966, 21.859,
#     20.482, 20.482, 22.892, 20.482, 21.343, 21.687, 22.547, 21.687, 19.449, 22.375,
#     22.547, 22.031, 23.064, 22.203, 22.375, 24.441, 23.236, 23.58, 22.375, 23.752,
#     23.752, 23.58, 24.441, 23.064, 24.269, 23.924, 25.129, 25.645
# ]

# # Additional entropy data
# entropy_values = [
#     0.775, 0.737, 0.759, 0.745, 0.763, 0.766, 0.745, 0.763, 0.713, 0.714,
#     0.74, 0.725, 0.733, 0.729, 0.729, 0.761, 0.744, 0.735, 0.73, 0.72,
#     0.697, 0.749, 0.709, 0.729, 0.732, 0.734, 0.688, 0.747, 0.717, 0.739,
#     0.731, 0.742, 0.699, 0.716, 0.72, 0.716, 0.724, 0.7, 0.69, 0.705,
#     0.7, 0.658, 0.673, 0.677, 0.653, 0.694, 0.627, 0.674
# ]


top1_accuracies = [
    16.867,
    45.439,
    19.621,
    20.998,
    22.892,
    24.785,
    28.227,
    30.12,
    34.596,
    43.029,
    45.439,
    48.193,
    46.816,
    20.31,
    19.793,
    18.933,
    19.449,
    20.31,
    20.482,
    19.966,
    18.417,
    20.138,
    20.138,
    21.859,
    19.621,
    20.654,
    20.826,
    21.515,
    19.793,
    19.105,
    20.998,
    20.654,
    21.687,
    21.17,
    20.998,
    20.998,
    20.31,
    21.859,
    22.892,
    21.859,
    23.064,
    21.515,
    23.064,
    21.17,
    22.892,
    23.924,
    23.924,
    23.58,
    21.17,
    23.064,
    24.269,
    22.375,
    25.129,
    24.096,
    23.064,
    22.892,
    23.58,
    25.473,
    24.785,
    24.613,
    24.269,
    25.818,
    25.473,
    27.022,
    25.645,
    26.85,
    25.645,
    26.678,
    28.227,
    27.367,
    28.227,
    29.948,
    30.12,
    30.293,
    32.014,
    28.227,
    30.293,
    31.153,
    32.702,
    31.325,
    33.219,
    34.596,
    36.833,
    34.94,
    34.251,
    36.489,
    37.866,
    37.694,
    38.382,
    38.038,
    39.759,
    40.103,
    40.448,
    41.997,
    43.546,
    44.578,
    43.89,
    44.406,
    44.406,
    43.029,
    45.955,
    47.676,
    45.439,
    45.955,
    44.923,
    48.193,
    49.742,
    48.881,
    47.332,
    46.816,
    48.193,
    47.676,
    49.053,
    47.849,
    48.881,
    46.816,
    46.472,
    49.053,
    48.537,
    47.504,
    46.816,
    48.881,
    44.923,
    46.816,
    46.816,
    46.299,
    44.75,
]

entropy_values = [
    0.781,
    0.017,
    0.383,
    0.292,
    0.228,
    0.147,
    0.112,
    0.088,
    0.057,
    0.023,
    0.017,
    0.016,
    0.01,
    0.471,
    0.476,
    0.47,
    0.467,
    0.443,
    0.417,
    0.409,
    0.397,
    0.386,
    0.393,
    0.372,
    0.383,
    0.373,
    0.362,
    0.36,
    0.325,
    0.349,
    0.325,
    0.318,
    0.297,
    0.31,
    0.292,
    0.278,
    0.285,
    0.273,
    0.273,
    0.266,
    0.244,
    0.245,
    0.228,
    0.231,
    0.228,
    0.221,
    0.216,
    0.199,
    0.215,
    0.183,
    0.177,
    0.179,
    0.181,
    0.158,
    0.164,
    0.172,
    0.162,
    0.156,
    0.147,
    0.142,
    0.147,
    0.133,
    0.127,
    0.127,
    0.118,
    0.124,
    0.106,
    0.109,
    0.112,
    0.11,
    0.102,
    0.101,
    0.088,
    0.089,
    0.078,
    0.074,
    0.072,
    0.074,
    0.075,
    0.077,
    0.068,
    0.057,
    0.056,
    0.06,
    0.05,
    0.056,
    0.051,
    0.046,
    0.042,
    0.035,
    0.039,
    0.037,
    0.036,
    0.031,
    0.03,
    0.025,
    0.021,
    0.029,
    0.023,
    0.023,
    0.022,
    0.021,
    0.017,
    0.022,
    0.019,
    0.019,
    0.016,
    0.016,
    0.016,
    0.017,
    0.016,
    0.014,
    0.018,
    0.014,
    0.014,
    0.012,
    0.012,
    0.012,
    0.011,
    0.012,
    0.01,
    0.01,
    0.01,
    0.007,
    0.01,
    0.009,
    0.007,
]


# top1_accuracies = top1_accuracies[:48]
# entropy_values = entropy_values[:48]

purge_sizes = list(range(len(top1_accuracies)))
# Font size settings
axis_label_fontsize = 18
axis_number_fontsize = 14
legend_fontsize = 18

# Create figure and primary axis with tight layout to remove margins
fig, ax1 = plt.subplots(figsize=(12, 7), tight_layout=True)

# Plot accuracy on primary axis
ax1.plot(
    purge_sizes,
    top1_accuracies,
    marker="o",
    linestyle="-",
    color="black",
    label="Top1 ACC",
)

# Highlight area from 0 to 32
ax1.axvspan(0, 32, color="yellow", alpha=0.3)

# Find max accuracy value in the range 0 to 32
max_value = max(top1_accuracies[:33])
max_index = top1_accuracies.index(max_value)

# Draw dashed lines only from the intersection to the axes
plt.axhline(y=max_value, color="green", linestyle="--", label="Max ACC [0-32]")
plt.axvline(x=max_index, color="green", linestyle="--")

# Draw reported result line at 22.89
ax1.axhline(y=22.89, color="orange", linestyle="--", label="Our Result (22.89)")

# Set x-axis ticks
ax1.set_xticks([0, 2, 4, 8, 16, 32, 48, 64, 96, 128])

ax1.tick_params(axis="x", labelsize=axis_number_fontsize)
ax1.tick_params(axis="y", labelsize=axis_number_fontsize)

# Labels for accuracy axis
ax1.set_xlabel("Purge Size", fontsize=axis_label_fontsize)
ax1.set_ylabel("Top1 Acc", fontsize=axis_label_fontsize)
ax1.legend(fontsize=legend_fontsize, loc="lower center", bbox_to_anchor=(0.6, 0))
ax1.grid(True)

# Create secondary axis for entropy
ax2 = ax1.twinx()
ax2.plot(purge_sizes, entropy_values, linestyle="-", color="crimson", label="Entropy")

# Labels for entropy axis
ax2.set_ylabel("Entropy", fontsize=axis_label_fontsize, color="crimson")
ax2.tick_params(axis="y", labelsize=axis_number_fontsize, colors="gray")


# Save and display plot
plt.savefig("plot_purge_size_entropy_background_48.pdf")
plt.show()
