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
    59.38,
    60.413,
    57.831,
    59.208,
    59.38,
    60.757,
    58.864,
    59.036,
    59.897,
    62.306,
    57.831,
    58.692,
    58.52,
    58.864,
    60.241,
    60.413,
    58.692,
    59.552,
    61.79,
    61.618,
    60.585,
    59.552,
    60.585,
    58.692,
    59.725,
    61.962,
    61.618,
    60.413,
    61.446,
    61.102,
    62.306,
    60.413,
    61.618,
    61.102,
    61.274,
    61.274,
    62.134,
    63.683,
    62.306,
    62.651,
    62.134,
    63.683,
    61.962,
    62.995,
    62.651,
    61.962,
    64.028,
    64.028,
    60.929,
    62.306,
    62.306,
    61.102,
    62.995,
    60.585,
    61.79,
    62.995,
    62.306,
    61.962,
    61.618,
    63.855,
    61.618,
    61.79,
    60.757,
    61.446,
    60.413,
    61.79,
    60.929,
    61.79,
    60.929,
    60.929,
    61.79,
    60.929,
    60.757,
    58.692,
    60.413,
    59.036,
    58.003,
    59.036,
    60.069,
    56.11,
    58.003,
    59.552,
    56.627,
    57.831,
    56.627,
    57.659,
    56.799,
    53.356,
    55.594,
    54.217,
    52.496,
    55.077,
    55.25,
    51.979,
    53.356,
    51.463,
    50.947,
    52.668,
    50.602,
    50.775,
    49.742,
    49.053,
    48.365,
    45.783,
    44.234,
    45.439,
    44.923,
    43.718,
    43.029,
    40.448,
    38.898,
    40.792,
    40.964,
    38.382,
    36.661,
    35.112,
    34.423,
    35.112,
    33.046,
    30.981,
    30.809,
    28.399,
    28.055,
    23.752,
    22.719,
    23.064,
    19.793,
]

entropy_values = [
    0.326,
    0.329,
    0.313,
    0.334,
    0.322,
    0.32,
    0.322,
    0.317,
    0.3,
    0.317,
    0.328,
    0.322,
    0.318,
    0.337,
    0.313,
    0.318,
    0.311,
    0.322,
    0.333,
    0.295,
    0.332,
    0.317,
    0.318,
    0.326,
    0.336,
    0.332,
    0.324,
    0.311,
    0.312,
    0.336,
    0.319,
    0.331,
    0.325,
    0.34,
    0.32,
    0.332,
    0.328,
    0.343,
    0.331,
    0.322,
    0.322,
    0.34,
    0.314,
    0.328,
    0.344,
    0.329,
    0.339,
    0.342,
    0.336,
    0.326,
    0.333,
    0.34,
    0.301,
    0.347,
    0.321,
    0.331,
    0.335,
    0.327,
    0.318,
    0.327,
    0.349,
    0.321,
    0.329,
    0.342,
    0.338,
    0.326,
    0.353,
    0.336,
    0.329,
    0.349,
    0.328,
    0.323,
    0.318,
    0.365,
    0.351,
    0.322,
    0.357,
    0.34,
    0.332,
    0.342,
    0.343,
    0.36,
    0.356,
    0.374,
    0.362,
    0.365,
    0.363,
    0.369,
    0.382,
    0.374,
    0.354,
    0.369,
    0.376,
    0.385,
    0.389,
    0.383,
    0.376,
    0.381,
    0.393,
    0.389,
    0.392,
    0.398,
    0.385,
    0.395,
    0.396,
    0.398,
    0.418,
    0.449,
    0.433,
    0.429,
    0.412,
    0.449,
    0.423,
    0.409,
    0.41,
    0.453,
    0.472,
    0.446,
    0.469,
    0.474,
    0.485,
    0.502,
    0.515,
    0.48,
    0.552,
    0.576,
    0.625,
]

# top1_accuracies = top1_accuracies[:48]
# entropy_values = entropy_values[:48]

purge_sizes = list(range(len(top1_accuracies)))
# Font size settings
axis_label_fontsize = 18
axis_number_fontsize = 14
legend_fontsize = 18

# Create figure and primary axis with tight layout to remove margins
fig, ax1 = plt.subplots(figsize=(16, 7), tight_layout=True)

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
ax1.axhline(y=62.48, color="orange", linestyle="--", label="Our Result (62.48)")

# Set x-axis ticks
ax1.set_xticks(
    [0, 2, 4, 8, 16, 32, 48] + list(range(64, 128, 8)) + [len(purge_sizes) - 1]
)

ax1.tick_params(axis="x", labelsize=axis_number_fontsize)
ax1.tick_params(axis="y", labelsize=axis_number_fontsize)

# Labels for accuracy axis
ax1.set_xlabel("Purge Size", fontsize=axis_label_fontsize)
ax1.set_ylabel("Top1 Acc", fontsize=axis_label_fontsize)
ax1.legend(fontsize=legend_fontsize, loc="lower center", bbox_to_anchor=(0.6, 0))
ax1.grid(True)

# Create secondary axis for entropy
ax2 = ax1.twinx()
ax2.plot(purge_sizes, entropy_values, linestyle="-", color="gray", label="Entropy")

# Labels for entropy axis
ax2.set_ylabel("Entropy", fontsize=axis_label_fontsize, color="gray")
ax2.tick_params(axis="y", labelsize=axis_number_fontsize, colors="gray")


# Save and display plot
plt.savefig("plot_purge_size_entropy_distortion_48.pdf")
plt.show()
