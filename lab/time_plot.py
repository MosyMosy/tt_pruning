import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["PG-SP", "BFTT3D"]
min_vals = [8.122, 221.008]
max_vals = [31.84, 239.454]
mean_vals = [10.632, 230.821]
std_vals = [5.703, 4.484]

# Compute boxplot stats
q1_vals = [mean_vals[0] - std_vals[0], mean_vals[1] - std_vals[1]]  # Approximate Q1
q3_vals = [mean_vals[0] + std_vals[0], mean_vals[1] + std_vals[1]]  # Approximate Q3
median_vals = mean_vals  # Assuming mean ~ median for visualization

# Create boxplot
fig, ax = plt.subplots(figsize=(6, 20))
box_data = [np.random.normal(mean, std, 100) for mean, std in zip(mean_vals, std_vals)]
bp = ax.boxplot(box_data, vert=True, patch_artist=True, labels=methods)

# Add mean markers
for i in range(len(methods)):
    ax.scatter(i + 1, mean_vals[i], color='red', marker='+', s=100, label="Mean" if i == 0 else "")

# Labels and title
ax.set_ylabel("Time (ms)")
ax.set_title("Boxplot of Execution Time by Method")

# Show plot
plt.legend()
plt.show()
