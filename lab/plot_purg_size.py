import matplotlib.pyplot as plt

# Data
purge_sizes = list(range(48))
top1_accuracies = [
    17.384, 17.556, 18.933, 17.212, 18.761, 19.277, 17.384, 18.589, 20.31, 20.482, 
    20.482, 19.449, 19.966, 19.621, 18.417, 19.277, 22.892, 19.105, 19.966, 21.859, 
    20.482, 20.482, 22.892, 20.482, 21.343, 21.687, 22.547, 21.687, 19.449, 22.375, 
    22.547, 22.031, 23.064, 22.203, 22.375, 24.441, 23.236, 23.58, 22.375, 23.752, 
    23.752, 23.58, 24.441, 23.064, 24.269, 23.924, 25.129, 25.645
]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(purge_sizes, top1_accuracies, marker='o', linestyle='-', color='b', label='Top1 Accuracy')

# Highlight area from 0 to 32
plt.axvspan(0, 32, color='yellow', alpha=0.3)

# Find max value in the range 0 to 32
max_value = max(top1_accuracies[:33])
max_index = top1_accuracies.index(max_value)

# Draw dashed lines only from the intersection to the axes
plt.axhline(y=max_value, linestyle='--', color='green', label='Max ACC [0-32]')  # From intersection to y-axis
plt.axvline(x=max_index, linestyle='--', color='green')  # From intersection to x-axis

# Draw reported result line at 22.89
reported_result = 22.89
plt.axhline(y=reported_result, color='orange', linestyle='--', label='Our Result (22.89)')

# Limit vertical range to the values range
# plt.ylim(min(top1_accuracies), max(top1_accuracies))

# Labels and Title
plt.xlabel('Purge Size Index')
plt.ylabel('Top1 Accuracy')
# plt.title('Top1 Accuracy vs. Purge Size')
plt.legend()
plt.grid(True)

# Show plot
plt.savefig('plot_purge_size.png')
