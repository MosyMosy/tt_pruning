import os
import re

# Directory containing the result files
corruption_name = "background_7"
directory = "results_final_tta/prototype_purge/" + corruption_name

top1_accuracies = []
entropy_values = []

# Compile a regex pattern for matching filenames
pattern_template = r"^test_purge_size_{i}_.*\.txt$"

# Loop through all possible values of ii
for i in range(128):  # 0 to 126
    pattern = re.compile(pattern_template.format(i=i))
    for filename in os.listdir(directory):
        if pattern.match(filename):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                lines = f.read().strip().splitlines()
                if len(lines) >= 2:
                    try:
                        acc = float(lines[-2])
                        entropy = float(lines[-1])
                        top1_accuracies.append(acc)
                        entropy_values.append(entropy)
                    except ValueError:
                        print(f"Warning: Could not parse floats in {filename}")
                else:
                    print(f"Warning: Not enough lines in {filename}")
            break  # Only read one file per ii

# Print results
print("top1_accuracies = [")
for val in top1_accuracies:
    print(f"    {val},")
print("]")

print("\nentropy_values = [")
for val in entropy_values:
    print(f"    {val},")
print("]")
