import os


def read_results_by_layer(dataset_name, name_suffix, directory="."):
    all_rows = []
    if len(name_suffix) > 0:
        name_suffix = name_suffix + "_"
    for layer in range(11, -1, -1):
        row = [f"Layer {layer:02d} - {name_suffix[:-1]}"]
        for filename in os.listdir(directory):
            if filename.startswith(
                f"layer-{layer}_{name_suffix}{dataset_name}_"
            ) and filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r") as file:
                    content = file.read()
                    if "--Start--" in content and "--End--" in content:
                        start = content.split("--Start--")[1].split("--End--")[0]
                        values = [line.strip() for line in start.strip().splitlines()]
                        row.extend(values)
                break
        all_rows.append(row)
    return all_rows


# Example usage
name_suffix = "BN_justattention" # "BN_justattention" "BN" "", "justattention"
dataset = "shapenetcore"  # 'scanobject' 'modelnet' 'shapenetcore'
result_path = "results_final_tta/tta_layer_prune/"
rows = read_results_by_layer(dataset, name_suffix, result_path)

# ✅ Use real tabs between elements
with open("results.tsv", "w", encoding="utf-8") as f:
    for row in rows:
        f.write("\t".join(row) + "\n")
