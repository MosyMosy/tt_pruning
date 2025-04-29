import torch
import numpy as np
import open3d as o3d
import os
import matplotlib.colors as mcolors


def generate_distinct_colors(k):
    assert k <= 16, "This function supports up to 16 distinct colors."

    color_names = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
        "lime",
        "teal",
        "pink",
        "brown",
        "gray",
        "olive",
        "navy",
        "maroon",
    ]

    rgb_colors = [mcolors.to_rgb(mcolors.CSS4_COLORS[name]) for name in color_names[:k]]
    return rgb_colors, color_names[:k]





def save_colored_point_clouds(
    idx, pc1_batch, pc2_batch, out_dir="colored_pcs", colors=None
):
    os.makedirs(out_dir, exist_ok=True)

    N, _ = pc1_batch.shape
    assert (
        colors is not None and len(colors) == N
    ), "Color list must match number of points (N)."

    # Colors: list of N RGB colors (for each point in pc1 and pc2)
    colors_array = np.array(colors)

    pc1 = pc1_batch  # Shape (N, 3)
    pc2 = pc2_batch  # Shape (N, 3)

    # Assign different colors to each point (same colors for pc1 and pc2)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd1.colors = o3d.utility.Vector3dVector(colors_array)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd2.colors = o3d.utility.Vector3dVector(colors_array)

    # Save each point cloud
    filename_1 = os.path.join(out_dir, f"1_{idx}_pair.ply")
    filename_2 = os.path.join(out_dir, f"2_{idx}_pair.ply")
    o3d.io.write_point_cloud(filename_1, pcd1)
    o3d.io.write_point_cloud(filename_2, pcd2)


matchings = torch.load("/home/moslem/Downloads/matchings.pt")
k = 10  # Number of distinct matchings
out_dir = "lab/colored_pcs"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

points1 = matchings["pts1"][10824:10851]  # (B, N, 3)
points2 = matchings["pred_pts"][10824:10851]  # (B, N, 3)
scores = matchings["assignment_score"][10824:10851]  # (B, N)

sorted_indices = torch.argsort(scores, dim=-1, descending=True)

points1 = points1[torch.arange(points1.size(0)).unsqueeze(-1), sorted_indices]
points2 = points2[torch.arange(points2.size(0)).unsqueeze(-1), sorted_indices]
scores = scores[torch.arange(scores.size(0)).unsqueeze(-1), sorted_indices]


points1 = points1.cpu().numpy()
points2 = points2.cpu().numpy()
# Generate distinct colors
colors, color_names = generate_distinct_colors(k)


with open(f"{out_dir}/scores.txt", "w") as f:
    for i in range(points1.shape[0]):
        save_colored_point_clouds(
            i, points1[i, :k, :], points2[i, :k, :], out_dir=out_dir, colors=colors
        )

        np.savetxt(f"{out_dir}/1_{i}_points.xyz", points1[i])
        np.savetxt(f"{out_dir}/2_{i}_points.xyz", points2[i])

        f.write(f"{dict(zip(color_names, scores[i]))}\n")
