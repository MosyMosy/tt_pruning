import torch
import numpy as np
import open3d as o3d
import os
from pytorch3d.ops.points_alignment import (
    corresponding_points_alignment,
    iterative_closest_point,
)
import torch.nn as nn


def weighted_svd_align(src, tgt, score, eps=1e-6):
    """
    Args:
        src: (B, N, 3) source point clouds
        tgt: (B, N, 3) target point clouds
        score: (B, N) matching score (weights)
    Returns:
        src_aligned: (B, N, 3) aligned source point clouds
        tgt: (B, N, 3) target point clouds (unchanged)
    """
    B, N, _ = src.shape

    # Normalize scores (optional but safer)
    weight = score / (score.sum(dim=1, keepdim=True) + eps)  # (B, N)

    # Compute weighted centroids
    src_centroid = (src * weight.unsqueeze(-1)).sum(dim=1)  # (B, 3)
    tgt_centroid = (tgt * weight.unsqueeze(-1)).sum(dim=1)  # (B, 3)

    # Center the point clouds
    src_centered = src - src_centroid.unsqueeze(1)  # (B, N, 3)
    tgt_centered = tgt - tgt_centroid.unsqueeze(1)  # (B, N, 3)

    # Compute weighted covariance matrix
    H = torch.einsum("bni,bnj,bn->bij", src_centered, tgt_centered, weight)  # (B, 3, 3)

    # SVD
    U, S, Vh = torch.linalg.svd(H)  # U: (B,3,3), S: (B,3), Vh: (B,3,3)

    # Compute rotation
    R = torch.matmul(Vh, U.transpose(-2, -1))  # (B, 3, 3)

    # Handle reflection case
    det = torch.linalg.det(R)
    mask = (det < 0).float().view(B, 1, 1)
    Vh_corrected = Vh.clone()
    Vh_corrected[:, :, -1] *= 1 - 2 * mask.squeeze(-1)  # flip last column if necessary
    R = torch.matmul(Vh_corrected, U.transpose(-2, -1))

    # Compute translation
    t = tgt_centroid - torch.matmul(
        src_centroid.unsqueeze(1), R.transpose(-2, -1)
    ).squeeze(1)  # (B, 3)

    return R, t


def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
    src_centroid=None,
    ref_centroid=None,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(
        torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights
    )
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    if src_centroid is None:
        src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    elif len(src_centroid.size()) == 2:
        src_centroid = src_centroid.unsqueeze(1)
    src_points_centered = src_points - src_centroid  # (B, N, 3)

    if ref_centroid is None:
        ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    elif len(ref_centroid.size()) == 2:
        ref_centroid = ref_centroid.unsqueeze(1)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    U, _, V = torch.svd(H)
    Ut, V = U.transpose(1, 2), V
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(src_points.device)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


class WeightedProcrustes(nn.Module):
    def __init__(self, weight_thresh=0.5, eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def forward(
        self, src_points, tgt_points, weights=None, src_centroid=None, ref_centroid=None
    ):
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
            src_centroid=src_centroid,
            ref_centroid=ref_centroid,
        )


svd_implementation = ["ours", "pytorch3d", "sam6d"]
k_score = [-1, 10]

use_weight = [True, False]

matchings = torch.load("/home/moslem/Downloads/matchings.pt")
k = 10  # Number of distinct matchings


points1 = matchings["pts1"][10824:10851]  # (B, N, 3)
points2 = matchings["pred_pts"][10824:10851]  # (B, N, 3)
scores = matchings["assignment_score"][10824:10851]  # (B, N)

sorted_indices = torch.argsort(scores, dim=-1, descending=True)

points1 = points1[torch.arange(points1.size(0)).unsqueeze(-1), sorted_indices]
points2 = points2[torch.arange(points2.size(0)).unsqueeze(-1), sorted_indices]
scores = scores[torch.arange(scores.size(0)).unsqueeze(-1), sorted_indices]


def match_and_save(
    points1,
    points2,
    scores,
    k_score=k_score,
    svd_implementation=svd_implementation,
    use_weight=use_weight,
):
    top_text = str(k_score)
    if k_score == -1:
        k_score = points1.shape[1]
        top_text = "all"

    if not use_weight:
        scores = torch.ones_like(scores)  # (B, N)

    if svd_implementation == "ours":
        R, t = weighted_svd_align(
            points2[:, :k_score, :], points1[:, :k_score, :], scores[:, :k_score]
        )

    elif svd_implementation == "pytorch3d":
        st = corresponding_points_alignment(
            points2[:, :k_score, :],
            points1[:, :k_score, :],
            weights=scores[:, :k_score],
        )
        R, t = st.R.transpose(1, 2), st.T

    elif svd_implementation == "sam6d":
        WSVD = WeightedProcrustes(weight_thresh=0.0)
        R, t = WSVD(
            points2[:, :k_score, :],
            points1[:, :k_score, :],
            weights=scores[:, :k_score],
        )
        
    # Apply transform
    src_aligned = torch.matmul(points2, R.transpose(-2, -1)) + t.unsqueeze(
        1
    )  # (B, N, 3)

    out_dir = f"lab/svd_matching/{svd_implementation}_{top_text}_{'weighted' if use_weight else 'unweighted'}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(8, 14):
        np.savetxt(
            f"{out_dir}/{i}_1_points_{i}.xyz", points1[i].cpu().numpy()
        )  # Save the original points
        # np.savetxt(
        #     f"{out_dir}/2_{i}_points_{i}.xyz", points2[i]
        # )  # Save the original points
        np.savetxt(
            f"{out_dir}/{i}_2_points_aligned_{i}.xyz", src_aligned[i].cpu().numpy()
        )  # Save the aligned points


for implementation in svd_implementation:
    for use_w in use_weight:
        for k in k_score:
            match_and_save(
                points1,
                points2,
                scores,
                k_score=k,
                svd_implementation=implementation,
                use_weight=use_w,
            )
