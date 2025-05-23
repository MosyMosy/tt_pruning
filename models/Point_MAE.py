import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)
from utils.logger import *
import random

# from knn_cuda import KNN
from pytorch3d.ops import knn_points  # pytorch3d

# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pytorch3d.loss import chamfer_distance  # pytorch3d
from functools import partial

import numpy as np
from scipy.ndimage import gaussian_filter

entropy_list = []


class ChamferDistanceL1(torch.nn.Module):  # pytorch3d
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return chamfer_distance(x, y, norm=1)[0]


class ChamferDistanceL2(torch.nn.Module):  # pytorch3d
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return chamfer_distance(x, y, norm=2)[0]


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel, group_norm=False):
        super().__init__()
        self.encoder_channel = encoder_channel
        if group_norm:
            first_norm = nn.GroupNorm(8, 128)
            second_norm = nn.GroupNorm(8, 512)
        else:
            first_norm = nn.BatchNorm1d(128)
            second_norm = nn.BatchNorm1d(512)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            first_norm,
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            second_norm,
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        """
        input: B N 3    N  number of points ,  M is number of centers (number of groups )
        ---------------------------
        output: B G M 3     G is group size 32
        center : B G 3
        """
        # print(xyz.shape)
        # if len(xyz.shape) == 2:
        #     xyz = torch.unsqueeze(xyz, dim=0)

        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(
            xyz, self.num_group
        )  # B G 3    sample 128 center points from 2048 points
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center)  # B G M,   kNN samples for each center  idx (B, M, G)   every center has G (group size) NN points
        idx = knn_points(center, xyz, K=self.group_size)[1]  # pytorch3d
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )  # idx_base  (8, 1, 1)
        idx = idx + idx_base  # for  batch 0 offset 0,   batch 1 ~7,  offset  1*2048
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[
            idx, :
        ]  # (8, 2048, 3) -> (8*2048, 3)   # todo sampling the neighborhoold points for each center in each batch
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()  # (8, 128, 32, 3)  128 groups, each group has 32 points,
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_head_attentions(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = v.transpose(1, 2).reshape(B, N, C).unsqueeze(1)
        x = attn @ v

        x = x.reshape(B * self.num_heads, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_weight(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        stat_distance_q = wasserstein_distance_gaussians(
            q[:, :, 0:1, :].mean(dim=-1),
            q[:, :, 0:1, :].std(dim=-1),
            q.mean(dim=-1),
            q.std(dim=-1),
        )
        stat_distance_k = wasserstein_distance_gaussians(
            k[:, :, 0:1, :].mean(dim=-1),
            k[:, :, 0:1, :].std(dim=-1),
            k.mean(dim=-1),
            k.std(dim=-1),
        )
        stat_sim_q = torch.exp(-1 * stat_distance_q).unsqueeze(-1)
        stat_sim_k = torch.exp(-1 * stat_distance_k).unsqueeze(-1)
        # stat_sim_q = 1 / (1 + stat_distance_q).unsqueeze(-1)
        # stat_sim_k = 1 / (1 + stat_distance_k).unsqueeze(-1)
        stat_sim_attn = stat_sim_q @ stat_sim_k.transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) * stat_sim_attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_entropy_weight(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # #################################################################

        H = attn.shape[1]  # number of heads
        diag_mask = (
            torch.eye(N - 1, dtype=torch.bool, device=attn.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, N-1, N-1)
        attn_no_diag = attn[:, :, 1:, 1:].masked_fill(diag_mask, 0.0)
        attn_no_diag = attn_no_diag / attn_no_diag.sum(dim=-1, keepdim=True)
        attn_entropy = -(attn_no_diag * attn_no_diag.clamp(min=1e-8).log()).sum(dim=-1)
        # to make the entropy value between 0 and 1
        attn_entropy = attn_entropy / math.log(N - 2)
        # entropy_list.append([attn_entropy[:,:,0].mean().cpu().detach(), attn_entropy[:,:,1:].mean().cpu().detach()])
        attn_entropy = attn_entropy.mean(dim=1)
        attn_score = torch.softmax(-attn_entropy, dim=-1)

        attn[:, :, :1, 1:] = (
            attn[:, :, :1, 1:] * attn_score[:, None, None, :]
        )  # multiply the row wise scores to the columns
        # #####################################################################

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_token_mask(self, x, entropy_threshold=0.5):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Replace random noise to high entropy tokens
        # first mask the diagonal of the attention matrix
        H = attn.shape[1]  # number of heads
        diag_mask = (
            ~torch.eye(N, dtype=torch.bool, device=attn.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, H, N, N)
        )
        attn_no_diag = attn  # .masked_select(diag_mask).view(B, H, N, N - 1)
        attn_no_diag = attn_no_diag.softmax(dim=-1)
        attn_entropy = -(attn_no_diag * attn_no_diag.clamp(min=1e-8).log()).sum(dim=-1)
        # to make the entropy value between 0 and 1
        attn_entropy = attn_entropy / math.log(N)
        # entropy_list.append([attn_entropy[:,:,0].mean().cpu().detach(), attn_entropy[:,:,1:].mean().cpu().detach()])
        attn_entropy = attn_entropy.mean(dim=1)  # B N mean over the head dimension
        high_entropy_mask = attn_entropy > entropy_threshold  # shape: (B, N)
        # cls_token is always low entropy
        high_entropy_mask[:, 0] = False  # cls token
        high_entropy_mask = high_entropy_mask.unsqueeze(-1)  # (B, N, 1)

        # the rest of the attention layer
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, high_entropy_mask

    def forward_no_attn(self, x):
        B, N, C = x.shape
        x_v = (self.qkv(x).reshape(B, N, 3, C))[:, :, 2, :]

        x_v = self.proj(x_v)
        x_v = self.proj_drop(x_v)
        return x_v


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_head_attentions(self, x):
        B, N, C = x.shape
        x_ = self.drop_path(self.attn.forward_head_attentions(self.norm1(x)))
        BB = x_.shape[0]
        h = BB // B
        x = x.unsqueeze(1).repeat(1, h, 1, 1).view(BB, N, C) + x_
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_weight(self, x):
        x = x + self.drop_path(self.attn.forward_weight(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_entropy_weight(self, x):
        x = x + self.drop_path(self.attn.forward_entropy_weight(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_token_mask(self, x, entropy_threshold=0.5):
        x_, entropy_mask = self.attn.forward_token_mask(
            self.norm1(x),
            entropy_threshold=entropy_threshold,
        )

        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, entropy_mask

    def forward_no_attn(self, x):
        x = x + self.drop_path(self.attn.forward_no_attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_norms_embedding(self, x):
        norms_embedding = []

        x_ = self.norm1(x)
        norms_embedding.append(x_.cpu().detach())
        x = x + self.drop_path(self.attn.forward_no_attn(x_))

        x_ = self.norm2(x)
        norms_embedding.append(x_.cpu().detach())
        x = x + self.drop_path(self.mlp(x_))
        return x, norms_embedding

    def forward_source_norm_embeddings(self, x):
        norms_embedding = []

        x_ = self.norm1(x)
        norms_embedding.append(x_.cpu().detach())
        x = x + self.drop_path(self.attn(x_))

        x_ = self.norm2(x)
        norms_embedding.append(x_.cpu().detach())
        x = x + self.drop_path(self.mlp(x_))
        return x, norms_embedding


# todo now it will return the features and feature list for part-segmentation classification head
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, list)
                        else drop_path_rate
                    ),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
        return x, None

    def forward_head_attentions(self, x, pos):
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                x = block.forward_head_attentions(x + pos)
            else:
                x = block(x + pos)
        return x, None

    def forward_intermediate_cls(self, x, pos):
        intermediate_cls = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            intermediate_cls.append(x[:, 0, :])
        return x, intermediate_cls

    def forward_out_intermediate(self, x, pos):
        intermediates = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            x_ = (x + pos)[:, 1:, :]  # remove the [CLS] token
            x_ = x_ - x_.mean(dim=(2), keepdim=True)
            x_ = x_ / (x_.std(dim=(2), keepdim=True) + 1e-6)
            intermediates.append(x_.flatten(0, 1).detach().cpu())
        return intermediates

    def forward_analyze(self, x, pos, source_stats=None, threshold_percentile=20):
        for i, block in enumerate(self.blocks):
            if source_stats is not None:
                if i in [0]:
                    surce_mean = source_stats[0][i][None, None, :]
                    surce_std = source_stats[1][i][None, None, :]

                    x_ = x[:, 1:] + pos[:, 1:]  # No CLS Token
                    x_ = x_ - x_.mean(dim=(2), keepdim=True)
                    x_ = x_ / (x_.std(dim=(2), keepdim=True) + 1e-6)

                    x_distance = misc.mahalanobis_distance(x_, surce_mean, surce_std)
                    # z_score = (x_distance - x_distance.mean()) / x_distance.std()
                    dist_min = x_distance.min()
                    dist_max = x_distance.max()
                    dist_mean = x_distance.mean()
                    threshold = dist_min + (dist_max - dist_min) * threshold_percentile
                    x_mask = x_distance <= threshold
                    # x_mask = x_distance <= 1.2
                    # remove the tokens from x based on the mask
                    masked_x = x[:, 1:][0][x_mask[0]].unsqueeze(0)

                    x = torch.cat([x[:, 0:1], masked_x], dim=1)
                    pos = torch.cat(
                        [pos[:, 0:1], pos[:, 1:][0][x_mask[0]].unsqueeze(0)], dim=1
                    )
                    # x = x * x_mask

            x = block(x + pos)

        return x, (dist_min, dist_max, dist_mean, threshold)

    def forward_prototype_purge(self, x, pos, source_stats, layer_idx, purge_size=16):
        for i, block in enumerate(self.blocks):
            if i in [0]:
                B, N, C = x.shape
                source_mean = source_stats[0][i][None, None, :].to(x.dtype)
                source_std = source_stats[1][i][None, None, :].to(x.dtype)

                x_ = x[:, 1:] + pos[:, 1:]  # No CLS Token
                x_ = x_ - x_.mean(dim=(2), keepdim=True)
                x_ = x_ / (x_.std(dim=(2), keepdim=True) + 1e-6)

                # sub_vector = misc.compute_coral_subspace(source_mean, source_std, 200)
                # x_distance = misc.mahalanobis_distance_subspace(x_, source_mean, source_std, sub_vector)

                x_distance = misc.mahalanobis_distance(x_, source_mean, source_std)
                # z_score = (x_distance - x_distance.mean()) / x_distance.std()
                threshold_max = misc.best_threshold_model(x_distance)
                x_mask_max = x_distance <= max(threshold_max, x_distance.min() + 2)
                to_mask_count = (
                    purge_size  # (~x_mask_max).sum(dim=-1).float().mean().int()
                )
                to_keep_indices = x_distance.argsort(dim=-1)[:, : N - 1 - to_mask_count]
                to_keep_indices = to_keep_indices.sort(dim=-1)[0]
                to_keep_indices = to_keep_indices.unsqueeze(-1).expand(-1, -1, C)

                masked_x = torch.gather(x[:, 1:], 1, to_keep_indices)
                masked_pose = torch.gather(pos[:, 1:], 1, to_keep_indices)

                x = torch.cat([x[:, 0:1], masked_x], dim=1)
                pos = torch.cat([pos[:, 0:1], masked_pose], dim=1)

            x = block(x + pos)
        return x

    def forward_cls_purge(self, x, pos, purge_size=16):
        for i, block in enumerate(self.blocks):
            if i in [0]:
                B, N, C = x.shape
                cls_token = x[:, 0:1]
                pos_cls = pos[:, 0:1]

                cls_query = (
                    block.attn.qkv(block.norm1(cls_token + pos_cls))
                    .reshape(B, 1, 3, C)
                    .permute(2, 0, 1, 3)
                )[0]

                token = x[:, 1:]
                pos_token = pos[:, 1:]

                token_key = (
                    block.attn.qkv(block.norm1(token + pos_token))
                    .reshape(B, N - 1, 3, C)
                    .permute(2, 0, 1, 3)
                )[1]

                cosine_distance = misc.cosine_distance(cls_query, token_key)

                to_keep_indices = cosine_distance.argsort(dim=-1)[
                    :, : N - 1 - purge_size
                ]
                to_keep_indices = to_keep_indices.sort(dim=-1)[0]
                to_keep_indices = to_keep_indices.unsqueeze(-1).expand(-1, -1, C)

                masked_x = torch.gather(x[:, 1:], 1, to_keep_indices)
                masked_pose = torch.gather(pos[:, 1:], 1, to_keep_indices)

                x = torch.cat([x[:, 0:1], masked_x], dim=1)
                pos = torch.cat([pos[:, 0:1], masked_pose], dim=1)

            x = block(x + pos)

        return x

    def forward_token_mask(self, x, pos, entropy_threshold=0.5):
        entropy_mask = None
        for i, block in enumerate(self.blocks):
            if i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                B, N, C = x.shape

                qkv = (
                    block.attn.qkv(block.norm1(x + pos))
                    .reshape(B, N, 3, 6, C // 6)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = (
                    qkv[0],
                    qkv[1],
                    qkv[2],
                )  # make torchscript happy (cannot use tensor as tuple)

                attn = (q @ k.transpose(-2, -1)) * 64**-0.5
                attn = attn.softmax(dim=-1)
                attn_entropy = -(attn * attn.clamp(min=1e-8).log()).sum(dim=-1)
                # to make the entropy value between 0 and 1
                attn_entropy = attn_entropy / math.log(N)
                attn_entropy = attn_entropy.mean(dim=1)

                attn_entropy[:, 0] = 0

                # Add random noise to high entropy tokens
                entropy_mask = (attn_entropy > entropy_threshold).unsqueeze(
                    -1
                )  # shape: (B, N)
                noise = torch.randn_like(x)
                x = torch.where(entropy_mask, noise, x + pos)  # (B, N, C)

                # filter x using the entropy mask
                # x = x[~entropy_mask.squeeze(-1)].reshape(B, -1, C)
                # pos = pos[~entropy_mask.squeeze(-1)].reshape(B, -1, C)
                # x = x + pos

                x = block(x)
            else:
                x = block(x + pos)

        return x, entropy_mask

    def forward_intermediate(self, x, pos):
        intermediate = []
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            intermediate.append([i + 1, x])
        return x, intermediate

    def forward_no_attn(self, x, pos, layer_list):
        for i in layer_list:
            if i in layer_list:
                x = self.blocks[i].forward_no_attn(x + pos)

        return x

    def forward_layer_prune(self, x, pos, layer_list, purne_attention=False):
        for i, block in enumerate(self.blocks):
            if i in layer_list:
                if purne_attention:
                    x = block.forward_no_attn(x + pos)
                else:
                    continue
            else:
                x = block(x + pos)
        return x

    def forward_norms_embedding(self, x, pos):
        norms_embedding_list = []
        intermediate_cls_list = []
        for i, block in enumerate(self.blocks):
            x, norms_embedding = block.forward_norms_embedding(x + pos)
            norms_embedding_list += norms_embedding
            intermediate_cls_list += x[:, 0, :]
        return x, norms_embedding_list, intermediate_cls_list

    def forward_source_norm_embeddings(self, x, pos):
        norms_embedding_list = []
        for i, block in enumerate(self.blocks):
            x, norms_embedding = block.forward_source_norm_embeddings(x + pos)
            norms_embedding_list += norms_embedding
        return x, norms_embedding_list


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        last_dim = self.trans_dim * 2
        class_blocks = []
        for cls_block in range(0, self.num_hid_cls_layers):
            class_blocks.extend(
                (
                    nn.Linear(last_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                )
            )
            last_dim = 256
        self.class_head = nn.Sequential(
            *class_blocks, nn.Linear(last_dim, self.cls_dim)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
            }

            base_ckpt = {
                k.replace("class_head.8.custom_last_layer_name", "class_head.8"): v
                for k, v in base_ckpt.items()
            }

            # delete the cls head in case it had a different number of classes
            to_delete_prefix = "class_head.8"
            # Check if the key exists and its shape meets the condition
            if f"{to_delete_prefix}.weight" in base_ckpt:
                shape = base_ckpt[f"{to_delete_prefix}.weight"].shape  # Get the shape
                # Replace `x` with the shape condition you want to check
                if shape[0] != self.cls_dim:
                    # Delete all keys that start with "class_head.8"
                    keys_to_delete = [
                        k
                        for k in list(base_ckpt.keys())
                        if k.startswith(to_delete_prefix)
                    ]
                    for k in keys_to_delete:
                        del base_ckpt[k]

            for k in list(base_ckpt.keys()):
                if k.startswith("MAE_encoder"):
                    base_ckpt[k[len("MAE_encoder.") :]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith("base_model"):
                    base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger="Transformer",
                )
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger="Transformer",
                )

            print_log(
                f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                logger="Transformer",
            )
        else:
            print_log("Training from scratch!!!", logger="Transformer")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)[0]
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head(concat_f)
        return ret

    def forward_last_hidden(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)[0]
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head[0](concat_f)
        ret = self.class_head[1](ret)
        ret = self.class_head[2](ret)
        ret = self.class_head[3](ret)
        ret = self.class_head[4](ret)
        ret = self.class_head[5](ret)
        ret = self.class_head[6](ret)
        ret = self.class_head[7](ret)
        last_hidden = ret
        ret = self.class_head[8](ret)
        return ret, last_hidden, self.class_head[8].weight, self.class_head[8].bias

    def forward_intermediate_cls(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x, intermediate_cls = self.blocks.forward_intermediate_cls(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head(concat_f)
        return ret, intermediate_cls

    def forward_out_intermediate(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        intermediates = self.blocks.forward_out_intermediate(x, pos)

        return intermediates

    def forward_analyze(self, pts, source_stats=None, threshold_percentile=20):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x, (dist_min, dist_max, dist_mean, threshold) = self.blocks.forward_analyze(
            x, pos, source_stats, threshold_percentile=threshold_percentile
        )
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head(concat_f)

        return ret, (dist_min, dist_max, dist_mean, threshold)

    def forward_prototype_purge(self, pts, source_stats, layer_idx=None, purge_size=16):
        if layer_idx is None:
            layer_idx = range(self.depth)
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks.forward_prototype_purge(
            x, pos, source_stats, layer_idx, purge_size=purge_size
        )
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head(concat_f)

        return ret

    def forward_cls_purge(self, pts, purge_size=16):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks.forward_cls_purge(x, pos, purge_size=purge_size)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head(concat_f)

        return ret

    def forward_token_mask(self, pts, entropy_threshold=0.5):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer

        x, entropy_mask = self.blocks.forward_token_mask(
            x, pos, entropy_threshold=entropy_threshold
        )
        x = self.norm(x)
        # x = x * (entropy_mask * (-1e6) + 1)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head(concat_f)

        # entropy_list = torch.FloatTensor(entropy_list)
        return ret

    def forward_intermediate_dual(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x, intermediates = self.blocks.forward_intermediate(x, pos)
        projected_intermediates = []
        for inte in intermediates:
            projected_intermediates.append(
                self.blocks.forward_no_attn(
                    inte[1], pos, layer_list=range(inte[0], self.depth)
                )
            )
        del intermediates
        projected_intermediates = torch.stack(projected_intermediates, dim=0)
        projected_intermediates = self.norm(projected_intermediates)

        x = projected_intermediates[-1]
        mean = projected_intermediates.mean(dim=0)

        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        concat_mean = torch.cat([mean[:, 0], mean[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head(concat_f)
        ret_mean = self.class_head(concat_mean)

        return ret, ret_mean

    def forward_intermediate(self, pts, layer_idx=-1):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x, intermediates = self.blocks.forward_intermediate(x, pos)
        projected_intermediates = []
        for inte in intermediates:
            projected_intermediates.append(
                self.blocks.forward_no_attn(
                    inte[1], pos, layer_list=range(inte[0], self.depth)
                )
            )
        del intermediates
        projected_intermediates = torch.stack(projected_intermediates, dim=0)
        projected_intermediates = self.norm(projected_intermediates)

        # --------------------- average_0_11 -------------------
        # x = (projected_intermediates[0] + projected_intermediates[-1]) / 2
        # x = projected_intermediates.mean(dim=0)
        # x = projected_intermediates[-1] - projected_intermediates.min(dim=0)[0]
        x = projected_intermediates[layer_idx]

        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head(concat_f)

        # --------------------- averag_0-11_logit -------------------
        # x = projected_intermediates
        # x = torch.cat([x[0:1], x[-2:-1]], dim=0)
        # L, B, N, C = x.shape
        # x = x.view(L * B, N, C)
        # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        # ret = self.class_head(concat_f)
        # ret = ret.view(L, B, -1)
        # ret = ret.mean(dim=0)
        return ret

    def forward_headless(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)[0]
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        # ret = self.class_head(concat_f)
        return concat_f

    def forward_layer_prune(self, pts, layer_list, purne_attention=False):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks.forward_layer_prune(x, pos, layer_list, purne_attention)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head(concat_f)
        return ret

    def forward_norms_embedding(self):  # (self, pts):
        # neighborhood, center = self.group_divider(pts)
        # group_input_tokens = self.encoder(neighborhood)  # B G N
        # cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        # cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        # pos = self.pos_embed(center)

        # x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        # pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x, norms_embedding, intermediate_cls_list = self.blocks.forward_norms_embedding(
            self.cls_token, self.cls_pos
        )
        x = self.norm(x)
        norms_embedding.append(x.cpu().detach())
        # concat_f = torch.cat([x[:, 0], x[:, 0]], dim=-1)

        # head_embbeding = self.class_head[0](concat_f)
        # head_embbeding = self.class_head[1](head_embbeding)
        # norms_embedding.append(head_embbeding.cpu().detach())
        # head_embbeding = self.class_head[2](head_embbeding)
        # head_embbeding = self.class_head[3](head_embbeding)
        # head_embbeding = self.class_head[4](head_embbeding)
        # head_embbeding = self.class_head[5](head_embbeding)
        # norms_embedding.append(head_embbeding.cpu().detach())
        # head_embbeding = self.class_head[6](head_embbeding)
        # head_embbeding = self.class_head[7](head_embbeding)
        # ret = self.class_head[8](head_embbeding)
        # ret = self.class_head(concat_f)
        return norms_embedding

    def forward_source_norm_embeddings(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x, norms_embedding = self.blocks.forward_source_norm_embeddings(x, pos)
        x = self.norm(x)
        norms_embedding.append(x.cpu().detach())
        return norms_embedding


@MODELS.register_module()
class PointTransformer_cls_only(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        last_dim = self.trans_dim
        class_blocks = []
        for cls_block in range(0, self.num_hid_cls_layers):
            class_blocks.extend(
                (
                    nn.Linear(last_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                )
            )
            last_dim = 256
        self.class_head = nn.Sequential(
            *class_blocks, nn.Linear(last_dim, self.cls_dim)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
            }

            base_ckpt = {
                k.replace("class_head.8.custom_last_layer_name", "class_head.8"): v
                for k, v in base_ckpt.items()
            }

            # delete the cls head in case it had a different number of classes
            to_delete_prefix = "class_head.8"
            # Check if the key exists and its shape meets the condition
            if f"{to_delete_prefix}.weight" in base_ckpt:
                shape = base_ckpt[f"{to_delete_prefix}.weight"].shape  # Get the shape
                # Replace `x` with the shape condition you want to check
                if shape[0] != self.cls_dim:
                    # Delete all keys that start with "class_head.8"
                    keys_to_delete = [
                        k
                        for k in list(base_ckpt.keys())
                        if k.startswith(to_delete_prefix)
                    ]
                    for k in keys_to_delete:
                        del base_ckpt[k]

            for k in list(base_ckpt.keys()):
                if k.startswith("MAE_encoder"):
                    base_ckpt[k[len("MAE_encoder.") :]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith("base_model"):
                    base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger="Transformer",
                )
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger="Transformer",
                )

            print_log(
                f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                logger="Transformer",
            )
        else:
            print_log("Training from scratch!!!", logger="Transformer")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)[0]
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.class_head(concat_f)
        return ret

    def forward_token_mask(self, pts, entropy_threshold=0.5):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer

        x, entropy_mask = self.blocks.forward_token_mask(
            x, pos, entropy_threshold=entropy_threshold
        )
        x = self.norm(x)
        # x = x * (entropy_mask * (-1e6) + 1)
        ret = self.class_head(x[:, 0])

        # entropy_list = torch.FloatTensor(entropy_list)
        return ret


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, list)
                        else drop_path_rate
                    ),
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(
            self.norm(x[:, -return_token_num:])
        )  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.group_norm = config.group_norm
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f"[args] {config.transformer_config}", logger="Transformer")
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(
            encoder_channel=self.encoder_dims, group_norm=self.group_norm
        )

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        """
        center : B G 3
        --------------
        mask : B G (bool)
        """
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(
                points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1
            )  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        """
        center : B G 3
        --------------
        mask : B G (bool)
        """
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)  # 115, 13 masked

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack(
                [
                    np.zeros(G - self.num_mask),
                    np.ones(self.num_mask),
                ]
            )
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, noaug=False, only_unmasked=True):
        # generate mask
        if self.mask_type == "rand":
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        if self.mask_ratio == 0:
            only_unmasked = False

        if only_unmasked:
            batch_size, seq_len, C = group_input_tokens.size()

            x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)

            masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
            pos = self.pos_embed(masked_center)
            x_vis = torch.cat((cls_tokens, x_vis), dim=1)
        else:
            pos = self.pos_embed(center)
            x_vis = torch.cat((cls_tokens, group_input_tokens), dim=1)

        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x_vis, x_vis_feature_list = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos, x_vis_feature_list, group_input_tokens


@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f"[Point_MAE] ", logger="Point_MAE")
        self.config = config
        self.cls_dim = config.cls_dim
        self.group_norm = config.group_norm
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.trans_dim = config.transformer_config.trans_dim

        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.regularize = config.regularize
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)
        ]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        last_dim = 2 * self.trans_dim
        class_blocks = []

        for cls_block in range(0, self.num_hid_cls_layers):
            if self.group_norm:
                norm_layer = nn.GroupNorm(8, 256)
            else:
                norm_layer = nn.BatchNorm1d(256)

            class_blocks.extend(
                (
                    nn.Linear(last_dim, 256),
                    norm_layer,
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                )
            )
            last_dim = 256
        self.class_head = nn.Sequential(
            *class_blocks, nn.Linear(last_dim, self.cls_dim)
        )

        print_log(
            f"[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...",
            logger="Point_MAE",
        )
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=0.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)
        self.l1_consistency_loss = torch.nn.L1Loss(reduction="mean")

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == "cdl2":
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

        self.loss_ce = nn.CrossEntropyLoss()

        # self.loss_func = emd().cuda()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=None):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
            }

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger="Transformer",
                )
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger="Transformer",
                )

            print_log(
                f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                logger="Transformer",
            )
        else:
            print_log("Training from scratch!!!", logger="Transformer")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def classification_only(self, pts, only_unmasked=True):
        neighborhood, center = self.group_divider(pts)
        x_vis_w_token = self.MAE_encoder(
            neighborhood, center, only_unmasked=only_unmasked
        )[0]
        feat = torch.cat([x_vis_w_token[:, 0], x_vis_w_token[:, 1:].max(1)[0]], dim=-1)
        class_ret = self.class_head(feat)
        return class_ret

    def forward(self, pts, vis=False, cyclic=False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        x_vis_w_token, mask, _, _ = self.MAE_encoder(neighborhood, center)
        x_vis = x_vis_w_token[:, 1:]
        B, _, C = x_vis.shape  # B VIS C
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        feat = torch.cat([x_vis_w_token[:, 0], x_vis_w_token[:, 1:].max(1)[0]], dim=-1)

        if not cyclic:
            class_ret = self.class_head(feat)
        else:
            class_ret = self.classification_only(
                pts, only_unmasked=False
            )  # return logits from 100% of tokens

        B, M, C = x_rec.shape
        rebuild_points = (
            self.increase_dim(x_rec.transpose(1, 2))
            .transpose(1, 2)
            .reshape(B * M, -1, 3)
        )  # B M 1024
        # if self.MAE_encoder.mask_ratio == 0:
        #     gt_points = neighborhood.reshape(B * M, -1, 3)
        # else:
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)

        loss1 = self.loss_func(rebuild_points, gt_points)

        if self.regularize:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)

            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0).reshape(
                B, self.num_group, 32, 3
            )

            mean_rebuild = torch.mean(full, dim=0)

            regularization_loss = torch.tensor(0, dtype=torch.float).cuda()

            for bs in range((full.shape[0])):
                regularization_loss += self.loss_func(
                    full[bs, :, :, :].squeeze(), mean_rebuild
                )
            regularization_loss = regularization_loss / full.shape[0]

            mean_class_ret = class_ret.mean(dim=0)
            ce_pred_consitency = torch.tensor(0, dtype=torch.float).cuda()

            for bs in range((class_ret.shape[0])):
                ce_pred_consitency += self.l1_consistency_loss(
                    class_ret[bs, :].squeeze(), mean_class_ret.squeeze()
                )
            class_ret = ce_pred_consitency / class_ret.shape[0]

        else:
            regularization_loss = torch.tensor(0, dtype=torch.float).cuda()
            class_ret = class_ret

        # print(self.loss_func)
        # vis = True
        if vis:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1, class_ret, regularization_loss


@MODELS.register_module()
class Point_MAE_only_cls(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f"[Point_MAE] ", logger="Point_MAE")
        self.config = config
        self.cls_dim = config.cls_dim
        self.group_norm = config.group_norm
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.trans_dim = config.transformer_config.trans_dim

        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.regularize = config.regularize
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)
        ]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        last_dim = self.trans_dim  # only_cls_token
        class_blocks = []

        for cls_block in range(0, self.num_hid_cls_layers):
            if self.group_norm:
                norm_layer = nn.GroupNorm(8, 256)
            else:
                norm_layer = nn.BatchNorm1d(256)

            class_blocks.extend(
                (
                    nn.Linear(last_dim, 256),
                    norm_layer,
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                )
            )
            last_dim = 256
        self.class_head = nn.Sequential(
            *class_blocks, nn.Linear(last_dim, self.cls_dim)
        )

        print_log(
            f"[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...",
            logger="Point_MAE",
        )
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=0.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)
        self.l1_consistency_loss = torch.nn.L1Loss(reduction="mean")

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == "cdl2":
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

        self.loss_ce = nn.CrossEntropyLoss()

        # self.loss_func = emd().cuda()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=None):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
            }

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger="Transformer",
                )
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger="Transformer",
                )

            print_log(
                f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                logger="Transformer",
            )
        else:
            print_log("Training from scratch!!!", logger="Transformer")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def classification_only(self, pts, only_unmasked=True):
        neighborhood, center = self.group_divider(pts)
        x_vis_w_token = self.MAE_encoder(
            neighborhood, center, only_unmasked=only_unmasked
        )[0]
        # feat = torch.cat([, x_vis_w_token[:, 1:].max(1)[0]], dim=-1)
        class_ret = self.class_head(x_vis_w_token[:, 0])  # only_cls_token
        return class_ret

    def forward(self, pts, vis=False, cyclic=False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        x_vis_w_token, mask, _, _ = self.MAE_encoder(neighborhood, center)
        x_vis = x_vis_w_token[:, 1:]
        B, _, C = x_vis.shape  # B VIS C
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        # feat = torch.cat([, x_vis_w_token[:, 1:].max(1)[0]], dim=-1)

        if not cyclic:
            class_ret = self.class_head(x_vis_w_token[:, 0])  # only_cls_token
        else:
            class_ret = self.classification_only(
                pts, only_unmasked=False
            )  # return logits from 100% of tokens

        B, M, C = x_rec.shape
        rebuild_points = (
            self.increase_dim(x_rec.transpose(1, 2))
            .transpose(1, 2)
            .reshape(B * M, -1, 3)
        )  # B M 1024
        # if self.MAE_encoder.mask_ratio == 0:
        #     gt_points = neighborhood.reshape(B * M, -1, 3)
        # else:
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)

        loss1 = self.loss_func(rebuild_points, gt_points)

        if self.regularize:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)

            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0).reshape(
                B, self.num_group, 32, 3
            )

            mean_rebuild = torch.mean(full, dim=0)

            regularization_loss = torch.tensor(0, dtype=torch.float).cuda()

            for bs in range((full.shape[0])):
                regularization_loss += self.loss_func(
                    full[bs, :, :, :].squeeze(), mean_rebuild
                )
            regularization_loss = regularization_loss / full.shape[0]

            mean_class_ret = class_ret.mean(dim=0)
            ce_pred_consitency = torch.tensor(0, dtype=torch.float).cuda()

            for bs in range((class_ret.shape[0])):
                ce_pred_consitency += self.l1_consistency_loss(
                    class_ret[bs, :].squeeze(), mean_class_ret.squeeze()
                )
            class_ret = ce_pred_consitency / class_ret.shape[0]

        else:
            regularization_loss = torch.tensor(0, dtype=torch.float).cuda()
            class_ret = class_ret

        # print(self.loss_func)
        # vis = True
        if vis:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1, class_ret, regularization_loss


# todo pointmae model for joint-training of RotNet (sun et al. TTT)
@MODELS.register_module()
class Point_MAE_rotnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f"[Point_MAE] ", logger="Point_MAE")
        self.config = config
        self.cls_dim = config.cls_dim
        self.cls_dim_rotation = config.cls_dim_rotation
        self.group_norm = config.group_norm
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.trans_dim = config.transformer_config.trans_dim

        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)
        ]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        last_dim = 2 * self.trans_dim
        class_blocks = []

        for cls_block in range(0, self.num_hid_cls_layers):
            if self.group_norm:
                norm_layer = nn.GroupNorm(8, 256)
            else:
                norm_layer = nn.BatchNorm1d(256)

            class_blocks.extend(
                (
                    nn.Linear(last_dim, 256),
                    norm_layer,
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                )
            )
            last_dim = 256
        self.class_head = nn.Sequential(
            *class_blocks, nn.Linear(last_dim, self.cls_dim)
        )  # outputs == num of classes
        self.class_head_rotnet = nn.Sequential(
            *class_blocks, nn.Linear(last_dim, self.cls_dim_rotation)
        )  # 4 outputs

        print_log(
            f"[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...",
            logger="Point_MAE",
        )
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        trunc_normal_(self.mask_token, std=0.02)
        # loss
        self.loss_ce = nn.CrossEntropyLoss()

    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=None):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
            }

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger="Transformer",
                )
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger="Transformer",
                )

            print_log(
                f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                logger="Transformer",
            )
        else:
            print_log("Training from scratch!!!", logger="Transformer")
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def classification_only(self, pts, pts_rot, gt, gt_rot, tta=False):
        if not tta:
            neighborhood, center = self.group_divider(pts)
            neighborhood_rot, center_rot = self.group_divider(pts_rot)

            x_vis_w_token = self.MAE_encoder(neighborhood, center, only_unmasked=False)[
                0
            ]
            x_vis_w_token_rot = self.MAE_encoder(
                neighborhood_rot, center_rot, only_unmasked=False
            )[0]

            feat = torch.cat(
                [x_vis_w_token[:, 0], x_vis_w_token[:, 1:].max(1)[0]], dim=-1
            )
            feat_rot = torch.cat(
                [x_vis_w_token_rot[:, 0], x_vis_w_token_rot[:, 1:].max(1)[0]], dim=-1
            )

            class_ret = self.class_head(feat)
            class_ret_rot = self.class_head_rotnet(feat_rot)

            pred_rot = class_ret_rot.argmax(-1)
            acc_cls_rot = (pred_rot == gt_rot).sum() / float(gt.size(0))
            pred = class_ret.argmax(-1)
            acc_cls = (pred == gt).sum() / float(gt.size(0))

            return acc_cls * 100, acc_cls_rot * 100
        else:
            neighborhood, center = self.group_divider(pts)
            x_vis_w_token = self.MAE_encoder(neighborhood, center, only_unmasked=False)[
                0
            ]
            feat = torch.cat(
                [x_vis_w_token[:, 0], x_vis_w_token[:, 1:].max(1)[0]], dim=-1
            )
            class_ret = self.class_head(feat)

            return class_ret

    def forward(self, pts, pts_rot, gt, gt_rot, tta=False, **kwargs):
        if not tta:
            neighborhood, center = self.group_divider(pts)
            neighborhood_rot, center_rot = self.group_divider(pts_rot)

            x_vis_w_token = self.MAE_encoder(neighborhood, center, only_unmasked=False)[
                0
            ]
            x_vis_w_token_rot = self.MAE_encoder(
                neighborhood_rot, center_rot, only_unmasked=False
            )[0]

            feat = torch.cat(
                [x_vis_w_token[:, 0], x_vis_w_token[:, 1:].max(1)[0]], dim=-1
            )
            feat_rot = torch.cat(
                [x_vis_w_token_rot[:, 0], x_vis_w_token_rot[:, 1:].max(1)[0]], dim=-1
            )

            class_ret = self.class_head(feat)
            class_ret_rot = self.class_head_rotnet(feat_rot)

            loss_cls = self.loss_ce(class_ret, gt.long())
            loss_rot = self.loss_ce(class_ret_rot, gt_rot.long())
            pred_rot = class_ret_rot.argmax(-1)
            acc_cls_rot = (pred_rot == gt_rot).sum() / float(gt.size(0))
            pred = class_ret.argmax(-1)
            acc_cls = (pred == gt).sum() / float(gt.size(0))
            return loss_cls, loss_rot, acc_cls * 100, acc_cls_rot * 100
        else:
            neighborhood_rot, center_rot = self.group_divider(pts_rot)
            x_vis_w_token_rot = self.MAE_encoder(
                neighborhood_rot, center_rot, only_unmasked=False
            )[0]
            feat_rot = torch.cat(
                [x_vis_w_token_rot[:, 0], x_vis_w_token_rot[:, 1:].max(1)[0]], dim=-1
            )
            class_ret_rot = self.class_head_rotnet(feat_rot)
            loss_rot = self.loss_ce(class_ret_rot, gt_rot.long())
            return loss_rot


##### PointMAE for Part Segmentation #####
@MODELS.register_module()
class Point_MAE_PartSegmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f"[Point_MAE_Segmentation] ", logger="Point_MAE_Segmentation")
        self.config = config
        self.npoint = config.npoint
        self.cls_dim = config.cls_dim
        self.num_classes = config.num_classes
        self.group_norm = config.group_norm
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)
        ]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        convs1 = nn.Conv1d(3392, 512, 1)
        dp1 = nn.Dropout(0.5)
        convs2 = nn.Conv1d(512, 256, 1)
        convs3 = nn.Conv1d(256, self.cls_dim, 1)
        bns1 = nn.BatchNorm1d(512)
        bns2 = nn.BatchNorm1d(256)

        relu = nn.ReLU()

        class_blocks = [convs1, bns1, relu, dp1, convs2, bns2, relu, convs3]

        self.class_head = nn.Sequential(*class_blocks)

        print_log(
            f"[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...",
            logger="Point_MAE_Segmentation",
        )
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=1152 + 3, mlp=[self.trans_dim * 4, 1024]
        )

        trunc_normal_(self.mask_token, std=0.02)
        self.loss = config.loss
        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == "cdl2":
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

        self.loss_seg = nn.NLLLoss()

    def get_acc(self, args, seg_pred, target):
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        acc = correct.item() / (args.batch_size * self.npoint)
        return acc

    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=False):
        if load_part_seg:
            ckpt = torch.load(bert_ckpt_path)

            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()
            }

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger="Transformer",
                )
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger="Transformer",
                )

            print_log(
                f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                logger="Transformer",
            )

        else:
            if bert_ckpt_path is not None:
                ckpt = torch.load(bert_ckpt_path)
                base_ckpt = {
                    k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
                }

                incompatible = self.load_state_dict(base_ckpt, strict=False)

                if incompatible.missing_keys:
                    print_log("missing_keys", logger="Transformer")
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger="Transformer",
                    )
                if incompatible.unexpected_keys:
                    print_log("unexpected_keys", logger="Transformer")
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger="Transformer",
                    )

                print_log(
                    f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                    logger="Transformer",
                )
            else:
                print_log("Training from scratch!!!", logger="Transformer")
                self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def classification_only(self, pts, cls_label, only_unmasked=True):
        B, N, C = pts.shape
        neighborhood, center = self.group_divider(pts)
        x_vis_w_token, mask, feature_list, group_input_tokens = self.MAE_encoder(
            neighborhood, center, only_unmasked=only_unmasked
        )
        feature_list = [
            self.norm(x).transpose(-1, -2).contiguous() for x in feature_list
        ]

        x = torch.cat(
            (feature_list[0], feature_list[1], feature_list[2]), dim=1
        )  # 1152

        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)

        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)

        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x_global_feature = torch.cat(
            (x_max_feature, x_avg_feature, cls_label_feature), 1
        )  # 1152*2 + 64
        f_level_0 = self.propagation_0(
            pts.transpose(-1, -2),
            center.transpose(-1, -2),
            pts.transpose(-1, -2),
            x,
            mask_ratio=self.MAE_encoder.mask_ratio,
        )

        x = torch.cat((f_level_0, x_global_feature), 1)

        class_ret = self.class_head(x)
        class_ret = F.log_softmax(class_ret, dim=1)
        class_ret = class_ret.permute(0, 2, 1)
        return class_ret

    def forward(
        self, pts, cls_label, cls_loss_masked=True, tta=False, vis=False, **kwargs
    ):
        B_, N_, _ = (
            pts.shape
        )  # pts (8, 2048, 3),  cls_label (8,1,16) one-hot ,   partnet_cls

        neighborhood, center = self.group_divider(
            pts
        )  # normalized neighborhood  (8, 128, 32, 3) 128 groups, each group has 32 points,   center (8, 128, 3)  128 group centers
        x_vis_w_token, mask, feature_list, group_input_tokens = self.MAE_encoder(
            neighborhood, center
        )
        #  todo x_vis_w_token (8, 14, 384), mask (8,128) feature_list 3-level features:  a list of (8,14,384),  group_input_tokens (8,128,384)
        x_vis = x_vis_w_token[:, 1:]
        B, _, C = x_vis.shape  # B VIS C
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(
            B, -1, C
        )  # positional embedding for visible tokens  13
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(
            B, -1, C
        )  # positional embedding for masked tokens   115

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)  # (8, 115, 384)
        x_full = torch.cat([x_vis, mask_token], dim=1)  # (8, 128, 384)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)  # (8, 128, 384)

        x_rec = self.MAE_decoder(
            x_full, pos_full, N
        )  # #  todo only the masked token are reconstructed  (8, 115, 384)
        if not tta:
            feature_list = [
                self.norm(x).transpose(-1, -2).contiguous() for x in feature_list
            ]  # feature_list  a list of (8,384, 14)
            x = torch.cat(
                (feature_list[0], feature_list[1], feature_list[2]), dim=1
            )  # (8,1152,14)    384x3 = 1152
            x_max = torch.max(x, 2)[0]  # (8, 1152)
            x_avg = torch.mean(x, 2)  # (8, 1152)
            # todo 3 types of features: maxpoing global feature, avgpooling global feature, object label feature,   duplicate to N=2048
            x_max_feature = (
                x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N_)
            )  # todo  duplicate the feature on 3rd dimension for N=2048 (8, 1152, 2048)
            x_avg_feature = (
                x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N_)
            )  # (8, 1152, 2048)
            # todo  cls_label is object category label, it is considered as a data source, which is used to compute features
            cls_label_one_hot = cls_label.view(B, self.num_classes, 1)
            cls_label_feature = self.label_conv(cls_label_one_hot).repeat(
                1, 1, N_
            )  # (8, 64, 2048)

            x_global_feature = torch.cat(
                (x_max_feature, x_avg_feature, cls_label_feature), 1
            )  # (8, 2368, 2048)    1152*2 + 64 = 2368, feature dim

            # todo  the problem is
            #   x is the concatenation of 3-level featurse only for the 14 visible tokens (note that only visible tokens have features from encoder)
            #    but here the center is still  all 128 centers
            #  todo ############ suggested correction ################################################
            #   ############################################################

            n_visible_tokens = x_vis.size(1)
            center_visible = center[~mask].reshape(B, n_visible_tokens, 3)
            f_level_0 = self.propagation_0(
                pts.transpose(-1, -2),
                center_visible.transpose(-1, -2),
                pts.transpose(-1, -2),
                x,
                mask_ratio=self.MAE_encoder.mask_ratio,
            )
            # todo instead of
            # f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)
            #  todo ############################################################
            #     ############################################################

            x = torch.cat((f_level_0, x_global_feature), 1)

            # todo - if we do not want to pass the tokens through the cls head set 'cls_loss_masked' to False
            # todo - if this is false, cls outputs are taken from the method - 'classification_only'
            # todo - the advantage of taking the outputs from 'classification_only' is that cls loss can be computed from
            # todo - 100% of the tokens!!!
            if cls_loss_masked:
                class_ret = self.class_head(x)
                class_ret = F.log_softmax(class_ret, dim=1)
                class_ret = class_ret.permute(0, 2, 1)
            else:
                class_ret = self.classification_only(
                    pts, cls_label, only_unmasked=False
                )
        else:
            class_ret = 0

        B, M, C = x_rec.shape
        rebuild_points = (
            self.increase_dim(x_rec.transpose(1, 2))
            .transpose(1, 2)
            .reshape(B * M, -1, 3)
        )  # B M 1024

        if vis:  # visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center

        # dummy = neighborhood[mask]
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)
        return loss1, class_ret


##### PointMAE for Semantic Segmentation #####
@MODELS.register_module()
class Point_MAE_SemSegmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f"[Point_MAE_Segmentation] ", logger="Point_MAE_Segmentation")
        self.config = config
        self.npoint = config.npoint
        self.cls_dim = config.cls_dim
        self.group_norm = config.group_norm
        self.num_hid_cls_layers = config.num_hid_cls_layers
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)
        ]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        convs1 = nn.Conv1d(3328, 512, 1)
        dp1 = nn.Dropout(0.5)
        convs2 = nn.Conv1d(512, 256, 1)
        convs3 = nn.Conv1d(256, self.cls_dim, 1)
        bns1 = nn.BatchNorm1d(512)
        bns2 = nn.BatchNorm1d(256)

        relu = nn.ReLU()

        class_blocks = [convs1, bns1, relu, dp1, convs2, bns2, relu, convs3]

        self.class_head = nn.Sequential(*class_blocks)

        print_log(
            f"[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...",
            logger="Point_MAE_Segmentation",
        )
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=1152 + 3, mlp=[self.trans_dim * 4, 1024]
        )

        trunc_normal_(self.mask_token, std=0.02)
        self.loss = config.loss
        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == "cdl2":
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

        self.loss_seg = nn.NLLLoss()

    def get_acc(self, args, seg_pred, target):
        pred_choice = seg_pred.data.max(1)[1]
        # import pdb
        # pdb.set_trace()
        correct = pred_choice.eq(target.data).cpu().sum()
        acc = correct.item() / (args.batch_size * self.npoint)
        return acc

    def load_model_from_ckpt(self, bert_ckpt_path, load_part_seg=False):
        if load_part_seg:
            ckpt = torch.load(bert_ckpt_path)

            base_ckpt = {
                k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()
            }

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log("missing_keys", logger="Transformer")
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger="Transformer",
                )
            if incompatible.unexpected_keys:
                print_log("unexpected_keys", logger="Transformer")
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger="Transformer",
                )

            print_log(
                f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                logger="Transformer",
            )

        else:
            if bert_ckpt_path is not None:
                ckpt = torch.load(bert_ckpt_path)
                base_ckpt = {
                    k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
                }

                incompatible = self.load_state_dict(base_ckpt, strict=False)

                if incompatible.missing_keys:
                    print_log("missing_keys", logger="Transformer")
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger="Transformer",
                    )
                if incompatible.unexpected_keys:
                    print_log("unexpected_keys", logger="Transformer")
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger="Transformer",
                    )

                print_log(
                    f"[Transformer] Successful Loading the ckpt from {bert_ckpt_path}",
                    logger="Transformer",
                )
            else:
                print_log("Training from scratch!!!", logger="Transformer")
                self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # todo # now it should be segmentation # todo
    def classification_only(self, pts, only_unmasked=True):
        B, N, C = pts.shape
        neighborhood, center = self.group_divider(pts)
        x_vis_w_token, mask, feature_list, group_input_tokens = self.MAE_encoder(
            neighborhood, center, only_unmasked=only_unmasked
        )
        feature_list = [
            self.norm(x).transpose(-1, -2).contiguous() for x in feature_list
        ]

        x = torch.cat(
            (feature_list[0], feature_list[1], feature_list[2]), dim=1
        )  # 1152

        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)

        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)

        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)  # 1152*2 + 64
        f_level_0 = self.propagation_0(
            pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x
        )

        x = torch.cat((f_level_0, x_global_feature), 1)

        class_ret = self.class_head(x)
        class_ret = F.log_softmax(class_ret, dim=1)
        class_ret = class_ret.permute(0, 2, 1)
        return class_ret

    def forward(self, pts, cls_loss_masked=True, tta=False, **kwargs):
        B_, N_, _ = (
            pts.shape
        )  # pts (8, 2048, 3),  cls_label (8,1,16) one-hot ,   partnet_cls

        neighborhood, center = self.group_divider(
            pts
        )  # normalized neighborhood  (8, 128, 32, 3) 128 groups, each group has 32 points,   center (8, 128, 3)  128 group centers
        x_vis_w_token, mask, feature_list, group_input_tokens = self.MAE_encoder(
            neighborhood, center
        )
        #  todo x_vis_w_token (8, 14, 384), mask (8,128) feature_list 3-level features:  a list of (8,14,384),  group_input_tokens (8,128,384)
        x_vis = x_vis_w_token[:, 1:]
        B, _, C = x_vis.shape  # B VIS C
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(
            B, -1, C
        )  # positional embedding for visible tokens  13
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(
            B, -1, C
        )  # positional embedding for masked tokens   115
        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)  # (8, 115, 384)
        x_full = torch.cat([x_vis, mask_token], dim=1)  # (8, 128, 384)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)  # (8, 128, 384)

        x_rec = self.MAE_decoder(
            x_full, pos_full, N
        )  # #  todo only the masked token are reconstructed  (8, 115, 384)
        if not tta:
            feature_list = [
                self.norm(x).transpose(-1, -2).contiguous() for x in feature_list
            ]  # feature_list  a list of (8,384, 14)
            # todo concatenation of 3-level features only for 14 visible tokens
            x = torch.cat(
                (feature_list[0], feature_list[1], feature_list[2]), dim=1
            )  # (8,1152,14)    384x3 = 1152
            x_max = torch.max(x, 2)[0]  # (8, 1152)
            x_avg = torch.mean(x, 2)  # (8, 1152)
            # todo 3 types of features: maxpoing global feature, avgpooling global feature, object label feature,   duplicate to N=2048
            x_max_feature = (
                x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N_)
            )  # todo  duplicate the feature on 3rd dimension for N=2048 (8, 1152, 2048)
            x_avg_feature = (
                x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N_)
            )  # (8, 1152, 2048)

            x_global_feature = torch.cat(
                (x_max_feature, x_avg_feature), 1
            )  # (8, 2368, 2048)    1152*2 + 64 = 2368, feature dim

            # todo  the problem is
            #   x is the concatenation of 3-level featurse only for the 14 visible tokens (note that only visible tokens have features from encoder)
            #    but here the center is still  all 128 centers
            #  todo ############ suggested correction ################################################
            #   ############################################################

            n_visible_tokens = x_vis.size(1)
            center_visible = center[~mask].reshape(B, n_visible_tokens, 3)
            f_level_0 = self.propagation_0(
                pts.transpose(-1, -2),
                center_visible.transpose(-1, -2),
                pts.transpose(-1, -2),
                x,
            )
            # todo instead of
            # f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)
            #  todo ############################################################
            #     ############################################################

            x = torch.cat((f_level_0, x_global_feature), 1)

            # todo - if we do not want to pass the tokens through the cls head set 'cls_loss_masked' to False
            # todo - if this is false, cls outputs are taken from the method - 'classification_only'
            # todo - the advantage of taking the outputs from 'classification_only' is that cls loss can be computed from
            # todo - 100% of the tokens!!!
            if cls_loss_masked:
                class_ret = self.class_head(x)
                class_ret = F.log_softmax(class_ret, dim=1)
                class_ret = class_ret.permute(0, 2, 1)
            else:
                class_ret = self.classification_only(pts, only_unmasked=False)
        else:
            class_ret = 0

        B, M, C = x_rec.shape
        rebuild_points = (
            self.increase_dim(x_rec.transpose(1, 2))
            .transpose(1, 2)
            .reshape(B * M, -1, 3)
        )  # B M 1024
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)
        return loss1, class_ret
