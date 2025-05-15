from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
from tools.tta_purge import runner
from tools.tta_BFTT3D import runner as runner_BFTT3D
from tools.tta import (
    tta_tent,
    tta_rotnet,
    tta_t3a,
    tta_shot,
    tta_dua,
    eval_source,
    tta_tent_intermediate,
    eval_source_layer_average,
    eval_source_all_BN,
)
from tools.tta_unclassified import runner as runner_unclassified
from tools.tta_intermediate import runner as runner_intermediate

# from tools.tta_x import runner as runner_x
from tools.tta_token_mask import runner as runner_token_mask
from tools.tta_layer_prune import runner as runner_layer_prune
from tools.tta_cls_stat import runner as runner_cls_stat
import time
import os
import torch
from tensorboardX import SummaryWriter

import random, numpy as np
from tools import builder
import datasets.tta_datasets as tta_datasets


import faiss
from tqdm import tqdm

import open3d as o3d
from pytorch3d.ops import sample_farthest_points  # > pip install pytorch3d
from pytorch3d.ops import knn_points  # fast Chamfer

from typing import Tuple, Any

# ──────────────────────────────────────────────────────────────── #


def prepare_config(args):
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(args.experiment_path, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, "train"))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, "test"))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger=logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get("extra_train"):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get("test"):
            config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get("extra_train"):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get("test"):
            config.dataset.test.others.bs = config.total_bs
            # log
    log_args_to_file(args, "args", logger=logger)
    log_config_to_file(config, "config", logger=logger)
    logger.info(f"Distributed training: {args.distributed}")
    # set random seeds
    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, deterministic: {args.deterministic}"
        )
        misc.set_random_seed(
            args.seed + args.local_rank, deterministic=args.deterministic
        )  # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold
    dataset_name = config.dataset.name
    assert args.ckpts is not None
    assert dataset_name is not None

    # args.disable_bn_adaptation = True
    # args.batch_size_tta = 48
    # args.batch_size = 1
    # config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    args.split = "test"
    return args, config


def load_tta_dataset(args, config):
    # we have 3 choices - every tta_loader returns only point and labels
    root = config.tta_dataset_path  # being lazy - 1

    if config.dataset.name == "modelnet":
        if args.corruption == "clean":
            inference_dataset = tta_datasets.ModelNet_h5(args, root)

        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)

    elif config.dataset.name == "scanobject":
        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)

    elif config.dataset.name == "shapenetcore":
        inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)

    else:
        raise NotImplementedError(f"TTA for {args.tta} is not implemented")

    print(f"\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n")

    return inference_dataset


def load_clean_dataset(args, config):
    (train_sampler, train_dataloader) = builder.dataset_builder(
        args, config.dataset.train
    )
    return train_dataloader.dataset


args, config = prepare_config(parser.get_args())

args.severity = 5
args.corruption = "lidar"

corrupt_dataset = load_tta_dataset(args, config)
clean_dataset = load_clean_dataset(args, config)


# for i in range(len(corrupt_dataset)):
#     corrupt_point = corrupt_dataset[0][0]
#     clean_point = clean_dataset[0][2][0]

#     os.makedirs(
#         f"lab/visualize/{config.dataset.name}/{args.corruption}/",
#         exist_ok=True,
#     )
#     np.savetxt(
#         f"lab/visualize/{config.dataset.name}/{args.corruption}/corrupt_point_{i}.xyz",
#         corrupt_point.cpu().numpy(),
#     )
#     np.savetxt(
#         f"lab/visualize/{config.dataset.name}/clean_point_{i}.xyz",
#         clean_point.cpu().numpy(),
#     )


# ────────────────────────────────────────────────────────────────
#  extra deps : pip install open3d faiss-cpu pytorch3d
# ────────────────────────────────────────────────────────────────
# FPS + Chamfer


def fpfh_descriptor(pc: np.ndarray, voxel: float = 0.03) -> np.ndarray:
    """36-D = 3 eigen-ratios + mean FPFH (33)."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc.astype(np.float32)))
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=32)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100),
    ).data  # (33,N)
    fpfh_mean = fpfh.mean(1)
    pts0 = np.asarray(pcd.points) - pcd.get_center()
    eig = np.sort(np.linalg.eigvalsh(np.cov(pts0.T)))[::-1]
    eig = (eig / eig.sum()).astype(np.float32)
    return np.concatenate([eig, fpfh_mean]).astype(np.float32)


@torch.no_grad()
def chamfer_fast(a_np: np.ndarray, b_np: np.ndarray, K: int = 1024) -> float:
    """
    Chamfer distance with PyTorch3D FPS-K subsampling (no in-place slice error).
    """
    def fps_downsample(pc: torch.Tensor) -> torch.Tensor:    # pc : (P,3)
        if pc.size(0) <= K:
            return pc
        pc_sub, _ = sample_farthest_points(
            pc.unsqueeze(0),  # (1,P,3)
            K=K,
            random_start_point=True
        )
        return pc_sub.squeeze(0)            # → (K,3)

    a = torch.from_numpy(a_np).float().cuda()
    b = torch.from_numpy(b_np).float().cuda()

    a = fps_downsample(a)
    b = fps_downsample(b)

    a = a.unsqueeze(0)                      # (1,P,3)
    b = b.unsqueeze(0)

    d1, _, _ = knn_points(a, b, K=1)
    d2, _, _ = knn_points(b, a, K=1)
    return (d1.mean() + d2.mean()).item()


# ---------- 2. build clean bank ---------------------------------------
clean_descs, clean_clouds, clean_labels = [], [], []
print("Extracting clean descriptors …")
for sample in tqdm(clean_dataset):
    pts, lbl = sample[2]
    pts = pts.cpu().numpy()
    clean_clouds.append(pts)
    clean_labels.append(lbl)
    clean_descs.append(fpfh_descriptor(pts, voxel=0.04))
clean_descs = np.stack(clean_descs).astype(np.float32)
clean_labels = np.array(clean_labels)

# build one FAISS index per class
label_to_index = {}
d = clean_descs.shape[1]
for lbl in np.unique(clean_labels):
    idx = np.where(clean_labels == lbl)[0]
    index = faiss.IndexFlatL2(d)
    index.add(clean_descs[idx])
    label_to_index[lbl] = (index, idx)  # (faiss index, global idxs)

# ---------- 3. scan corrupted set --------------------------------------
best_per_class = {}  # lbl → (corr_idx, clean_idx, cd)
corr_clouds = []                                    # ← ADD THIS LIST

print("Searching best matches per class …")
for j_corr, sample in enumerate(tqdm(corrupt_dataset)):
    pts_c, lbl_c = sample
    # pts_c = pts_c.cpu().numpy()
    corr_clouds.append(pts_c)                       # ← cache here

    if lbl_c not in label_to_index:
        continue  # label absent in clean set

    desc_c = fpfh_descriptor(pts_c, voxel=0.04)[None, :]
    index, clean_sub_idx = label_to_index[lbl_c]

    _, nn = index.search(desc_c, 1)  # (1,1)
    clean_idx_global = clean_sub_idx[int(nn[0, 0])]

    # fine check with Chamfer
    cd = chamfer_fast(pts_c, clean_clouds[clean_idx_global])

    if (lbl_c not in best_per_class) or (cd < best_per_class[lbl_c][-1]):
        best_per_class[lbl_c] = (j_corr, clean_idx_global, cd)

# ---------- 4. save results -------------------------------------------
out_root = f"lab/visualize/{config.dataset.name}/{args.corruption}"
print("\nSaving best pairs …")
for lbl, (j_corr, j_clean, cd) in best_per_class.items():
    cls_dir = f"{out_root}/class_{lbl}"
    os.makedirs(cls_dir, exist_ok=True)

    # use the cached numpy arrays – no .cpu(), no .numpy() needed
    np.savetxt(f"{cls_dir}/corrupt_{j_corr}.xyz", corr_clouds[j_corr])
    np.savetxt(f"{cls_dir}/clean_{j_clean}.xyz",   clean_clouds[j_clean])

    print(f"class {lbl:2d}  ↔  corrupt #{j_corr:5d}  clean #{j_clean:5d}  "
          f"Chamfer={cd:.6f}")
