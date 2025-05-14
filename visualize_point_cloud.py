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

args.severity = 3
args.corruption = "lidar"

corrupt_dataset = load_tta_dataset(args, config)
clean_dataset = load_clean_dataset(args, config)


corrupt_point = corrupt_dataset[0][0]
clean_point = clean_dataset[0][2][0]


np.savetxt("corrupt_point.xyz", corrupt_point.cpu().numpy())
np.savetxt("clean_point.xyz", clean_point.cpu().numpy())