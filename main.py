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


def main(args):
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

    methods_dict = {
        "source_only": runner,
        "prototype_purge": runner,
        "cls_purge": runner,
        "bftt3d": runner_BFTT3D,
        "tent": tta_tent,
        "rotnet": tta_rotnet,
        "t3a": tta_t3a,
        "shot": tta_shot,
        "dua": tta_dua,
        # "tta_x": runner_x,
        "tta_token_mask": runner_token_mask,
        "with_intermediate": runner_intermediate,
        "unclassified": runner_unclassified,
        "tent_intermediate": tta_tent_intermediate,
        "tta_layer_prune": runner_layer_prune,
        "layer_average": eval_source_layer_average,
        "tta_all_BN": eval_source_all_BN,
        "tta_cls_stat": runner_cls_stat,
    }

    args.split = "test"
    methods_dict[args.method](args, config)


if __name__ == "__main__":
    args = parser.get_args()

    main(args)
