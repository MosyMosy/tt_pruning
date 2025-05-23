import os
import argparse
from pathlib import Path


def none_or_str(value):
    if value == "None":
        return None
    return value


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_ratio", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--batch_size_tta", type=int, default=1)
    parser.add_argument("--stride_step", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_steps", type=int, default=1)
    parser.add_argument(
        "--disable_bn_adaptation",
        action="store_true",
        default=False,
        help="to disable bn_for adaptation",
    )
    parser.add_argument(
        "--online", action="store_true", default=False, help="online-adapt"
    )
    parser.add_argument("--visualize_data", action="store_true", help="image creation")
    parser.add_argument(
        "--ckpts",
        type=none_or_str,
        default="checkpoints/scan_object_src_only.pth",
        help="test used ckpt path",
    )  # default="checkpoints/modelnet_src_only.pth", help='test used ckpt path'
    parser.add_argument(
        "--config",
        type=str,
        default="cfgs/tta_purge/tta_purge_scanobject.yaml",
        help="yaml config file",
    )
    parser.add_argument(
        "--group_norm",
        action="store_true",
        help="If Group Norm shall be used instead of Batch Norm",
    )
    parser.add_argument(
        "--test_source", action="store_true", default=False, help="test source model"
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="test mode for test-time adaptation",
    )
    parser.add_argument(
        "--tta_seg",
        action="store_true",
        default=False,
        help="test mode for test-time adaptation for part segmentation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="only load small number of samples",
    )
    parser.add_argument(
        "--partnet_cls",
        action="store_true",
        default=False,
        help="train partnet for obj classification task",
    )
    parser.add_argument(
        "--jt", action="store_true", default=False, help="train model with JT"
    )
    parser.add_argument(
        "--only_cls",
        action="store_true",
        default=False,
        help="train model only for cls task / without JT",
    )
    parser.add_argument(
        "--train_aug",
        action="store_true",
        default=False,
        help="weather to use augmentations for train/test",
    )
    parser.add_argument(
        "--cyclic",
        action="store_true",
        default=False,
        help="get cls loss with 100% tokens and recon loss with 10% - used for joint pretraining!!!",
    )
    parser.add_argument(
        "--tta_rot", action="store_true", default=False, help="do tta for rotnet"
    )
    parser.add_argument(
        "--train_tttrot", action="store_true", default=False, help="train ttt rotnet"
    )
    parser.add_argument(
        "--only_unmasked",
        action="store_true",
        default=False,
        help="weather to use 100% tokens for classification or not",
    )
    parser.add_argument(
        "--test_source_rotnet", action="store_true", default=False, help="train-ttt-rot"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="shuffle data for adaptation",
    )
    parser.add_argument(
        "--tta_dua",
        action="store_true",
        default=False,
        help="for running adaptatation with DUA",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="tent",
        help="run adaptive baselines, choose from TENT, DUA, T3A, SHOT",
    )
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    # seed
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend. -1 for random",
    )
    # bn
    parser.add_argument(
        "--sync_bn", action="store_true", default=False, help="whether to use sync bn"
    )
    # some args
    parser.add_argument(
        "--exp_name", type=str, default="default", help="experiment name"
    )
    parser.add_argument("--loss", type=str, default="cd1", help="loss name")
    parser.add_argument(
        "--start_ckpts", type=str, default=None, help="reload used ckpt path"
    )
    parser.add_argument("--val_freq", type=int, default=1, help="test freq")
    parser.add_argument("--vote", action="store_true", default=False, help="vote acc")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="autoresume training (interrupted by accident)",
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="test mode for certain ckpt"
    )
    parser.add_argument(
        "--finetune_model",
        action="store_true",
        default=False,
        help="finetune modelnet with pretrained weight",
    )
    parser.add_argument(
        "--scratch_model",
        action="store_true",
        default=False,
        help="training modelnet from scratch",
    )

    parser.add_argument(
        "--BN_reset",
        action="store_true",
        help="Reset batch norm running statistics similar to TENT",
        # default=True
    )

    parser.add_argument(
        "--mode",
        choices=["easy", "median", "hard", None],
        default=None,
        help="difficulty mode for shapenet",
    )
    parser.add_argument("--way", type=int, default=-1)
    parser.add_argument("--shot", type=int, default=-1)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--train_with_purge", action="store_true", default=False)
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "source_only",
            "prototype_purge",
            "cls_purge",
            "bftt3d",
            "tent",
            "rotnet",
            "t3a",
            "shot",
            "dua",
            "tta_x",
            "tta_token_mask",
            "with_intermediate",
            "unclassified",
            "tent_intermediate",
            "tta_layer_prune",
            "layer_average",
            "tta_all_BN",
            "tta_cls_stat",
        ],
        default="prototype_purge",
    )

    parser.add_argument(
        "--cls_fixer_mode",
        type=str,
        choices=[
            "source_only",
            "source_only_cls-fixer",
            "update_tent",
            "update_tent_cls-fixer",
        ],
        default="source_only",
    )

    parser.add_argument(
        "--selected_corruption",
        type=str,
        choices=[
            "scanobject_bg",
            "scanobject_hd",
            "uniform",
            "gaussian",
            "background",
            "impulse",
            "upsampling",
            "distortion_rbf",
            "distortion_rbf_inv",
            "density",
            "density_inc",
            "shear",
            "rotation",
            "cutout",
            "distortion",
            "occlusion",
            "lidar",
        ],
        default=None,
    )

    parser.add_argument("--prune_list", nargs="*", type=int, default=[])
    parser.add_argument(
        "--purne_attention",
        action="store_true",
        default=False,
        help="whether to prune the whole layer or just attention",
    )

    parser.add_argument(
        "--purge_size_list", nargs="*", type=int, default=[0, 2, 4, 8, 16, 32]
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.5,
        help="entropy threshold for tta token mask",
    )

    parser.add_argument(
        "--layer_idx",
        type=int,
        default=11,
        help="layer index for with_intermediate",
    )

    # parser.add_argument("--LR", type=float, default=1e-5)
    parser.add_argument("--LR", type=float, default=0.001)
    parser.add_argument("--BETA", type=float, default=0.9)
    parser.add_argument("--WD", type=float, default=0.0)

    parser.add_argument("--bftt3d_stages", type=int, default=3)
    parser.add_argument("--bftt3d_dim", type=int, default=72)
    parser.add_argument("--bftt3d_k", type=int, default=120)
    parser.add_argument("--bftt3d_alpha", type=int, default=1000)
    parser.add_argument("--bftt3d_beta", type=int, default=100)
    parser.add_argument("--bftt3d_gamma", type=int, default=205)

    parser.add_argument("--t3a_filter_k", type=int, default=-1)

    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError("--test and --resume cannot be both activate")

    if args.resume and args.start_ckpts is not None:
        raise ValueError("--resume and --start_ckpts cannot be both activate")

    if args.test and args.ckpts is None:
        raise ValueError("ckpts shouldnt be None while test mode")

    if args.finetune_model and args.ckpts is None:
        print("training from scratch")

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.test:
        args.exp_name = "test_" + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + "_" + args.mode
    args.experiment_path = os.path.join(
        "./experiments",
        Path(args.config).stem,
        Path(args.config).parent.stem,
        args.exp_name,
    )
    args.tfboard_path = os.path.join(
        "./experiments",
        Path(args.config).stem,
        Path(args.config).parent.stem,
        "TFBoard",
        args.exp_name,
    )
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print("Create experiment path successfully at %s" % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print("Create TFBoard path successfully at %s" % args.tfboard_path)
