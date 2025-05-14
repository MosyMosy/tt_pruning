import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tools import builder
from utils import misc, dist_utils
from utils.logger import get_logger, print_log, get_writer_to_all_result

from utils.misc import corruptions

import datasets.tta_datasets as tta_datasets
from tqdm import tqdm

from models.Point_MAE import Block, Attention

# =====================
# Config
# =====================
level = [5]


# =====================
# Helper Functions
# =====================


@torch.jit.script
def softmax_entropy(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


# =====================
# Custom Normalization Layers
# =====================


def generate_source_cls_embeddings(args, config, source_model):
    def update_stats(x, count, mean):
        count += 1
        delta = x - mean
        mean = mean + delta / count
        return count, mean

    def finalize_stats(count, mean):
        if count < 2:
            # Not enough data points to compute variance
            return mean
        return mean

    clean_dataloader = load_clean_dataset(args, config)
    count = 0
    mean = torch.zeros(25, 384, dtype=float)
    for i, (_, _, data) in tqdm(
        enumerate(clean_dataloader), total=len(clean_dataloader)
    ):
        points = data[0].cuda()
        points = misc.fps(points, config.npoints)
        with torch.no_grad():
            intermediates = source_model.forward_source_norm_embeddings(points)
        intermediates = torch.stack(intermediates, dim=0).flatten(
            1, 2
        )  # (num_layers, batch_size * tokens, emb_dim)

        for i in range(intermediates.shape[1]):
            x = intermediates[:, i]  # x has shape (L, C)
            count, mean = update_stats(x, count, mean)

    return mean


class LayerNormWithClsTransform(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        cls_mean=None,
        cls_std=None,
        out_prototype=None,
    ):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.cls_mean = cls_mean
        self.cls_std = cls_std
        self.out_prototype = out_prototype

    def forward(self, input):
        out = super().forward(input)
        # out = (out - out.mean(dim=-1, keepdim=True)) / (
        #     out.std(dim=-1, keepdim=True) + self.eps
        # )
        out = out * self.cls_std + self.cls_mean
        return out


def replace_layernorm_with_cls_versions(
    model, cls_norms_embedding, device=None, skip=2
):
    if device is None:
        device = next(model.parameters()).device

    norm_idx = 0

    def _recursive_replace(module):
        nonlocal norm_idx
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm1d):
                norm_idx += 1
            elif isinstance(child, nn.LayerNorm):
                if norm_idx >= skip:
                    new_module = LayerNormWithClsTransform(
                        normalized_shape=child.normalized_shape,
                        eps=child.eps,
                        elementwise_affine=child.elementwise_affine,
                        cls_mean=cls_norms_embedding[norm_idx - skip].mean().item(),
                        cls_std=cls_norms_embedding[norm_idx - skip].std().item(),
                        out_prototype=cls_norms_embedding[norm_idx - skip],
                    )
                    setattr(module, name, new_module.to(device))
                norm_idx += 1
            else:
                _recursive_replace(child)

    if isinstance(model, nn.DataParallel):
        model = model.module

    _recursive_replace(model)


# =====================
# Dataset Loaders
# =====================


def load_tta_dataset(args, config):
    root = config.tta_dataset_path

    def seed_worker(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    dataset_name = config.dataset.name.lower()
    if dataset_name == "modelnet":
        if args.corruption == "clean":
            dataset = tta_datasets.ModelNet_h5(args, root)
        else:
            dataset = tta_datasets.ModelNet40C(args, root)
    elif dataset_name == "scanobject":
        dataset = tta_datasets.ScanObjectNN(args=args, root=root)
    elif dataset_name == "shapenetcore":
        dataset = tta_datasets.ShapeNetCore(args=args, root=root)
    else:
        raise NotImplementedError(f"TTA for {args.tta} is not implemented")

    tta_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=False,
    )

    print(f"\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n")
    return tta_loader


def load_clean_dataset(args, config):
    train_sampler, train_dataloader = builder.dataset_builder(
        args, config.dataset.train
    )
    return train_dataloader


# =====================
# Model Management
# =====================


def load_base_model(args, config, load_part_seg=False, pretrained=True):
    base_model = builder.model_builder(config.model)
    if pretrained:
        base_model.load_model_from_ckpt(args.ckpts)
    return base_model.cuda()


def reset_model(args, config, source_model, cls_norms_embedding):
    model = load_base_model(args, config, pretrained=False)
    if "cls-fixer" in args.cls_fixer_mode:
        replace_layernorm_with_cls_versions(model, cls_norms_embedding)
    model.load_state_dict(source_model.state_dict())
    model.requires_grad_(False)

    optimizer = None
    if "update_tent" in args.cls_fixer_mode:
        params = []
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                m.requires_grad_(True)
                for n_p, p in m.named_parameters():
                    if n_p in ["weight", "bias"]:
                        params.append(p)
                        p.requires_grad_(True)

        optimizer = optim.Adam(
            params, lr=args.LR, betas=(args.BETA, 0.999), weight_decay=args.WD
        )

    if args.BN_reset:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        

    return model, optimizer


def reset_model_preserve_rng(*args, **kwargs):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    model, opt = reset_model(*args, **kwargs)
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state)
    return model, opt


# =====================
# Evaluation Runner
# =====================


def runner(args, config):
    dataset_name = config.dataset.name
    logger = get_logger(args.log_name)

    source_model = load_base_model(args, config)
    source_model.eval()

    result_file_path = os.path.join(
        "results_final_tta/",
        args.method,
        f"{args.exp_name}_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
    )

    eval(args, config, logger, source_model, result_file_path)


# =====================
# Evaluation Logic
# =====================


def eval(args, config, logger, source_model, result_file_path):
    time_list = []
    acc_list = []

    if False:
        cls_norms_embedding = generate_source_cls_embeddings(args, config, source_model)
    else:
        cls_norms_embedding = source_model.forward_norms_embedding()

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            if args.corruption == "clean":
                continue  # Skipping clean data

            start_time = time.time()
            acc_avg = []

            # Initialize writer if not done
            if "f_write" not in locals():
                f_write = get_writer_to_all_result(
                    args, config, custom_path=result_file_path
                )
                f_write.write(f"All Corruptions: {corruptions}\n\n")
                f_write.write(f"TTA Results for Dataset: {config.dataset.name}\n\n")
                f_write.write(f"Checkpoint Used: {args.ckpts}\n\n")
                f_write.write(f"Corruption LEVEL: {args.severity}\n\n")

            tta_loader = load_tta_dataset(args, config)
            test_pred, test_label = [], []

            if args.online:
                base_model, optimizer = reset_model_preserve_rng(
                    args, config, source_model, cls_norms_embedding
                )
                base_model.eval()

            for idx, (data, labels) in enumerate(tta_loader):
                if data.shape[0] == 1:
                    data = data.repeat(2, 1, 1)
                    labels = labels.repeat(2, 1)

                if not args.online:
                    base_model, optimizer = reset_model_preserve_rng(
                        args, config, source_model, cls_norms_embedding
                    )
                    base_model.eval()

                points = data.cuda()
                labels = labels.cuda()
                points = [
                    misc.fps(points, config.npoints, random_start_point=False)
                    for _ in range(args.batch_size_tta)
                ]
                points = torch.cat(points, dim=0)
                labels = torch.cat([labels for _ in range(args.batch_size_tta)], dim=0)

                if "source_only" in args.cls_fixer_mode:
                    base_model.eval()
                    with torch.no_grad():
                        logits = base_model(points)

                elif "update_tent" in args.cls_fixer_mode:
                    base_model.train()
                    for _ in range(args.grad_steps):
                        logits = base_model(points)

                        loss = softmax_entropy(logits).mean()

                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    base_model.eval()
                    if args.batch_size_tta > 1:
                        B, N, C = points.shape
                        # when running in episodic mode
                        points = points.view(args.batch_size_tta, -1, N, C)[0]
                        labels = labels.view(args.batch_size_tta, -1)[0]
                        # when running in online mode
                        logits = logits.view(args.batch_size_tta, -1, logits.shape[1])[
                            0
                        ]
                    if not args.online:
                        with torch.no_grad():
                            logits = base_model(points)
                else:
                    raise NotImplementedError(
                        f"cls_fixer_mode {args.cls_fixer_mode} not implemented"
                    )

                pred = logits.argmax(-1).view(-1)
                test_pred.append(pred.detach())
                test_label.append(labels.view(-1).detach())

                if idx % 50 == 0:
                    acc = compute_accuracy(test_pred, test_label, args)
                    print_log(
                        f"\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n", logger
                    )
                    acc_avg.append(acc.cpu())

            end_time = time.time()
            time_list.append(end_time - start_time)

            acc = compute_accuracy(test_pred, test_label, args)
            acc_list.append(acc.cpu().item())
            print_log(f"\nFinal Accuracy ::: {args.corruption} ::: {acc:.2f}%", logger)
            f_write.write(f"{acc:.3f}\n")
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                write_final_stats(f_write, time_list, acc_list, result_file_path)


# =====================
# Utilities for Evaluation
# =====================


def compute_accuracy(test_pred, test_label, args):
    test_pred = torch.cat(test_pred, dim=0)
    test_label = torch.cat(test_label, dim=0)

    if args.distributed:
        test_pred = dist_utils.gather_tensor(test_pred, args)
        test_label = dist_utils.gather_tensor(test_label, args)

    acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.0
    return acc


def write_final_stats(f_write, time_list, acc_list, result_file_path):
    f_write.write(
        " ".join(
            [
                f"{round(x, 3)}"
                for x in [
                    min(time_list),
                    max(time_list),
                    sum(time_list) / len(time_list),
                    np.var(time_list),
                ]
            ]
        )
        + "\n"
    )

    acc_list.append(sum(acc_list) / len(acc_list))
    acc_list = [f"{round(x, 3)}" for x in acc_list]
    f_write.write("\t".join(acc_list) + "\n")

    f_write.flush()
    f_write.close()
    print("Final Results Saved at:", result_file_path)
