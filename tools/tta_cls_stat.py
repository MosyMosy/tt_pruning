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


# Jensen-Shannon divergence between two univariate Gaussians
def kl_divergence_gaussians(mu_p, sigma_p, mu_q, sigma_q):
    term1 = torch.log(sigma_q / sigma_p)
    term2 = (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2)
    return term1 + term2 - 0.5


def jensen_shannon_divergence_gaussians(mu_p, sigma_p, mu_q, sigma_q):
    mu_m = (mu_p + mu_q) / 2
    sigma_m = torch.sqrt((sigma_p**2 + sigma_q**2) / 2)

    kl_p_m = kl_divergence_gaussians(mu_p, sigma_p, mu_m, sigma_m)
    kl_q_m = kl_divergence_gaussians(mu_q, sigma_q, mu_m, sigma_m)

    return 0.5 * (kl_p_m + kl_q_m)


def wasserstein_distance_gaussians(mu_p, sigma_p, mu_q, sigma_q):
    return torch.sqrt((mu_p - mu_q) ** 2 + (sigma_p - sigma_q) ** 2)


# =====================
# Custom Normalization Layers
# =====================


class BatchNorm1dWithClsTransform(nn.BatchNorm1d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        cls_mean=None,
        cls_std=None,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.cls_mean = cls_mean
        self.cls_std = cls_std

    def forward_(self, input):
        target_mean = input.mean(dim=0, keepdim=True)
        target_std = torch.sqrt(
            input.var(dim=0, keepdim=True, unbiased=False) + self.eps
        ).clamp_min(1e-3)

        source_mean = self.running_mean.unsqueeze(0)
        source_std = torch.sqrt(self.running_var + self.eps).unsqueeze(0)

        gamma = self.weight.unsqueeze(0)
        beta = self.bias.unsqueeze(0)

        a = source_std / target_std
        b = (source_mean - target_mean) / target_std

        gamma_new = gamma * a
        beta_new = beta + gamma * b

        normalized_target = (input - target_mean) / (target_std + self.eps)
        out = normalized_target * gamma + beta
        out = out * self.cls_std + self.cls_mean
        return out

    def forward__(self, input):
        target_std = torch.sqrt(
            input.var(dim=0, keepdim=True, unbiased=False) + self.eps
        ).clamp_min(1e-3)
        target_mean = input.mean(dim=0, keepdim=True)

        out = (input - target_mean) / (target_std + self.eps)
        diff = gaussian_difference(out).mean(dim=(0, 1), keepdim=True)

        out = out * self.weight + (self.bias - diff)
        return out

    def forward_main(self, input):
        in_dim = input.dim()
        if in_dim == 3:
            input = input.permute(0, 2, 1)

        target_std = torch.sqrt(
            input.var(dim=tuple(range(in_dim - 1)), keepdim=True, unbiased=False)
            + self.eps
        )
        target_mean = input.mean(dim=tuple(range(in_dim - 1)), keepdim=True)

        out = (input - target_mean) / (target_std + self.eps)
        out = out * self.weight + self.bias

        if in_dim == 3:
            out = out.permute(0, 2, 1)
        return out


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
        out = out * self.cls_std + self.cls_mean
        return out


# =====================
# Model Modification Utilities
# =====================


def replace_batchnorm(model, cls_norms_embedding):
    device = next(model.parameters()).device
    idx = 0

    def _recursive_replace(module):
        nonlocal idx
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm1d):
                new_module = BatchNorm1dWithClsTransform(
                    num_features=child.num_features,
                    eps=child.eps,
                    momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats,
                    cls_mean=cls_norms_embedding[idx].mean().item(),
                    cls_std=cls_norms_embedding[idx].std().item(),
                )
                setattr(module, name, new_module.to(device))
                idx += 1
            else:
                _recursive_replace(child)

    if isinstance(model, nn.DataParallel):
        model = model.module

    _recursive_replace(model)


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

    cls_pred, cls_norms_embedding, block_cls_embedding = (
        source_model.forward_norms_embedding()
    )
    block_cls_embedding = torch.stack(block_cls_embedding, dim=0).detach()

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
                    base_model.zero_grad()
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
