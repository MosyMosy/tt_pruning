import os
import time
from tools import builder
from utils import misc, dist_utils
from utils.logger import *
import datasets.tta_datasets as tta_datasets
from torch.utils.data import DataLoader
from utils.misc import *
import numpy as np
import torch.nn.functional as F
import utils.tent_shot as tent_shot_utils
import torch.optim as optim


level = [5]

from tqdm import tqdm


def gaussian_difference(x: torch.Tensor, eps: float = 1e-5):

    if len(x.shape) == 2:
        B, d = x.shape
    elif len(x.shape) == 3:
        B, N, d = x.shape
    else:
        raise ValueError("Input tensor must be 2D or 3D")

    x_flat = x.view(-1, d)  # shape: (B*N, d)

    # Allocate result tensors
    x_sorted, sort_idx = x_flat.sort(dim=1)
    reverse_idx = sort_idx.argsort(dim=1)

    # Compute per-row mean and std
    mean = x_flat.mean(dim=1, keepdim=True)
    std = x_flat.std(dim=1, keepdim=True)

    # Generate Gaussian targets (same for all rows, normalized)
    z = torch.linspace(eps, 1 - eps, steps=d, device=x.device)
    target_gauss = torch.sqrt(torch.tensor(2.0, device=x.device)) * torch.erfinv(
        2 * z - 1
    )  # (d,)

    # Scale and shift to each row's mean and std
    target_gauss = target_gauss.unsqueeze(0) * std + mean  # (B*N, d)

    # Compute delta and reverse sort
    delta_sorted = x_sorted - target_gauss
    delta = torch.gather(delta_sorted, dim=1, index=reverse_idx)

    return delta.view(*x.shape)


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

        # Learnable cls transformation parameters (one per channel)
        self.cls_mean = cls_mean
        self.cls_std = cls_std

    def forward_(self, input):
        # Normal BatchNorm output
        # out = super().forward(input)  # shape: (B, C, N) or (B, C)

        target_var = input.var(dim=0, keepdim=True, unbiased=False)
        target_mean = input.mean(dim=0, keepdim=True)
        target_std = torch.sqrt(target_var + self.eps).clamp_min(1e-3)

        # 2. Source stats
        source_mean = self.running_mean.unsqueeze(0)
        source_std = torch.sqrt(self.running_var + self.eps).unsqueeze(0)

        # 3. Affine reparam (core idea preserved)
        a = source_std / target_std
        b = (source_mean - target_mean) / target_std

        gamma = self.weight.unsqueeze(0)
        beta = self.bias.unsqueeze(0)

        gamma_new = gamma * a
        beta_new = beta + gamma * b

        normalized_target = (input - target_mean) / (target_std + self.eps)

        out = normalized_target * gamma + beta

        # fix the output using the cls-specific transformation
        out = out * self.cls_std + self.cls_mean

        return out

    def forward__(self, input):

        target_var = input.var(dim=0, keepdim=True, unbiased=False)
        target_mean = input.mean(dim=0, keepdim=True)
        target_std = torch.sqrt(target_var + self.eps).clamp_min(1e-3)

        out = (input - target_mean) / (target_std + self.eps)

        # out = out * self.weight + self.bias
        diff = gaussian_difference(out)
        diff_mean = diff.mean(dim=(0, 1), keepdim=True)
        out = out * self.weight + (self.bias - diff_mean)

        return out

    def forward_main(self, input):
        in_dim = input.dim()
        if in_dim == 3:
            input = input.permute(0, 2, 1)

        target_var = input.var(
            dim=tuple(range(in_dim - 1)), keepdim=True, unbiased=False
        )
        target_mean = input.mean(dim=tuple(range(in_dim - 1)), keepdim=True)
        target_std = torch.sqrt(target_var + self.eps)

        out = (input - target_mean) / (target_std + self.eps)

        out = out * self.weight + self.bias

        # out = out * self.cls_std + self.cls_mean # has no effect on scanobject and modelnet. Degrades on shapenet

        if in_dim == 3:
            out = out.permute(0, 2, 1)
        return out

    def forward_failed(self, input):

        k = 1.5

        target_var = input.var(dim=0, keepdim=True, unbiased=False)
        target_mean = input.mean(dim=0, keepdim=True)
        target_std = torch.sqrt(target_var + self.eps).clamp_min(1e-3)
        out = (input - target_mean) / (target_std + self.eps)

        mask = (out >= target_mean - k * target_std) & (
            out <= target_mean + k * target_std
        )

        out = out * (~mask).float()

        # out = out * self.weight + self.bias

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

        # Register cls-specific transformation parameters
        shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )
        self.cls_mean = cls_mean
        self.cls_std = cls_std
        self.out_prototype = out_prototype

    def forward_fail(self, input):
        # This didn't work for some reason

        input_mean = input.mean(dim=-1, keepdim=True)
        input_std = (input.var(dim=-1, keepdim=True) + self.eps).sqrt()
        out = (input - input_mean) / (input_std + self.eps)

        out_mean = out.mean(dim=(0, 1), keepdim=True)
        out_std = (out.var(dim=(0, 1), keepdim=True) + self.eps).sqrt()
        out = (out - out_mean) / (out_std + self.eps)
        # out = out * self.weight + self.bias

        return out

    def forward___(self, input):

        input_mean = input.mean(dim=-1, keepdim=True)
        input_std = (input.var(dim=-1, keepdim=True) + self.eps).sqrt()
        out = (input - input_mean) / (input_std + self.eps)

        diff = gaussian_difference(out)
        diff_mean = diff.mean(dim=(0, 1), keepdim=True)
        out = out * self.weight + (self.bias - diff_mean)

        return out

    def forward_(self, input):

        input_mean = input.mean(dim=-1, keepdim=True)
        input_std = (input.var(dim=-1, keepdim=True) + self.eps).sqrt()
        out = (input - input_mean) / (input_std + self.eps)

        out = out * self.weight + self.bias

        cls_out_mean = out[:, 0:1].mean()
        cls_out_std = (out[:, 0:1].var() + self.eps).sqrt()

        out = (out - out.mean(dim=-1, keepdim=True)) / (
            out.std(dim=-1, keepdim=True) + self.eps
        )
        out = out * cls_out_std + cls_out_mean

        return out

    def forward(self, input):
        # Standard LayerNorm
        out = super().forward(input)
        # input_mean = input.mean(dim=-1, keepdim=True)
        # input_std = (input.var(dim=-1, keepdim=True) + self.eps).sqrt()
        # out = (input - input_mean) / (input_std + self.eps)
        # out = out * self.weight + self.bias

        out = out * self.cls_std + self.cls_mean

        return out

    def forward_bn(self, input):

        input_mean = input.mean(dim=-1, keepdim=True)
        input_std = (input.var(dim=-1, keepdim=True) + self.eps).sqrt()
        out = (input - input_mean) / (input_std + self.eps)

        out = out + out.mean(dim=(0, 1), keepdim=True)

        out = out * self.weight + self.bias

        return out


def replace_batchnorm(model, cls_norms_embedding):

    device = next(model.parameters()).device
    idx = 0

    def _recursive_replace(module):
        nonlocal idx
        for name, child in module.named_children():

            if isinstance(child, nn.BatchNorm1d):
                if idx >= 0:
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
                # return
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

    norm_idx = 0  # index over all norm layers

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


def load_tta_dataset(args, config):
    # we have 3 choices - every tta_loader returns only point and labels
    root = config.tta_dataset_path  # being lazy - 1

    if config.dataset.name == "modelnet":
        if args.corruption == "clean":
            inference_dataset = tta_datasets.ModelNet_h5(args, root)
            tta_loader = DataLoader(
                dataset=inference_dataset,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                drop_last=False,
            )
        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)
            tta_loader = DataLoader(
                dataset=inference_dataset,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                drop_last=False,
            )

    elif config.dataset.name == "scanobject":
        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            drop_last=False,
        )

    elif config.dataset.name == "shapenetcore":
        inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)
        tta_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            drop_last=False,
        )

    else:
        raise NotImplementedError(f"TTA for {args.tta} is not implemented")

    print(f"\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n")

    return tta_loader


def load_clean_dataset(args, config):
    (train_sampler, train_dataloader) = builder.dataset_builder(
        args, config.dataset.train
    )
    return train_dataloader


def load_base_model(args, config, load_part_seg=False, pretrained=True):
    base_model = builder.model_builder(config.model)
    if pretrained:
        base_model.load_model_from_ckpt(args.ckpts)
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            # print_log("Using Synchronized BatchNorm ...", logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
        # print_log("Using Distributed Data parallel ...", logger=logger)
    else:
        # print_log("Using Data parallel ...", logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    return base_model


def reset_model(args, config, source_model, cls_norms_embedding):
    model = load_base_model(args, config, pretrained=False)
    if "cls-fixer" in args.cls_fixer_mode:
        # replace_batchnorm(base_model.module.class_head, cls_norms_embedding[-2:])
        replace_layernorm_with_cls_versions(model, cls_norms_embedding)
    model.load_state_dict(source_model.state_dict())

    model.requires_grad_(False)  # freeze the model
    if "update_tent" in args.cls_fixer_mode:
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) or isinstance(
                m, torch.nn.LayerNorm
            ):
                m.requires_grad_(True)
                for n_p, p in m.named_parameters():
                    if n_p in [
                        "weight",
                        "bias",
                    ]:  # weight is scale gamma, bias is shift beta
                        params.append(p)
                        names.append(f"{nm}.{n_p}")

        optimizer = optim.Adam(
            params,
            lr=args.LR,
            betas=(args.BETA, 0.999),
            weight_decay=args.WD,
        )
    else:
        optimizer = None

    return model, optimizer


def runner(args, config):
    dataset_name = config.dataset.name
    logger = get_logger(args.log_name)

    source_model = load_base_model(args, config)
    source_model.eval()

    resutl_file_path = os.path.join(
        "results_final_tta/",
        args.method,
        f"{args.exp_name}_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
    )

    eval(
        args,
        config,
        logger,
        source_model,
        resutl_file_path,
    )


@torch.jit.script
def softmax_entropy(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def eval(
    args,
    config,
    logger,
    source_model,
    resutl_file_path,
):
    time_list = []
    acc_list = []
    _, cls_norms_embedding = source_model.module.forward_norms_embedding()
    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            start_time = time.time()
            acc_sliding_window = list()
            acc_avg = list()
            if args.corruption == "clean":
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            # if corr_id not in [2]:
            #     continue

            if (
                "f_write" not in locals()
            ):  # for saving results for easy copying to google sheet
                f_write = get_writer_to_all_result(
                    args, config, custom_path=resutl_file_path
                )
                f_write.write(f"All Corruptions: {corruptions}" + "\n\n")
                f_write.write(
                    f"TTA Results for Dataset: {config.dataset.name}" + "\n\n"
                )
                f_write.write(f"Checkpoint Used: {args.ckpts}" + "\n\n")
                f_write.write(f"Corruption LEVEL: {args.severity}" + "\n\n")

            tta_loader = load_tta_dataset(args, config)
            test_pred = []
            test_label = []
            entropy_list = []

            if args.online:
                base_model, optimizer = reset_model(
                    args, config, source_model, cls_norms_embedding
                )

            for idx, (data, labels) in enumerate(tta_loader):
                # now inferring on this one sample
                a = iter(tta_loader)
                data, labels = next(a)
                if data.shape[0] == 1:
                    data = data.repeat(2, 1, 1)
                    labels = labels.repeat(2, 1)
                # reset batchnorm running stats

                if ~args.online:
                    base_model, optimizer = reset_model(
                        args, config, source_model, cls_norms_embedding
                    )

                base_model.eval()

                if data.shape[0] < args.batch_size:
                    n_repeat = (args.batch_size + data.shape[0] - 1) // data.shape[
                        0
                    ]  # Ceiling division
                    data = data.repeat(n_repeat, 1, 1)[: args.batch_size]

                if labels.shape[0] < args.batch_size:
                    n_repeat = (args.batch_size + labels.shape[0] - 1) // labels.shape[
                        0
                    ]
                    labels = labels.repeat(n_repeat, 1)[: args.batch_size]

                # if args.distributed:
                #     data = dist_utils.scatter_tensor(data, args)
                #     labels = dist_utils.scatter_tensor(labels, args)
                # else:
                #     data = data.cuda()
                #     labels = labels.cuda()
                # # data = data.cuda()

                # reset batchnorm running stats
                if args.BN_reset:
                    for m in base_model.modules():
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            # m.track_running_stats = False
                            m.running_mean = None  # for original implementation of tent
                            m.running_var = None  # for original implementation of tent

                points = data.cuda()
                labels = labels.cuda()
                points = [
                    misc.fps(points, config.npoints, random_start_point=True)
                    for _ in range(args.batch_size_tta)
                ]
                points = torch.cat(points, dim=0)

                labels = [labels for _ in range(args.batch_size_tta)]
                labels = torch.cat(labels, dim=0)

                if "source_only" in args.cls_fixer_mode:
                    base_model.eval()
                    with torch.no_grad():
                        logits = base_model(
                            points,
                        )
                elif "update_tent" in args.cls_fixer_mode:
                    base_model.train()
                    for it in range(args.grad_steps):
                        logits = base_model(points)
                        loss = softmax_entropy(logits).mean()

                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    if not args.online:
                        base_model.eval()
                        if args.batch_size_tta > 1:
                            B, N, C = points.shape
                            points = points.view(args.batch_size_tta, -1, N, C)[0]
                            labels = labels.view(args.batch_size_tta, -1)[0]
                        with torch.no_grad():
                            logits = base_model(
                                points,
                            )
                else:
                    raise NotImplementedError(
                        f"cls_fixer_mode {args.cls_fixer_mode} not implemented"
                    )

                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 50 == 0:
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                        test_label_ = dist_utils.gather_tensor(test_label_, args)

                    acc = (
                        (test_pred_ == test_label_).sum()
                        / float(test_label_.size(0))
                        * 100.0
                    )

                    print_log(
                        f"\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n",
                        logger=logger,
                    )

                    acc_avg.append(acc.cpu())
            end_time = time.time()
            time_list.append(end_time - start_time)
            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.0
            acc_list.append(acc.cpu().item())
            print_log(
                f"\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n",
                logger=logger,
            )
            f_write.write(" ".join([str(round(float(xx), 3)) for xx in [acc]]) + "\n")
            # f_write.write(
            #     " ".join([str(round(float(xx), 3)) for xx in [torch.stack(entropy_list).mean().item()]]) + "\n"
            # )
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                # write min, max, and average, variance,  of times
                f_write.write(
                    " ".join(
                        [
                            str(round(float(xx), 3))
                            for xx in [
                                min(time_list),
                                max(time_list),
                                sum(time_list) / len(time_list),
                                np.var(time_list),
                            ]
                        ]
                    )
                    + "\n"
                )

                acc_list.append(
                    sum(acc_list) / len(acc_list),
                )
                acc_list = [str(round(float(xx), 3)) for xx in acc_list]
                f_write.write("\t".join(acc_list) + "\n")

                f_write.flush()
                f_write.close()

                print(
                    f"Final Results Saved at:",
                    resutl_file_path,
                )
