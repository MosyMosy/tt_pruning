import os
import time
from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
import datasets.tta_datasets as tta_datasets
from torch.utils.data import DataLoader
from utils.rotnet_utils import rotate_batch
import utils.tent_shot as tent_shot_utils
import utils.t3a as t3a_utils
from utils.misc import *
import torch.optim as optim


level = [5]

from tqdm import tqdm


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
                drop_last=True,
            )
        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)
            tta_loader = DataLoader(
                dataset=inference_dataset,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                drop_last=True,
            )

    elif config.dataset.name == "scanobject":
        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            drop_last=True,
        )

    elif config.dataset.name == "shapenetcore":
        inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)
        tta_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            drop_last=True,
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


def load_base_model(args, config, logger, load_part_seg=False, pretrained=True):
    base_model = builder.model_builder(config.model)
    if pretrained:
        base_model.load_model_from_ckpt(args.ckpts)
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log("Using Synchronized BatchNorm ...", logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
        print_log("Using Distributed Data parallel ...", logger=logger)
    else:
        print_log("Using Data parallel ...", logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    return base_model


def eval_source(args, config):
    npoints = config.npoints
    logger = get_logger(args.log_name)
    dataset_name = config.dataset.name

    resutl_file_path = os.path.join(
        "results_final_tta/",
        args.method,
        f"{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
    )

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            if corr_id == 0:
                f_write = get_writer_to_all_result(
                    args, config, custom_path=resutl_file_path
                )  # for saving results for easy copying to google sheet
                f_write.write(f"All Corruptions: {corruptions}" + "\n\n")
                f_write.write(
                    f"Source Only Results for Dataset: {dataset_name}" + "\n\n"
                )
                f_write.write(f"Check Point: {args.ckpts}" + "\n\n")

            base_model = load_base_model(args, config, logger)
            print("Testing Source Performance...")
            test_pred = []
            test_label = []
            base_model.eval()

            inference_loader = load_tta_dataset(args, config)

            with torch.no_grad():
                for idx_inference, (data, labels) in enumerate(inference_loader):
                    points = data.cuda()
                    points = misc.fps(points, npoints)
                    label = labels.cuda()

                    points = points.cuda()
                    labels = label.cuda()
                    logits = base_model(points)
                    target = labels.view(-1)
                    pred = logits.argmax(-1).view(-1)

                    test_pred.append(pred.detach())
                    test_label.append(target.detach())

                test_pred = torch.cat(test_pred, dim=0)
                test_label = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred = dist_utils.gather_tensor(test_pred, args)
                    test_label = dist_utils.gather_tensor(test_label, args)

                acc = (
                    (test_pred == test_label).sum() / float(test_label.size(0)) * 100.0
                )
                print(
                    f"Source Peformance ::: Corruption ::: {args.corruption} ::: {acc}"
                )

                f_write.write(
                    " ".join([str(round(float(xx), 3)) for xx in [acc]]) + "\n"
                )
                f_write.flush()
                if corr_id == len(corruptions) - 1:
                    f_write.close()
                    print(
                        f"Final Results Saved at:",
                        resutl_file_path,
                    )


def source_prune(args, config):
    dataset_name = config.dataset.name
    npoints = config.npoints
    logger = get_logger(args.log_name)

    source_model = load_base_model(args, config, logger)
    source_model.eval()

    if args.method in ["source_prune", "source_prune_analyze"]:
        clean_intermediates_path = (
            f"intermediate_features/{dataset_name}_clean_intermediates.pth"
        )
        if os.path.exists(clean_intermediates_path):
            clean_intermediates = torch.load(clean_intermediates_path)
            clean_intermediates_mean, clean_intermediates_std = (
                clean_intermediates["mean"],
                clean_intermediates["std"],
            )
        else:
            clean_intermediates_mean, clean_intermediates_std = (
                generate_intermediate_embeddings(args, config, source_model)
            )
            torch.save(
                {
                    "mean": clean_intermediates_mean.cpu(),
                    "std": clean_intermediates_std.cpu(),
                },
                clean_intermediates_path,
            )
        clean_intermediates_mean = clean_intermediates_mean.cuda()
        clean_intermediates_std = clean_intermediates_std.cuda()

        resutl_file_path = os.path.join(
            "results_final_tta/",
            args.method,
            f"{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        )

    if args.method == "source_prune":
        eval_source_prune(
            args,
            config,
            logger,
            source_model,
            clean_intermediates_mean,
            clean_intermediates_std,
            resutl_file_path,
        )
    elif args.method == "source_prune_analyze":
        source_prune_analyze(
            args,
            config,
            logger,
            source_model,
            clean_intermediates_mean,
            clean_intermediates_std,
            resutl_file_path,
        )


def generate_intermediate_embeddings(args, config, source_model):
    def update_stats(x, count, mean, M2):
        count += 1
        delta = x - mean
        mean = mean + delta / count
        delta2 = x - mean
        M2 = M2 + delta * delta2
        return count, mean, M2

    def finalize_stats(count, mean, M2):
        if count < 2:
            # Not enough data points to compute variance
            return mean, torch.full_like(M2, float("nan"))
        variance = M2 / (count - 1)
        std = torch.sqrt(variance)
        return mean, std

    clean_dataloader = load_clean_dataset(args, config)
    count = 0
    mean = torch.zeros(12, 384, dtype=float)
    M2 = torch.zeros(12, 384, dtype=float)
    for i, (_, _, data) in tqdm(
        enumerate(clean_dataloader), total=len(clean_dataloader)
    ):
        points = data[0].cuda()
        points = misc.fps(points, config.npoints)
        with torch.no_grad():
            intermediates = source_model.module.forward_out_intermediate(points)
        intermediates = torch.stack(
            intermediates, dim=0
        )  # (num_layers, batch_size * tokens, emb_dim)

        for i in range(intermediates.shape[1]):
            x = intermediates[:, i, :]  # x has shape (L, C)
            count, mean, M2 = update_stats(x, count, mean, M2)

    return finalize_stats(count, mean, M2)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def setup_tent_optimizer(model, args):
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.requires_grad_(True)
            m.track_running_stats = False # for original implementation this is False
            m.running_mean = None # for original implementation uncomment this
            m.running_var = None # for original implementation uncomment this
        # if isinstance(m, torch.nn.LayerNorm):
        #     m.requires_grad_(True)
        
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm): # or isinstance(m, torch.nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale gamma, bias is shift beta
                    params.append(p)
                    names.append(f"{nm}.{np}")
                        
    optimizer = optim.Adam(params,
                      lr=args.tent_LR,
                      betas=(args.tent_BETA, 0.999),
                      weight_decay=args.tent_WD)
    return model, optimizer

def eval_source_prune(
    args,
    config,
    logger,
    source_model,
    clean_intermediates_mean,
    clean_intermediates_std,
    resutl_file_path,
):
    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            acc_sliding_window = list()
            acc_avg = list()
            if args.corruption == "clean":
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            # if corr_id not in [ 2]:
            #     continue

            if corr_id == 0:  # for saving results for easy copying to google sheet
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
            total_batches = len(tta_loader)
            test_pred = []
            test_label = []

            if args.online:
                base_model = load_base_model(args, config, logger, pretrained=False)
                base_model.load_state_dict(source_model.state_dict())
                base_model, optimizer =setup_tent_optimizer(base_model, args)
                args.grad_steps = 1

            for idx, (data, labels) in enumerate(tta_loader):
                losses = AverageMeter(["Reconstruction Loss"])

                if not args.online:
                    # base_model = load_base_model(args, config, logger)
                    base_model = load_base_model(args, config, logger, pretrained=False)
                    base_model.load_state_dict(source_model.state_dict())
                    base_model, optimizer  =setup_tent_optimizer(base_model, args)

                # TTA Loop (for N grad steps)
                if args.train_with_prune:
                    base_model.zero_grad()
                    base_model.train()
                    if args.disable_bn_adaptation:  # disable statistical alignment
                        for m in base_model.modules():
                            if (
                                isinstance(m, nn.BatchNorm2d)
                                or isinstance(m, nn.BatchNorm1d)
                                or isinstance(m, nn.BatchNorm3d)
                            ):
                                m.eval()
                    for grad_step in range(args.grad_steps):
                        points = data.cuda()
                        # make a batch
                        points = [
                            misc.fps(points, config.npoints, random_start_point=True)
                            for _ in range(args.batch_size_tta)
                        ]
                        points = torch.cat(points, dim=0)
                        
                        if idx % args.stride_step == 0 or idx == len(tta_loader) - 1:
                            logits = base_model.module.forward_source_prune(
                                points,
                                source_stats=(clean_intermediates_mean, clean_intermediates_std),
                            )
                            # tent loss calculation to minimize the entropy of the logits
                            loss = softmax_entropy(logits)                            
                            
                            loss = loss.mean()
                            loss.backward()
                            optimizer.step()
                            base_model.zero_grad()
                            optimizer.zero_grad()
                        else:
                            continue

                        if args.distributed:
                            loss = dist_utils.reduce_tensor(loss, args)
                            losses.update([loss.item() * 1000])
                        else:
                            losses.update([loss.item() * 1000])

                        print_log(
                            f"[TEST - {args.corruption}], Sample - {idx} / {total_batches},"
                            f"GradStep - {grad_step} / {args.grad_steps},"
                            f"Reconstruction Loss {[l for l in losses.val()]}",
                            logger=logger,
                        )

                # now inferring on this one sample
                base_model.eval()

                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, config.npoints)
                with torch.no_grad():
                    logits = base_model.module.forward_source_prune(
                        points,
                        source_stats=(clean_intermediates_mean, clean_intermediates_std),
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
            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.0
            print_log(
                f"\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n",
                logger=logger,
            )
            f_write.write(" ".join([str(round(float(xx), 3)) for xx in [acc]]) + "\n")
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(
                    f"Final Results Saved at:",
                    resutl_file_path,
                )


def source_prune_analyze(
    args,
    config,
    logger,
    source_model,
    clean_intermediates_mean,
    clean_intermediates_std,
    resutl_file_path,
):
    percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            acc_sliding_window = list()
            if args.corruption == "clean":
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            # if corr_id not in [0, 1]:
            #     continue

            if corr_id == 0:
                f_write = get_writer_to_all_result(
                    args, config, custom_path=resutl_file_path
                )
                f_write.write(f"All Corruptions: {corruptions}" + "\n\n")
                f_write.write(
                    f"TTA Results for Dataset: {config.dataset.name}" + "\n\n"
                )
                f_write.write(f"Checkpoint Used: {args.ckpts}" + "\n\n")
                f_write.write(f"Corruption LEVEL: {args.severity}" + "\n\n")

                f_write_analyzr = get_writer_to_all_result(
                    args, config, custom_path=resutl_file_path + "_analyzer.txt"
                )

            f_write_analyzr.write(f" ------------ {args.corruption} --------------- \n")
            f_write.write(f" ------------ {args.corruption} --------------- \n")

            for perc in percentiles:
                dist_min = []
                dist_max = []
                dist_mean = []
                threshold = []

                # if corr_id == 0:  # for saving results for easy copying to google sheet

                tta_loader = load_tta_dataset(args, config)
                total_batches = len(tta_loader)
                test_pred = []
                test_label = []

                if args.online:
                    base_model = load_base_model(args, config, logger, pretrained=False)
                    base_model.load_state_dict(source_model.state_dict())
                    optimizer = builder.build_opti_sche(base_model, config)[0]
                    args.grad_steps = 1

                for idx, (data, labels) in enumerate(tta_loader):
                    losses = AverageMeter(["Reconstruction Loss"])

                    if not args.online:
                        # base_model = load_base_model(args, config, logger)
                        base_model = load_base_model(
                            args, config, logger, pretrained=False
                        )
                        base_model.load_state_dict(source_model.state_dict())
                        optimizer = builder.build_opti_sche(base_model, config)[0]
                    base_model.zero_grad()
                    base_model.train()
                    if args.disable_bn_adaptation:  # disable statistical alignment
                        for m in base_model.modules():
                            if (
                                isinstance(m, nn.BatchNorm2d)
                                or isinstance(m, nn.BatchNorm1d)
                                or isinstance(m, nn.BatchNorm3d)
                            ):
                                m.eval()
                    else:
                        pass

                    # TTA Loop (for N grad steps)
                    if args.train_with_prune:
                        for grad_step in range(args.grad_steps):
                            points = data.cuda()
                            points = misc.fps(points, args.npoints)

                            # make a batch
                            if (
                                idx % args.stride_step == 0
                                or idx == len(tta_loader) - 1
                            ):
                                points = [points for _ in range(args.batch_size_tta)]
                                points = torch.squeeze(torch.vstack(points))

                                loss_recon, loss_p_consistency, loss_regularize = (
                                    base_model(points)
                                )
                                loss = loss_recon + (
                                    args.alpha * loss_regularize
                                )  # + (0.0001 * loss_p_consistency)
                                loss = loss.mean()
                                loss.backward()
                                optimizer.step()
                                base_model.zero_grad()
                                optimizer.zero_grad()
                            else:
                                continue

                            if args.distributed:
                                loss = dist_utils.reduce_tensor(loss, args)
                                losses.update([loss.item() * 1000])
                            else:
                                losses.update([loss.item() * 1000])

                            print_log(
                                f"[TEST - {args.corruption}], Sample - {idx} / {total_batches},"
                                f"GradStep - {grad_step} / {args.grad_steps},"
                                f"Reconstruction Loss {[l for l in losses.val()]}",
                                logger=logger,
                            )

                    # now inferring on this one sample
                    base_model.eval()
                    points = data.cuda()
                    labels = labels.cuda()
                    points = misc.fps(points, args.npoints)

                    logits, (dist_min_, dist_max_, dist_mean_, threshold_) = (
                        base_model.module.forward_analyze(
                            points,
                            source_stats=(
                                clean_intermediates_mean,
                                clean_intermediates_std,
                            ),
                            threshold_percentile=perc,
                        )
                    )
                    target = labels.view(-1)
                    pred = logits.argmax(-1).view(-1)
                    dist_min.append(dist_min_.detach().cpu())
                    dist_max.append(dist_max_.detach().cpu())
                    dist_mean.append(dist_mean_.detach().cpu())
                    threshold.append(threshold_.detach().cpu())

                    test_pred.append(pred.detach())
                    test_label.append(target.detach())

                    if (idx + 1) % 50 == 0:
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

                test_pred = torch.cat(test_pred, dim=0)
                test_label = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred = dist_utils.gather_tensor(test_pred, args)
                    test_label = dist_utils.gather_tensor(test_label, args)

                acc = (
                    (test_pred == test_label).sum() / float(test_label.size(0)) * 100.0
                )
                dist_min = torch.stack(dist_min, dim=0).mean(dim=0)
                dist_max = torch.stack(dist_max, dim=0).mean(dim=0)
                dist_mean = torch.stack(dist_mean, dim=0).mean(dim=0)
                threshold = torch.stack(threshold, dim=0).mean(dim=0)
                f_write_analyzr.write(
                    f"{dist_min.item()}, {dist_max.item()}, {dist_mean.item()}, {threshold.item()}, {acc.item()}\n"
                )

                print_log(
                    f"\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n",
                    logger=logger,
                )
                f_write.write(
                    " ".join([str(round(float(xx), 3)) for xx in [acc]]) + "\n"
                )

                f_write.flush()
                f_write_analyzr.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()
                f_write_analyzr.close()

                print(
                    f"Final Results Saved at:",
                    resutl_file_path,
                )


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y
