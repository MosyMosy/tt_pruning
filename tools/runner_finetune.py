import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import os
import numpy as np
from datasets import data_transforms, tta_datasets
# from pointnet2_ops import pointnet2_utils
from pytorch3d.ops import sample_farthest_points # pytorch3d
from torchvision import transforms
from torch.utils.data import DataLoader

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        # data_transforms.RandomHorizontalFlip(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                               builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            # fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = sample_farthest_points(points, K=point_all)[1]
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            # points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            points = torch.gather(points, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3)) # pytorch3d
            
            # import pdb; pdb.set_trace()
            points = train_transforms(points)

            ret = base_model(points)

            loss, acc = base_model.module.get_loss_acc(ret, label)

            _loss = loss

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if idx % 10 == 0:
            #     print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
            #                 (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
            #                 ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
                print_log(
                    "--------------------------------------------------------------------------------------------",
                    logger=logger)
            if args.vote:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config,
                                                 logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote,
                                                'ckpt-best_vote', args, logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            # fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            fps_idx_raw = sample_farthest_points(points_raw, K=point_all) # pytorch3d
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                # points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                #                                         fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                points = torch.gather(points_raw, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3)) # pytorch3d

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)


def eval_tta(args, config, test_source = True, train_writer=None):
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm

    npoints = config.npoints
    logger = get_logger(args.log_name)
    corruptions = [
        # 'clean',
        'uniform', 'gaussian', 'background', 'impulse', 'upsampling', 'distortion_rbf', 'distortion_rbf_inv',
        # 'density', 'density_inc', 'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar'
    ]
    level = 5
    for corr_id, corruption_type in enumerate(corruptions):
        all_corr_acc = []

        args.corruption = corruption_type
        args.severity = level
        base_model = builder.model_builder(config.model)
        base_model.load_model_from_ckpt(args.ckpts)
        if args.use_gpu:
            base_model.to(args.local_rank)
        if args.distributed:
            if args.sync_bn:
                base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
                print_log('Using Synchronized BatchNorm ...', logger=logger)
            base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[
                args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
            print_log('Using Distributed Data parallel ...', logger=logger)
        else:
            print_log('Using Data parallel ...', logger=logger)
            base_model = nn.DataParallel(base_model).cuda()

        if corruption_type == 'clean':
            split = "clean"
        else:
            split = corruption_type + '_' + str(level)
        # args.split = split
        if test_source and args.corruption != 'clean':
            print('Testing Source Performance...')
            inference_dataset = tta_datasets.ModelNet40C(args)
            inference_loader = DataLoader(dataset=inference_dataset, batch_size=32)
            if args.corruption == 'uniform':  # for saving results for easy copying to google sheet
                f_write, logtime = get_writer_to_all_result(args, custom_path='results_final/')
            test_pred = []
            test_label = []
            base_model.eval()
            with torch.no_grad():
                for idx_inference, (data_inference, labels) in enumerate(inference_loader):
                    points = data_inference.cuda()
                    labels = labels.cuda()
                    points = misc.fps(points, npoints)
                    logits = base_model.module.classification_only(points, only_unmasked=False)
                    target = labels.view(-1)
                    pred = logits.argmax(-1).view(-1)

                    test_pred.append(pred.detach())
                    test_label.append(target.detach())

                test_pred = torch.cat(test_pred, dim=0)
                test_label = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred = dist_utils.gather_tensor(test_pred, args)
                    test_label = dist_utils.gather_tensor(test_label, args)

                acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
                # all_corr_acc.append(acc)
                print(f'Source Peformance ::: Corruption ::: {args.corruption} ::: {acc}')
                f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
                f_write.flush()
                if corr_id == len(corruptions) - 1:
                    f_write.close()
                    print(f'Final Results Saved at:', os.path.join('results_final/', f'{logtime}_results.txt'))
                    if train_writer is not None:
                        train_writer.close()

        elif args.corruption == 'clean' and not test_source:

            clean_dataset = tta_datasets.ModelNetClean(args)
            clean_loader = DataLoader(dataset=clean_dataset, batch_size=args.batch_size)
            test_pred = []
            test_label = []
            base_model.eval()
            with torch.no_grad():
                for idx_inference, (data_inference, labels) in enumerate(clean_loader):
                    points = data_inference.cuda()
                    labels = labels.cuda()
                    points = misc.fps(points, npoints)
                    logits = base_model.module.classification_only(points, only_unmasked=False)
                    target = labels.view(-1)
                    pred = logits.argmax(-1).view(-1)

                    test_pred.append(pred.detach())
                    test_label.append(target.detach())

                test_pred = torch.cat(test_pred, dim=0)
                test_label = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred = dist_utils.gather_tensor(test_pred, args)
                    test_label = dist_utils.gather_tensor(test_label, args)

                acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
                print(f'Clean Accuracy ::: {acc}')

        elif args.corruption in corruptions and not test_source and not args.corruption == 'clean':

            dataset = tta_datasets.ModelNet40C(args)
            tta_loader = DataLoader(dataset=dataset, batch_size=32, drop_last=False)
            optimizer = builder.build_opti_sche(base_model, config)[0]
            base_model.zero_grad()
            total_batches = len(tta_loader)
            losses = AverageMeter(['Reconstruction Loss'])
            inference_dataset = tta_datasets.ModelNet40C(args)
            inference_loader = DataLoader(dataset=inference_dataset, batch_size=32)

            if args.corruption == 'uniform':  # for saving results for easy copying to google sheet
                f_write, logtime = get_writer_to_all_result(args, custom_path='results_final/')
            for idx, (data, _) in enumerate(tta_loader):
                test_pred = []
                test_label = []
                base_model.eval()
                with torch.no_grad():
                    for idx_inference, (data_inference, labels) in enumerate(inference_loader):
                        points = data_inference.cuda()
                        labels = labels.cuda()
                        points = misc.fps(points, npoints)
                        logits = base_model.module.classification_only(points, only_unmasked=False)
                        target = labels.view(-1)
                        pred = logits.argmax(-1).view(-1)

                        test_pred.append(pred.detach())
                        test_label.append(target.detach())

                    test_pred = torch.cat(test_pred, dim=0)
                    test_label = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred = dist_utils.gather_tensor(test_pred, args)
                        test_label = dist_utils.gather_tensor(test_label, args)

                    acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
                    all_corr_acc.append(acc)

                    ### Start TTA ###

                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                points = data.cuda()
                points = misc.fps(points, npoints)

                for _ in range(args.grad_steps):
                    loss = base_model(points)[0]

                    try:
                        loss.backward()
                        # print("Using one GPU")
                    except:
                        loss = loss.mean()
                        loss.backward()

                    optimizer.step()
                    base_model.zero_grad()
                    optimizer.zero_grad()

                if args.distributed:
                    loss = dist_utils.reduce_tensor(loss, args)
                    losses.update([loss.item() * 1000])
                else:
                    losses.update([loss.item() * 1000])

                if args.distributed:
                    torch.cuda.synchronize()

                if train_writer is not None:
                    train_writer.add_scalar(f'{split} - Loss/Batch/Loss', loss.item(), idx)
                    train_writer.add_scalar(f'{split} - Loss/Batch/TestAcc', acc.item(), idx)
                    train_writer.add_scalar(f'{split} - Loss/Batch/LR', optimizer.param_groups[0]['lr'], idx)
                current_lr = optimizer.param_groups[0]['lr']
                print_log(
                    f'[TEST - {split}] acc = {acc:.4f}, loss {[l for l in losses.val()]} before adapting on batch {idx}/{total_batches}, lr = {current_lr}',
                    logger=logger)

                final_acc = max(all_corr_acc)

            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [final_acc]]) + '\n')
            f_write.flush()
            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('results_final/', f'{logtime}_results.txt'))

                if train_writer is not None:
                    train_writer.close()




def load_base_model(args, config, logger):
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts)
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[
            args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    return base_model


def eval_source(args, config):
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    npoints = config.npoints
    logger = get_logger(args.log_name)
    corruptions = [
        'clean',
        'uniform', 'gaussian', 'background', 'impulse', 'upsampling', 'distortion_rbf', 'distortion_rbf_inv',
        'density', 'density_inc', 'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar'
    ]
    level = 5
    for corr_id, corruption_type in enumerate(corruptions):
        if args.corruption == 'uniform':  # for saving results for easy copying to google sheet
            f_write, logtime = get_writer_to_all_result(args, custom_path='results_final/')
        args.corruption = corruption_type
        args.severity = level
        base_model = builder.model_builder(config.model)
        base_model.load_model_from_ckpt(args.ckpts)
        if args.use_gpu:
            base_model.to(args.local_rank)
        if args.distributed:
            if args.sync_bn:
                base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
                print_log('Using Synchronized BatchNorm ...', logger=logger)
            base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[
                args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
            print_log('Using Distributed Data parallel ...', logger=logger)
        else:
            print_log('Using Data parallel ...', logger=logger)
            base_model = nn.DataParallel(base_model).cuda()
        print('Testing Source Performance...')
        if corruption_type == 'clean':
            inference_dataset = tta_datasets.ModelNetClean(args)
            inference_loader = DataLoader(dataset=inference_dataset, batch_size=6)
        else:
            inference_dataset = tta_datasets.ModelNet40C(args)
            inference_loader = DataLoader(dataset=inference_dataset, batch_size=6)
        test_pred = []
        test_label = []
        base_model.eval()
        with torch.no_grad():
            for idx_inference, (data_inference, labels) in enumerate(inference_loader):
                points = data_inference.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)
                logits = base_model.module.classification_only(points, only_unmasked=False)
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            print(f'Source Peformance ::: Corruption ::: {args.corruption} ::: {acc}')


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    base_model.load_model_from_ckpt(args.ckpts)  # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        print_log(f"[TEST_VOTE]", logger=logger)
        acc = 0.
        for time in range(1, 10):
            this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
            if acc < this_acc:
                acc = this_acc
            print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)


def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            # fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            fps_idx_raw = sample_farthest_points(points_raw, K=point_all) # pytorch3d

            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                # points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                #                                         fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                points = torch.gather(points_raw, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3)) # pytorch3d

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)

    return acc
