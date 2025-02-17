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
level = [5]

from tqdm import tqdm


def load_tta_dataset(args, config):
    # we have 3 choices - every tta_loader returns only point and labels
    root = config.tta_dataset_path  # being lazy - 1

    if args.dataset_name == 'modelnet':
        root = os.path.join(root, f'{args.dataset_name}_c')

        if args.corruption == 'clean':
            inference_dataset = tta_datasets.ModelNet_h5(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    elif args.dataset_name == 'partnet':
        if args.corruption != 'clean':
            root = os.path.join(root, f'{args.dataset_name}_c',
                                f'{args.corruption}_{args.severity}')
        else:
            root = os.path.join(root, f'{args.dataset_name}_c',
                                f'{args.corruption}')

        inference_dataset = tta_datasets.PartNormalDataset(root=root, npoints=config.npoints, split='test',
                                                           normal_channel=config.normal, debug=args.debug)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
    elif args.dataset_name == 'scanobject':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    elif args.dataset_name == 'shapenetcore':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    else:
        raise NotImplementedError(f'TTA for {args.tta} is not implemented')

    print(f'\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n')

    return tta_loader



def load_clean_dataset(args, config):
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
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
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    dataset_name = args.dataset_name

    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':  # for with background
        config.model.cls_dim = 15
    elif dataset_name == 'scanobject_nbg':  # for no background
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):

            if corr_id == 0:
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path='source_only_results/')  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'Source Only Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Check Point: {args.ckpts}' + '\n\n')

            base_model = load_base_model(args, config, logger)
            print('Testing Source Performance...')
            test_pred = []
            test_label = []
            base_model.eval()

            inference_loader = load_tta_dataset(args, config)

            with torch.no_grad():
                for idx_inference, (data, labels) in enumerate(inference_loader):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name in ['scanobject', 'scanobject_wbg', 'scanobject_nbg']:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                        label = labels.cuda()
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()

                    points = points.cuda()
                    labels = label.cuda()
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

                f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
                f_write.flush()
                if corr_id == len(corruptions) - 1:
                    f_write.close()
                    print(f'Final Results Saved at:', os.path.join('source_only_results/', f'{logtime}_results.txt'))


def source_prune(args, config, train_writer=None):
    dataset_name = args.dataset_name
    npoints = config.npoints
    logger = get_logger(args.log_name)
    
    source_model = load_base_model(args, config, logger)
    source_model.eval()
    
    
    clean_intermediates_path = f'intermediate_features/{args.dataset_name}_clean_intermediates.pth'
    if os.path.exists(clean_intermediates_path):
        clean_intermediates = torch.load(clean_intermediates_path)
        clean_intermediates_mean, clean_intermediates_std = clean_intermediates['mean'], clean_intermediates['std']
    else:
        clean_intermediates = None
        clean_dataloader = load_clean_dataset(args, config) 
        for i, (_, _, data) in tqdm(enumerate(clean_dataloader), total=len(clean_dataloader)):
            points = data[0].cuda()
            points = misc.fps(points, npoints)
            intermediates = source_model(points, out_intermediate=True)[1]
            intermediates = torch.stack(intermediates, dim=0) # (num_layers, batch_size, tokens, emb_dim)
            if i == 0:
                clean_intermediates = intermediates.detach().cpu()
            else:
                clean_intermediates = torch.cat([clean_intermediates, intermediates.detach().cpu()], dim=1)
            
        clean_intermediates_mean = clean_intermediates.mean(dim=(1,2)) # (num_layers, emb_dim)
        clean_intermediates_std = clean_intermediates.std(dim=(1,2)) # (num_layers, emb_dim)
        torch.save({'mean': clean_intermediates_mean, 'std': clean_intermediates_std}, clean_intermediates_path)
        

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            acc_sliding_window = list()
            acc_avg = list()
            if args.corruption == 'clean':
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            if corr_id == 0:  # for saving results for easy copying to google sheet

                f_write, logtime = get_writer_to_all_result(args, config, custom_path='results_final_tta/')
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
                f_write.write(f'Corruption LEVEL: {args.severity}' + '\n\n')

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
                losses = AverageMeter(['Reconstruction Loss'])

                if not args.online:
                    # base_model = load_base_model(args, config, logger)
                    base_model = load_base_model(args, config, logger, pretrained=False)
                    base_model.load_state_dict(source_model.state_dict())
                    optimizer = builder.build_opti_sche(base_model, config)[0]
                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)
                if args.train_with_prune:
                    for grad_step in range(args.grad_steps):
                        points = data.cuda()
                        points = misc.fps(points, npoints)


                        # make a batch
                        if idx % args.stride_step == 0 or idx == len(tta_loader) - 1:
                            points = [points for _ in range(args.batch_size_tta)]
                            points = torch.squeeze(torch.vstack(points))

                            loss_recon, loss_p_consistency, loss_regularize = base_model(points)
                            loss = loss_recon + (args.alpha * loss_regularize)  # + (0.0001 * loss_p_consistency)
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

                        print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                                f'GradStep - {grad_step} / {args.grad_steps},'
                                f'Reconstruction Loss {[l for l in losses.val()]}',
                                logger=logger)

                # now inferring on this one sample
                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)

                logits = base_model(points, source_stats=(clean_intermediates_mean, clean_intermediates_std))
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

                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.

                    print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                              logger=logger)

                    acc_avg.append(acc.cpu())
            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                      logger=logger)
            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('results_final/', f'{logtime}_results.txt'))
                if train_writer is not None:
                    train_writer.close()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y
