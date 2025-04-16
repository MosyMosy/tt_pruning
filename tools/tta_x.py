# import os
# import time
# import torch
# from tools import builder
# from utils import misc, dist_utils
# from utils.logger import *
# import datasets.tta_datasets as tta_datasets
# from torch.utils.data import DataLoader
# from utils.misc import *
# import numpy as np

# from transformers import CLIPModel, CLIPTokenizer
# from tqdm import tqdm
# from datasets.label_list import imagenet_templates, dataset_label_descriptions

# level = [5]


# def load_tta_dataset(args, config):
#     # we have 3 choices - every tta_loader returns only point and labels
#     root = config.tta_dataset_path  # being lazy - 1

#     if config.dataset.name == "modelnet":
#         if args.corruption == "clean":
#             inference_dataset = tta_datasets.ModelNet_h5(args, root)
#             tta_loader = DataLoader(
#                 dataset=inference_dataset,
#                 batch_size=args.batch_size,
#                 shuffle=args.shuffle,
#                 drop_last=False,
#             )
#         else:
#             inference_dataset = tta_datasets.ModelNet40C(args, root)
#             tta_loader = DataLoader(
#                 dataset=inference_dataset,
#                 batch_size=args.batch_size,
#                 shuffle=args.shuffle,
#                 drop_last=False,
#             )

#     elif config.dataset.name == "scanobject":
#         inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
#         tta_loader = DataLoader(
#             inference_dataset,
#             batch_size=args.batch_size,
#             shuffle=args.shuffle,
#             drop_last=False,
#         )

#     elif config.dataset.name == "shapenetcore":
#         inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)
#         tta_loader = DataLoader(
#             inference_dataset,
#             batch_size=args.batch_size,
#             shuffle=args.shuffle,
#             drop_last=False,
#         )

#     else:
#         raise NotImplementedError(f"TTA for {args.tta} is not implemented")

#     print(f"\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n")

#     return tta_loader


# def load_clean_dataset(args, config):
#     (train_sampler, train_dataloader) = builder.dataset_builder(
#         args, config.dataset.train
#     )
#     return train_dataloader


# def load_base_model(args, config, logger, load_part_seg=False, pretrained=True):
#     base_model = builder.model_builder(config.model)
#     if pretrained:
#         base_model.load_model_from_ckpt(args.ckpts)
#     if args.use_gpu:
#         base_model.to(args.local_rank)
#     if args.distributed:
#         if args.sync_bn:
#             base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
#             print_log("Using Synchronized BatchNorm ...", logger=logger)
#         base_model = nn.parallel.DistributedDataParallel(
#             base_model,
#             device_ids=[args.local_rank % torch.cuda.device_count()],
#             find_unused_parameters=True,
#         )
#         print_log("Using Distributed Data parallel ...", logger=logger)
#     else:
#         print_log("Using Data parallel ...", logger=logger)
#         base_model = nn.DataParallel(base_model).cuda()
#     return base_model


# def extract_label_features(classnames, templates, tokenizer, model):
#     with torch.no_grad():
#         zeroshot_weights = []
#         for classname in tqdm(classnames):
#             texts = [
#                 template.format(classname) for template in templates
#             ]  # format templates
#             inputs = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
#             class_template_features = model.get_text_features(
#                 **inputs
#             )  # extract features
#             class_template_features = (
#                 class_template_features
#                 / class_template_features.norm(dim=-1, keepdim=True)
#             )
#             # class_embedding = text_embeddings.mean(dim=0)
#             # class_embedding = class_embedding / class_embedding.norm()
#             zeroshot_weights.append(class_template_features)
#         zeroshot_weights = torch.stack(zeroshot_weights).cuda()
#     return zeroshot_weights


# def runner(args, config):
#     dataset_name = config.dataset.name
#     logger = get_logger(args.log_name)

#     source_model = load_base_model(args, config, logger)
#     source_model.eval()

#     label_features_path = f"label_features/{dataset_name}_label_features.pt"
#     if os.path.exists(label_features_path):
#         label_features = torch.load(label_features_path)
#     else:
#         # Load the CLIP model and tokenizer
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model_clip_text = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
#             device
#         )
#         tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#         model_clip_text.eval()


#         label_features = extract_label_features(
#             dataset_label_descriptions[dataset_name], imagenet_templates, tokenizer, model_clip_text
#         )
#         torch.save(
#             label_features.cpu(),
#             label_features_path,
#         )
#     label_features = label_features.cuda()

#     resutl_file_path = os.path.join(
#         "results_final_tta/",
#         args.method,
#         f"{args.exp_name}_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.txt",
#     )

#     eval(
#         args,
#         config,
#         logger,
#         source_model,
#         label_features,
#         resutl_file_path,
#     )


# @torch.jit.script
# def softmax_entropy(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


# def eval(
#     args,
#     config,
#     logger,
#     source_model,
#     label_features,
#     resutl_file_path,
# ):
#     label_features = label_features.mean(dim=1)
#     label_features = label_features / label_features.norm(dim=-1, keepdim=True)
#     label_features_sim = (
#         label_features @ label_features.t()
#     )  # shape (num_classes, num_classes)
#     label_features_sim = label_features_sim / label_features_sim.norm(
#         dim=-1, keepdim=True
#     )  # shape (num_classes, num_classes)

#     time_list = []
#     for args.severity in level:
#         for corr_id, args.corruption in enumerate(corruptions):
#             start_time = time.time()
#             acc_sliding_window = list()
#             acc_avg = list()
#             if args.corruption == "clean":
#                 continue
#                 # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

#             # if corr_id not in [2]:
#             #     continue

#             if (
#                 "f_write" not in locals()
#             ):  # for saving results for easy copying to google sheet
#                 f_write = get_writer_to_all_result(
#                     args, config, custom_path=resutl_file_path
#                 )
#                 f_write.write(f"All Corruptions: {corruptions}" + "\n\n")
#                 f_write.write(
#                     f"TTA Results for Dataset: {config.dataset.name}" + "\n\n"
#                 )
#                 f_write.write(f"Checkpoint Used: {args.ckpts}" + "\n\n")
#                 f_write.write(f"Corruption LEVEL: {args.severity}" + "\n\n")

#             tta_loader = load_tta_dataset(args, config)
#             test_pred = []
#             test_label = []
#             entropy_list = []

#             base_model = load_base_model(args, config, logger, pretrained=False)
#             base_model.load_state_dict(source_model.state_dict())

#             for idx, (data, labels) in enumerate(tta_loader):
#                 # now inferring on this one sample

#                 # reset batchnorm running stats

#                 base_model.eval()

#                 points = data.cuda()
#                 labels = labels.cuda()
#                 points = [
#                     misc.fps(points, config.npoints, random_start_point=True)
#                     for _ in range(args.batch_size_tta)
#                 ]
#                 points = torch.cat(points, dim=0)

#                 labels = [labels for _ in range(args.batch_size_tta)]
#                 labels = torch.cat(labels, dim=0)

#                 # reset batchnorm running stats
#                 if args.BN_reset:
#                     for m in base_model.modules():
#                         if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
#                             m.running_mean = None  # for original implementation of tent
#                             m.running_var = None  # for original implementation of tent
#                 logits = base_model(points)
#                 logits = logits / logits.norm(dim=-1, keepdim=True)
#                 logits_label_sim = logits @ label_features_sim.t()

#                 pred = (logits_label_sim + logits).argmax(-1).view(-1)

#                 test_pred.append(pred.detach())

#                 target = labels.view(-1)
#                 test_label.append(target.detach())

#                 if idx % 50 == 0:
#                     test_pred_ = torch.cat(test_pred, dim=0)
#                     test_label_ = torch.cat(test_label, dim=0)

#                     if args.distributed:
#                         test_pred_ = dist_utils.gather_tensor(test_pred_, args)
#                         test_label_ = dist_utils.gather_tensor(test_label_, args)

#                     acc = (
#                         (test_pred_ == test_label_).sum()
#                         / float(test_label_.size(0))
#                         * 100.0
#                     )

#                     print_log(
#                         f"\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n",
#                         logger=logger,
#                     )

#                     acc_avg.append(acc.cpu())
#             end_time = time.time()
#             time_list.append(end_time - start_time)
#             test_pred = torch.cat(test_pred, dim=0)
#             test_label = torch.cat(test_label, dim=0)

#             if args.distributed:
#                 test_pred = dist_utils.gather_tensor(test_pred, args)
#                 test_label = dist_utils.gather_tensor(test_label, args)

#             acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.0
#             print_log(
#                 f"\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n",
#                 logger=logger,
#             )
#             f_write.write(" ".join([str(round(float(xx), 3)) for xx in [acc]]) + "\n")
#             # f_write.write(
#             #     " ".join([str(round(float(xx), 3)) for xx in [torch.stack(entropy_list).mean().item()]]) + "\n"
#             # )
#             f_write.flush()

#             if corr_id == len(corruptions) - 1:
#                 # write min, max, and average, variance,  of times
#                 f_write.write(
#                     " ".join(
#                         [
#                             str(round(float(xx), 3))
#                             for xx in [
#                                 min(time_list),
#                                 max(time_list),
#                                 sum(time_list) / len(time_list),
#                                 np.var(time_list),
#                             ]
#                         ]
#                     )
#                     + "\n"
#                 )

#                 f_write.flush()
#                 f_write.close()

#                 print(
#                     f"Final Results Saved at:",
#                     resutl_file_path,
#                 )



# def eval_topk(
#     args,
#     config,
#     logger,
#     source_model,
#     label_features,
#     resutl_file_path,
# ):
#     # label_features = label_features.mean(dim=1)
#     label_features = label_features / label_features.norm(dim=-1, keepdim=True)
#     label_features = label_features.permute(
#         1, 0, 2
#     )  # shape (num_classes, num_features)
#     label_features_sim = label_features @ label_features.transpose(
#         -2, -1
#     )  # shape (num_classes, num_classes)
#     label_features_sim = label_features_sim / label_features_sim.norm(
#         dim=-1, keepdim=True
#     )  # shape (num_classes, num_classes)

#     time_list = []
#     for args.severity in level:
#         for corr_id, args.corruption in enumerate(corruptions):
#             start_time = time.time()
#             acc_sliding_window = list()
#             acc_avg = list()
#             if args.corruption == "clean":
#                 continue
#                 # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

#             # if corr_id not in [2]:
#             #     continue

#             if (
#                 "f_write" not in locals()
#             ):  # for saving results for easy copying to google sheet
#                 f_write = get_writer_to_all_result(
#                     args, config, custom_path=resutl_file_path
#                 )
#                 f_write.write(f"All Corruptions: {corruptions}" + "\n\n")
#                 f_write.write(
#                     f"TTA Results for Dataset: {config.dataset.name}" + "\n\n"
#                 )
#                 f_write.write(f"Checkpoint Used: {args.ckpts}" + "\n\n")
#                 f_write.write(f"Corruption LEVEL: {args.severity}" + "\n\n")

#             tta_loader = load_tta_dataset(args, config)
#             test_pred = []
#             test_label = []
#             entropy_list = []

#             base_model = load_base_model(args, config, logger, pretrained=False)
#             base_model.load_state_dict(source_model.state_dict())

#             for idx, (data, labels) in enumerate(tta_loader):
#                 # now inferring on this one sample

#                 # reset batchnorm running stats

#                 base_model.eval()

#                 points = data.cuda()
#                 labels = labels.cuda()
#                 points = [
#                     misc.fps(points, config.npoints, random_start_point=True)
#                     for _ in range(args.batch_size_tta)
#                 ]
#                 points = torch.cat(points, dim=0)

#                 labels = [labels for _ in range(args.batch_size_tta)]
#                 labels = torch.cat(labels, dim=0)

#                 # reset batchnorm running stats
#                 if args.BN_reset:
#                     for m in base_model.modules():
#                         if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
#                             m.running_mean = None  # for original implementation of tent
#                             m.running_var = None  # for original implementation of tent
#                 logits = base_model(points)
#                 logits = logits / logits.norm(dim=-1, keepdim=True)
#                 logits_label_sim = logits.unsqueeze(1).unsqueeze(
#                     1
#                 ) @ label_features_sim.transpose(-2, -1)
#                 logits_label_sim = logits_label_sim.squeeze(2)
#                 logits_label_entropy = softmax_entropy(logits_label_sim, dim=-1)
#                 top_indices = logits_label_entropy.topk(5, dim=-1).indices
#                 logits_label_sim = logits_label_sim.gather(
#                     1,
#                     top_indices.unsqueeze(-1).expand(-1, -1, logits_label_sim.size(-1)),
#                 )
#                 logits_label_sim = logits_label_sim.mean(dim=1)
#                 logits_label_sim = logits_label_sim / logits_label_sim.norm(
#                     dim=-1, keepdim=True
#                 )

#                 pred = (logits_label_sim).argmax(-1).view(-1)

#                 test_pred.append(pred.detach())

#                 target = labels.view(-1)
#                 test_label.append(target.detach())

#                 if idx % 50 == 0:
#                     test_pred_ = torch.cat(test_pred, dim=0)
#                     test_label_ = torch.cat(test_label, dim=0)

#                     if args.distributed:
#                         test_pred_ = dist_utils.gather_tensor(test_pred_, args)
#                         test_label_ = dist_utils.gather_tensor(test_label_, args)

#                     acc = (
#                         (test_pred_ == test_label_).sum()
#                         / float(test_label_.size(0))
#                         * 100.0
#                     )

#                     print_log(
#                         f"\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n",
#                         logger=logger,
#                     )

#                     acc_avg.append(acc.cpu())
#             end_time = time.time()
#             time_list.append(end_time - start_time)
#             test_pred = torch.cat(test_pred, dim=0)
#             test_label = torch.cat(test_label, dim=0)

#             if args.distributed:
#                 test_pred = dist_utils.gather_tensor(test_pred, args)
#                 test_label = dist_utils.gather_tensor(test_label, args)

#             acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.0
#             print_log(
#                 f"\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n",
#                 logger=logger,
#             )
#             f_write.write(" ".join([str(round(float(xx), 3)) for xx in [acc]]) + "\n")
#             # f_write.write(
#             #     " ".join([str(round(float(xx), 3)) for xx in [torch.stack(entropy_list).mean().item()]]) + "\n"
#             # )
#             f_write.flush()

#             if corr_id == len(corruptions) - 1:
#                 # write min, max, and average, variance,  of times
#                 f_write.write(
#                     " ".join(
#                         [
#                             str(round(float(xx), 3))
#                             for xx in [
#                                 min(time_list),
#                                 max(time_list),
#                                 sum(time_list) / len(time_list),
#                                 np.var(time_list),
#                             ]
#                         ]
#                     )
#                     + "\n"
#                 )

#                 f_write.flush()
#                 f_write.close()

#                 print(
#                     f"Final Results Saved at:",
#                     resutl_file_path,
#                 )
