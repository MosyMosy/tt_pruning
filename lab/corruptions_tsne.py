import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import time
import random
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tools import builder
from utils import misc, dist_utils
from utils.logger import get_logger, print_log, get_writer_to_all_result

from utils.misc import corruptions

import datasets.tta_datasets as tta_datasets


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


def plot_tsne_corruptions(logits, labels, titles, max_points_per_corruption=500):
    """
    logits: Tensor (N_corr, BB, l)
    labels: Tensor (N_corr, BB, L)
    titles: List of corruption names (length N_corr)
    """
    N_corr, BB, l = logits.shape
    L = labels.shape[-1]

    # Optional downsampling for speed
    if BB > max_points_per_corruption:
        idx = torch.randperm(BB)[:max_points_per_corruption]
        logits = logits[:, idx]
        labels = labels[:, idx]

    # Reshape for t-SNE
    logits_2d = logits.reshape(-1, l).cpu().numpy()  # (N_corr*BB, l)
    labels_flat = labels.reshape(-1, L).cpu().numpy()  # (N_corr*BB, L)

    if L > 1:
        labels_flat = np.argmax(labels_flat, axis=1)

    # Run t-SNE
    tsne = TSNE(n_components=2, init="random", random_state=42)
    tsne_result = tsne.fit_transform(logits_2d)  # (N_corr*BB, 2)

    # Plot: one subplot per corruption
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    num_classes = np.unique(labels_flat).size

    for i in range(N_corr):
        ax = axes[i]
        start = i * BB
        end = (i + 1) * BB
        tsne_part = tsne_result[start:end]
        labels_part = labels_flat[start:end]
        for c in range(num_classes):
            idxs = labels_part == c
            ax.scatter(
                tsne_part[idxs, 0],
                tsne_part[idxs, 1],
                label=f"Class {c}",
                s=5,
                alpha=0.6,
            )
        ax.set_title(titles[i])
        ax.axis("off")

    fig.suptitle("t-SNE of Logits across Corruptions", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()



