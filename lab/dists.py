import torch
import matplotlib.pyplot as plt


def gaussianize_difference(x: torch.Tensor, eps: float = 1e-5):

    B, N, d = x.shape
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

    return delta.view(B, N, d)


# 1. Original data
x = torch.randn((3, 5, 384)) * 2 + 3  # Not exactly Gaussian

difference = gaussianize_difference(x)  # Call the function to ensure it's working

x_gaussianized = x - difference  # Gaussianized data

for i in range(3):
    x_ = x[i, 0, :].cpu()  # Take the first sample for plotting
    x_gaussianized_ = x_gaussianized[
        i, 0, :
    ].cpu()  # Take the first sample for plotting

    # 7. Plot original vs transformed
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(x_.numpy(), bins=50, density=True, label="Original")
    plt.title("Original Data")
    plt.subplot(1, 2, 2)
    plt.hist(
        x_gaussianized_.numpy(),
        bins=50,
        density=True,
        label="Gaussianized",
        color="orange",
    )
    plt.title("Transformed to Gaussian")
    plt.tight_layout()
    plt.show()
