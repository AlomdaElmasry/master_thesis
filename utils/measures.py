import numpy as np
import torch
import skimage.metrics


def psnr(y, y_hat_comp):
    """Computes the PSNR between two images.

    Args:
        y (torch.FloatTensor): tensor of size (B,C,H,W) containing ground-truth images.
        y_hat_comp (torch.FloatTensor): tensor of size (B,C,H,W) containing predicted images.
    """
    items_psnr = []
    for b in range(y.size(0)):
        items_psnr.append(skimage.metrics.peak_signal_noise_ratio(y[b].numpy(), y_hat_comp[b].numpy()))
    return [item_psnr for item_psnr in items_psnr if not np.isinf(item_psnr) and not np.isnan(item_psnr)]


def ssim(y, y_hat_comp):
    items_ssim = []
    for b in range(y.size(0)):
        items_ssim.append(skimage.metrics.structural_similarity(
            y[b].permute(1, 2, 0).numpy(), y_hat_comp[b].permute(1, 2, 0).numpy(), multichannel=True)
        )
    return items_ssim


def lpips(y, y_hat_comp, model):
    # Escale both y and y_hat_comp to be between [-1, 1]. The initial state is between [0, 1]
    y, y_hat_comp = y * 2 - 1, y_hat_comp * 2 - 1
    with torch.no_grad():
        return model.forward(y, y_hat_comp).flatten().cpu().tolist()
