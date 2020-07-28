import numpy as np
import torch
import skimage.metrics
import models.lpips


class UtilsMeasures:
    model_lpips = None

    def init_lpips(self, device):
        self.model_lpips = models.lpips.PerceptualLoss(model='net-lin', net='alex', use_gpu='cuda' in device)
        for param in self.model_lpips.parameters():
            param.requires_grad = False

    def destroy_lpips(self):
        self.model_lpips = None

    def psnr(self, input, target):
        """Computes the PSNR between two images.

        Args:
            input (torch.FloatTensor): tensor of size (C,F,H,W) containing predicted images.
            target (torch.FloatTensor): tensor of size (C,F,H,W) containing ground-truth images.
        """
        items_psnr = []
        for f in range(target.size(1)):
            items_psnr.append(skimage.metrics.peak_signal_noise_ratio(
                target[:, f].numpy(), input[:, f].numpy())
            )
        return np.mean([item_psnr for item_psnr in items_psnr if not np.isinf(item_psnr) and not np.isnan(item_psnr)])

    def ssim(self, input, target):
        """Computes the SSIM between two images.

        Args:
            input (torch.FloatTensor): tensor of size (C,F,H,W) containing predicted images.
            target (torch.FloatTensor): tensor of size (C,F,H,W) containing ground-truth images.
        """
        items_ssim = []
        for f in range(target.size(1)):
            items_ssim.append(skimage.metrics.structural_similarity(
                target[:, f].permute(1, 2, 0).numpy(), input[:, f].permute(1, 2, 0).numpy(), multichannel=True)
            )
        return np.mean(items_ssim)

    def lpips(self, y, y_hat_comp):
        # Escale both y and y_hat_comp to be between [-1, 1]. The initial state is between [0, 1]
        y, y_hat_comp = y * 2 - 1, y_hat_comp * 2 - 1
        with torch.no_grad():
            return self.model_lpips.forward(y, y_hat_comp).flatten().cpu().tolist()
