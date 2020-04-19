from models.vgg_16 import get_pretrained_model
import torch
import torch.nn.functional as F


class LossesUtils:
    model_vgg = None

    def init_vgg(self, device):
        self.model_vgg = get_pretrained_model(device)
        for param in self.model_vgg.features.parameters():
            param.requires_grad = False

    def masked_l1(self, x, x_hat, mask, reduction, weight=1):
        masked_l1_loss = F.l1_loss(x_hat * mask, x * mask, reduction=reduction)
        return weight * masked_l1_loss / (torch.sum(mask) if reduction == 'sum' else 1)

    def perceptual(self, x, x_hat, weight=1):
        x_vgg = self.model_vgg(x.contiguous())
        x_hat_vgg = self.model_vgg(x_hat.contiguous())
        loss_perceptual = 0
        for p in range(len(x_vgg)):
            loss_perceptual += F.l1_loss(x_hat_vgg[p], x_vgg[p])
        return loss_perceptual * weight / len(x_vgg), x_vgg, x_hat_vgg

    def style(self, x_vgg, x_hat_vgg, weight=1):
        loss_style = 0
        for p in range(len(x_vgg)):
            b, c, h, w = x_vgg[p].size()
            g_x = torch.mm(x_vgg[p].view(b * c, h * w), x_vgg[p].view(b * c, h * w).t())
            g_x_comp = torch.mm(x_hat_vgg[p].view(b * c, h * w), x_hat_vgg[p].view(b * c, h * w).t())
            loss_style += F.l1_loss(g_x_comp / (b * c * h * w), g_x / (b * c * h * w))
        return loss_style * weight / len(x_vgg)

    def tv(self, x_hat, weight=1):
        loss_tv_h = (x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :]).pow(2).sum()
        loss_tv_w = (x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1]).pow(2).sum()
        loss_tv = (loss_tv_h + loss_tv_w) / (x_hat.size(0) * x_hat.size(1) * x_hat.size(2) * x_hat.size(3))
        return loss_tv * weight