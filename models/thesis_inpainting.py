import torch
import torch.nn as nn
import models.rrdb_net
import matplotlib.pyplot as plt


class ThesisInpaintingVisible(nn.Module):

    def __init__(self, in_c=3):
        super(ThesisInpaintingVisible, self).__init__()
        # self.nn = models.rrdb_net.RRDBNet(in_c, 3)
        self.nn = nn.Sequential(
            nn.Conv2d(in_c, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
        )
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, x_target, v_target, y_target, x_aligned, v_aligned):
        b, c, f, h, w = x_aligned.size()

        # Normalize the input images
        xs_target_norm = (x_target - self.mean.squeeze(2)) / self.std.squeeze(2)
        xs_aligned_norm = (x_aligned - self.mean) / self.std

        # Compute visible zones mask
        visible_zones_mask = (v_aligned - v_target.unsqueeze(2)).clamp(0, 1)

        # Combine the input of the NN
        nn_input = torch.cat([
            xs_aligned_norm.transpose(1, 2).reshape(b * f, c, h, w),
            xs_target_norm.repeat(1, 1, f, 1, 1).transpose(1, 2).reshape(b * f, c, h, w),
            v_aligned.transpose(1, 2).reshape(b * f, 1, h, w),
            v_target.repeat(1, 1, f, 1, 1).transpose(1, 2).reshape(b * f, 1, h, w),
            visible_zones_mask.transpose(1, 2).reshape(b * f, 1, h, w)
        ], dim=1)

        # PATH TEST
        nn_input = xs_target_norm.repeat(1, 1, f, 1, 1).transpose(1, 2).reshape(b * f, c, h, w)

        # Propagate data through the NN
        y_hat = self.nn(nn_input).reshape(b, f, c, h, w).transpose(1, 2) * self.std + self.mean
        y_hat_comp = (y_target * v_target).unsqueeze(2).repeat(1, 1, f, 1, 1) + \
                     y_hat * (1 - v_target).unsqueeze(2).repeat(1, 1, f, 1, 1)

        # Return the data
        return y_hat, y_hat_comp, visible_zones_mask