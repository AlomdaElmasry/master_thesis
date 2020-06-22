import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ThesisInpaintingVisible(nn.Module):

    def __init__(self, in_c=12):
        super(ThesisInpaintingVisible, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def get_n_params(self, model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def forward(self, xs_target, vs_target, ys_target, xs_aligned, vs_aligned):
        b, c, f, h, w = xs_aligned[2].size()

        # Normalize the input images
        xs_target_norm = (xs_target[2] - self.mean.squeeze(2)) / self.std.squeeze(2)
        xs_aligned_norm = (xs_aligned[2] - self.mean) / self.std

        # Hard combination between the target and the aux frames
        visible_zones_mask = (vs_aligned[2] - vs_target[2].unsqueeze(2)).clamp(0, 1)
        xs_combined = xs_aligned_norm * visible_zones_mask + xs_target_norm.unsqueeze(2) * (1 - visible_zones_mask)

        # Combine the input of the NN
        nn_input = torch.cat([
            xs_combined.transpose(1, 2).reshape(b * f, c, h, w),
            xs_aligned_norm.transpose(1, 2).reshape(b * f, c, h, w),
            xs_target_norm.repeat(1, 1, f, 1, 1).transpose(1, 2).reshape(b * f, c, h, w),
            vs_aligned[2].transpose(1, 2).reshape(b * f, 1, h, w),
            vs_target[2].repeat(1, 1, f, 1, 1).transpose(1, 2).reshape(b * f, 1, h, w),
            visible_zones_mask.transpose(1, 2).reshape(b * f, 1, h, w)
        ], dim=1)

        # Propagate data through the NN
        y_hat = self.convs(nn_input).reshape(b, f, c, h, w).transpose(1, 2) * self.std + self.mean
        y_hat_comp = (ys_target[2] * vs_target[2]).unsqueeze(2).repeat(1, 1, f, 1, 1) + y_hat * (1 - vs_aligned[2])

        # Return the data
        return None, None, None, None, y_hat, y_hat_comp, visible_zones_mask