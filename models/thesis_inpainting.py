import torch
import torch.nn as nn
import models.rrdb_net
import matplotlib.pyplot as plt


class ThesisInpaintingModel(nn.Module):

    def __init__(self, in_c):
        super(ThesisInpaintingModel, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.nn(x)


class ThesisInpaintingVisible(nn.Module):
    model_type = None

    def __init__(self, in_c=9, model_type='simple'):
        super(ThesisInpaintingVisible, self).__init__()
        self.model_type = model_type
        if model_type == 'separable':
            self.nn = models.rrdb_net.RRDBNetSeparable(in_nc=3, out_nc=3, nb=15)
        elif model_type == 'simple':
            self.nn = ThesisInpaintingModel(in_c=in_c)
        else:
            self.nn = models.rrdb_net.RRDBNet(in_nc=in_c, out_nc=3, nb=10)
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, x_target, v_target, y_target, x_ref_aligned, v_ref_aligned, v_map):
        b, c, f, h, w = x_ref_aligned.size()

        # Normalize the input images
        x_target_norm = (x_target - self.mean.squeeze(2)) / self.std.squeeze(2)
        x_ref_aligned_norm = (x_ref_aligned - self.mean) / self.std

        # Transform masks sizes
        v_target = v_target.unsqueeze(2).repeat(1, 1, f, 1, 1)
        y_target = y_target.unsqueeze(2).repeat(1, 1, f, 1, 1)

        # Combine x_target with the aligned version
        x_target_inpainted_norm = v_map * x_ref_aligned_norm + \
                                  (1 - v_map) * x_target_norm.unsqueeze(2).repeat(1, 1, f, 1, 1)
        v_target_c = torch.clamp(v_target + v_map, 0, 1)

        # Predict output depending on the NN
        if self.model_type == 'separable':
            nn_output = self.nn(
                x_target_norm.unsqueeze(2).repeat(1, 1, f, 1, 1).transpose(1, 2).reshape(b * f, c, h, w),
                x_ref_aligned_norm.transpose(1, 2).reshape(b * f, c, h, w),
                v_target.unsqueeze(2).repeat(1, 1, f, 1, 1).transpose(1, 2).reshape(b * f, 1, h, w),
                v_ref_aligned.transpose(1, 2).reshape(b * f, 1, h, w),
                v_map.transpose(1, 2).reshape(b * f, 1, h, w)
            ).reshape(b, f, c, h, w).transpose(1, 2)
        else:
            nn_input = torch.cat([x_target_inpainted_norm, x_ref_aligned_norm, v_target_c, v_ref_aligned, v_map], dim=1)
            nn_output = self.nn(nn_input.transpose(1, 2).reshape(b * f, 9, h, w)).reshape(b, f, c, h, w).transpose(1, 2)

        # Propagate data through the NN
        y_hat = torch.clamp(nn_output * self.std + self.mean, 0, 1)
        y_hat_comp = v_target * y_target + y_hat * (1 - v_target)

        # Return the data
        return y_hat, y_hat_comp
