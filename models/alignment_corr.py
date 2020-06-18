import torch.nn as nn
import torch.nn.functional as F
import models.corr
import torch
import utils.movement


class AlignmentCorrelationMixer(nn.Module):
    def __init__(self, corr_size=16):
        super(AlignmentCorrelationMixer, self).__init__()
        assert corr_size == 16
        self.mixer = nn.Sequential(
            nn.Conv2d(corr_size ** 2, corr_size ** 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(corr_size ** 2, corr_size ** 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(corr_size ** 2, corr_size, kernel_size=3, padding=1), nn.ReLU(),  # Out = 16
            nn.Conv2d(corr_size, corr_size, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(corr_size, corr_size, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(corr_size, corr_size // 2, kernel_size=3, padding=1), nn.ReLU(),  # Out = 8
            nn.Conv2d(corr_size // 2, corr_size // 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(corr_size // 2, corr_size // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(corr_size // 2, corr_size // 4, kernel_size=3, padding=1), nn.ReLU(),  # Out = 4
            nn.Conv2d(corr_size // 4, corr_size // 4, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(corr_size // 4, corr_size // 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(corr_size // 4, corr_size // 8, kernel_size=3, padding=1),  # Out = 2
            nn.Conv2d(corr_size // 8, corr_size // 8, kernel_size=5, padding=2),
            nn.Conv2d(corr_size // 8, corr_size // 8, kernel_size=3, padding=1)
        )

    def forward(self, corr):
        b, f, h, w, *_ = corr.size()
        corr = corr.reshape(b * f, -1, 16, 16)
        return self.mixer(corr).reshape(b, f, 2, h, w).permute(0, 1, 3, 4, 2)


class FlowEstimator(nn.Module):
    def __init__(self, in_c=10):
        super(FlowEstimator, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, in_c * 2, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 2, in_c * 4, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 4, in_c * 4, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c * 4, in_c * 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 4, in_c * 8, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 8, in_c * 8, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c * 8, in_c * 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(in_c * 8, in_c * 4, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(in_c * 4, in_c * 4, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c * 4, in_c * 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(in_c * 4, in_c * 2, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(in_c * 2, in_c, kernel_size=5, padding=2, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, 3, kernel_size=3, padding=1)
        )

    def forward(self, x, m, flow, t, r_list):
        b, c, f, h, w = x.size()

        # Prepare data and propagate it through the model
        nn_input = torch.cat([
            x[:, :, r_list].transpose(1, 2).reshape(b * (f - 1), c, h, w),
            x[:, :, t].unsqueeze(1).repeat(1, f - 1, 1, 1, 1).reshape(b * (f - 1), c, h, w),
            m[:, :, r_list].transpose(1, 2).reshape(b * (f - 1), 1, h, w),
            m[:, :, t].unsqueeze(1).repeat(1, f - 1, 1, 1, 1).reshape(b * (f - 1), 1, h, w),
            flow.transpose(1, 2).reshape(b * (f - 1), 2, h, w)
        ], dim=1)
        nn_output = self.nn(nn_input).reshape(b, f - 1, 3, h, w).transpose(1, 2)

        # Apply Sigmoid to channel 3 (v_map)
        nn_output[:, 2] = torch.sigmoid(nn_output[:, 2])

        # Return flow and v_map separately
        return nn_output[:, :2].permute(0, 2, 3, 4, 1), nn_output[:, 2].unsqueeze(1)


class AlignmentCorrelation(nn.Module):

    def __init__(self, model_vgg):
        super(AlignmentCorrelation, self).__init__()
        self.corr = models.corr.CorrelationVGG(model_vgg)
        self.corr_mixer = AlignmentCorrelationMixer()
        self.flow_64 = FlowEstimator()
        self.flow_256 = FlowEstimator()
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

    def forward(self, x, m, t, r_list):
        b, c, f, h, w = x.size()

        # Normalize the input
        x = (x - self.mean) / self.std

        # Apply the CorrelationVGG module. Corr is (b, t, h, w, h, w)
        corr = self.corr(x, m, t, r_list)

        # Mix the corr 4D volume to obtain a 16x16 dense flow estimation of size (b, t, 16, 16, 2)
        flow_16 = self.corr_mixer(corr)

        # Interpolate x, m and corr_mixed to be 64x64
        x_64 = F.interpolate(
            x.transpose(1, 2).reshape(b * f, c, h, w), (64, 64), mode='bilinear'
        ).reshape(b, f, c, 64, 64).transpose(1, 2)
        m_64 = F.interpolate(
            m.transpose(1, 2).reshape(b * f, 1, h, w), (64, 64), mode='nearest'
        ).reshape(b, f, 1, 64, 64).transpose(1, 2)
        flow_64_pre = F.interpolate(
            flow_16.reshape(b * (f - 1), 16, 16, 2).permute(0, 3, 1, 2), (64, 64), mode='bilinear'
        ).reshape(b, f - 1, 2, 64, 64).transpose(1, 2)

        # Estimate 64x64 flow correction of size (b, t, 64, 64, 2)
        flow_64, vmap_64 = self.flow_64(x_64, m_64, flow_64_pre, t, r_list)

        # Interpolate flow_64 to be 256x256
        flow_256_pre = F.interpolate(
            flow_64.reshape(b * (f - 1), 64, 64, 2).permute(0, 3, 1, 2), (h, w), mode='bilinear'
        ).reshape(b, f - 1, 2, h, w).transpose(1, 2)

        # Estimate 256x256 flow correction of size (b, t, 256, 256, 2)
        flow_256, vmap_256 = self.flow_256(x, m, flow_256_pre, t, r_list)

        # Return both corr and corr_mixed
        return corr, flow_16, flow_64, flow_256, vmap_64, vmap_256
