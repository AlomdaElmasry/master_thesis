import torch.nn as nn
import torch.nn.functional as F
import torch
import utils.movement
import utils.correlation


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.nn(x)


class SeparableConv4d(nn.Module):
    def __init__(self, in_c=1, out_c=1):
        super(SeparableConv4d, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_c, 128, 3, padding=1)
        self.conv_2 = torch.nn.Conv2d(128, out_c, 3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(4)
        b, t, h, w, c, *_ = x.size()

        # reshape (b*t*H*W, c, H, W)
        # shape is b, t, H*W, inter_dim, H*W then permute
        x2_bis = self.conv_1(x.reshape(-1, c, h, w))
        x2_bis = x2_bis.reshape(b, t, h * w, x2_bis.size(1), h * w).permute(0, 1, 4, 3, 2)

        # reshape (b*t*H*W, inter_dim, H, W)
        x3_bis = self.conv_2(x2_bis.reshape(-1, x2_bis.size(3), h, w))
        x3_bis = x3_bis.reshape(b, t, h, w, x3_bis.size(1), h, w).squeeze(4)

        # Return last layer
        return x3_bis


class Softmax3d(torch.nn.Module):

    def __init__(self):
        super(Softmax3d, self).__init__()

    def forward(self, input):
        assert input.dim() == 6  # Expect (B,T,H1,W1,H,W)

        # Get dimensions
        b, t, h, w, _, _ = input.size()

        # Transform input to be (B,H,W,H*W*T)
        input = input.permute(0, 2, 3, 4, 5, 1).reshape(b, h, w, -1)

        # Apply Softmax
        input = F.softmax(input, dim=3)

        # Restore original dimensions
        return input.reshape(b, h, w, h, w, t).permute(0, 5, 1, 2, 3, 4)


class CorrelationVGG(nn.Module):

    def __init__(self, use_softmax=False):
        super(CorrelationVGG, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.conv = SeparableConv4d()
        self.softmax = Softmax3d() if use_softmax else None

    def forward(self, x, m, t, r_list):
        b, c, ref_n, h, w = x.size()

        # Get the features of the frames from VGG
        # with torch.no_grad():
        # x_feats = self.model_vgg(x.transpose(1, 2).reshape(b * ref_n, c, h, w), normalize_input=False)
        # x_feats = x_feats[3].reshape(b, ref_n, -1, 16, 16).transpose(1, 2)
        x_feats = self.feature_extractor(
            x.transpose(1, 2).reshape(b * ref_n, c, h, w)
        ).reshape(b, ref_n, -1, 16, 16).transpose(1, 2)

        # Update the parameters to the VGG features
        b, c, ref_n, h, w = x_feats.size()

        # Interpolate the feature masks
        corr_masks = F.interpolate(
            1 - m.transpose(1, 2).reshape(b * ref_n, 1, m.size(3), m.size(4)), size=(h, w), mode='nearest'
        ).reshape(b, ref_n, 1, h, w).transpose(1, 2)

        # Compute the feature correlation
        corr = utils.correlation.compute_masked_4d_correlation(x_feats, corr_masks, t, r_list)

        # Fill holes in the correlation matrix using a NN
        corr = self.conv(corr)

        # Compute the Softmax over each pixel (b, t, h, w, h, w)
        return self.softmax(corr) if self.softmax else corr, x_feats


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
            nn.Conv2d(in_c, 2, kernel_size=3, padding=1)
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
        nn_output = self.nn(nn_input).reshape(b, f - 1, 2, h, w).transpose(1, 2)

        # Return flow and v_map separately
        return nn_output[:, :2].permute(0, 2, 3, 4, 1)


class ThesisAlignmentModel(nn.Module):

    def __init__(self, model_vgg):
        super(ThesisAlignmentModel, self).__init__()
        self.corr = CorrelationVGG(model_vgg)
        self.corr_mixer = AlignmentCorrelationMixer()
        self.flow_64 = FlowEstimator()
        self.flow_256 = FlowEstimator()
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, x, m, t, r_list):
        b, c, f, h, w = x.size()

        # Normalize the input
        x = (x - self.mean) / self.std

        # Apply the CorrelationVGG module. Corr is (b, t, h, w, h, w)
        corr, x_feats = self.corr(x, m, t, r_list)

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
        flow_64 = self.flow_64(x_64, m_64, flow_64_pre, t, r_list)

        # Interpolate flow_64 to be 256x256
        flow_256_pre = F.interpolate(
            flow_64.reshape(b * (f - 1), 64, 64, 2).permute(0, 3, 1, 2), (h, w), mode='bilinear'
        ).reshape(b, f - 1, 2, h, w).transpose(1, 2)

        # Estimate 256x256 flow correction of size (b, t, 256, 256, 2)
        flow_256 = self.flow_256(x, m, flow_256_pre, t, r_list)

        # Return both corr and corr_mixed
        return x_feats, corr, flow_16, flow_64, flow_256
