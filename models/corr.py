import torch
import torch.nn as nn
import torch.nn.functional as F
import models.cpn_encoders
import models.cpn_decoders
import models.vgg_16
import matplotlib.pyplot as plt
import utils.correlation


class SeparableConv4d(nn.Module):
    def __init__(self, kernel_size=3, input_dim=1, inter_dim=128, output_dim=1, bias=True, padding=None):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(input_dim, inter_dim, kernel_size, padding=1)
        self.conv_2 = torch.nn.Conv2d(inter_dim, output_dim, kernel_size, padding=1)

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

    def __init__(self, model_vgg, use_softmax=False):
        super(CorrelationVGG, self).__init__()
        self.model_vgg = model_vgg
        self.conv = SeparableConv4d()
        self.softmax = Softmax3d()
        self.use_softmax = use_softmax

    def forward(self, x, m, t, r_list):
        b, c, ref_n, h, w = x.size()

        # Get the features of the frames from VGG
        with torch.no_grad():
            x_vgg_feats = self.model_vgg(x.transpose(1, 2).reshape(b * ref_n, c, h, w), normalize_input=False)
        x_vgg_feats = x_vgg_feats[3].reshape(b, ref_n, -1, 16, 16).transpose(1, 2)

        # Update the parameters to the VGG features
        b, c, ref_n, h, w = x_vgg_feats.size()

        # Interpolate the feature masks
        corr_masks = F.interpolate(
            1 - m.transpose(1, 2).reshape(b * ref_n, 1, m.size(3), m.size(4)), size=(h, w), mode='nearest'
        ).reshape(b, ref_n, 1, h, w).transpose(1, 2)

        # Compute the feature correlation
        corr = utils.correlation.compute_masked_4d_correlation(x_vgg_feats, corr_masks, t, r_list)

        # Fill holes in the correlation matrix using a NN
        corr = self.conv(corr)

        # Compute the Softmax over each pixel (b, t, h, w, h, w)
        return self.softmax(corr) if self.use_softmax else corr


class CorrelationModel(nn.Module):

    def __init__(self, device):
        super(CorrelationModel, self).__init__()
        self.encoder = models.cpn_encoders.CPNEncoderDefault()
        self.correlation = CorrelationVGG(device)
        self.decoder = models.cpn_decoders.CPNDecoderDefault(in_c=128)
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, x, m, y, t, r_list):
        b, c, f, h, w = x.size()

        # Normalize the input
        x = (x - self.mean) / self.std

        # Get features of the frames
        feats, feats_mid = self.encoder(
            x.transpose(1, 2).reshape(-1, c, h, w),
            (1 - m).transpose(1, 2).reshape(-1, 1, h, w)
        )
        feats = feats.reshape(b, f, feats.size(1), feats.size(2), feats.size(3)).transpose(1, 2)

        # Apply the CorrelationVGG module
        corr = self.correlation(x, m, t, r_list)

        # Mix the features using corr as weight
        c = 128
        h = w = 64
        c1 = feats[:, :, r_list].permute(0, 1, 3, 4, 2).reshape(b, c, h * w * (f - 1)).unsqueeze(3).permute(0, 1, 3, 2)
        c2 = corr.permute(0, 2, 3, 4, 5, 1).reshape(b, h * w, h * w * (f - 1)).permute(0, 2, 1).unsqueeze(1)
        decoder_input = torch.matmul(c1, c2).reshape(b, c, h, w)

        # Mix the features using corr as weight
        # decoder_input = torch.zeros((b, feats.size(1), feats.size(3), feats.size(4))).to(x.device)
        # for i in range(corr.size(2)):
        #     for j in range(corr.size(3)):
        #         pixel_corr = F.interpolate(corr[:, :, i, j], size=(feats.size(3), feats.size(4))).unsqueeze(1) / 4 ** 2
        #         pixel_feats = torch.sum(feats[:, :, r_list] * pixel_corr, dim=(2, 3, 4)).view(b, feats.size(1), 1, 1)
        #         decoder_input[:, :, 4 * i:4 * i + 4, 4 * j:4 * j + 4] = pixel_feats.repeat(1, 1, 4, 4)

        # Decode the output
        y_hat = self.decoder(decoder_input, None)
        y_hat = torch.clamp(y_hat * self.std.squeeze(4) + self.mean.squeeze(4), 0, 1)

        # Combine prediction with GT of the frame.
        y_hat_comp = y_hat * m[:, :, t] + y[:, :, t] * (1 - m[:, :, t])

        # Return everything
        return y_hat, y_hat_comp, corr
