import torch
import torch.nn as nn
import torch.nn.functional as F
import models.cpn_encoders
import models.cpn_decoders
import models.vgg_16


class SeparableConv4d(nn.Module):
    def __init__(self, kernel_size=3, input_dim=1, inter_dim=7, output_dim=1, bias=True, padding=None):
        super().__init__()
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.weight1 = nn.Parameter(torch.zeros(inter_dim, input_dim, *kernel_size), requires_grad=True)
        self.weight2 = nn.Parameter(torch.zeros(output_dim, inter_dim, *kernel_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_dim), requires_grad=True) if bias else None
        self.padding = [k // 2 for k in kernel_size] if padding is None else padding

    def forward(self, x):
        x = x.unsqueeze(4)
        b, t, h, w, c, *_ = x.size()

        # reshape (b*t*H*W, c, H, W)
        # shape is b, t, H*W, inter_dim, H*W then permute
        x2 = F.conv2d(x.reshape(-1, c, h, w), self.weight1, padding=self.padding)
        x2 = x2.reshape(b, t, h * w, x2.size(1), h * w).permute(0, 1, 4, 3, 2)

        # reshape (b*t*H*W, inter_dim, H, W)
        x3 = F.conv2d(x2.reshape(-1, x2.size(3), h, w), self.weight2, bias=self.bias, padding=self.padding)
        x3 = x3.reshape(b, t, h, w, x3.size(1), h, w)  # reshape (b, t, H, W, output_dim, H, W)

        return x3.squeeze(4)


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

    def __init__(self, device, target_size=16):
        super(CorrelationVGG, self).__init__()
        self.target_size = target_size
        self.model_vgg = models.vgg_16.get_pretrained_model(device)
        self.conv = SeparableConv4d()
        self.softmax = Softmax3d()

    def forward(self, x, m, t, r_list):
        b, c, ref_n, h, w = x.size()

        # Get the features of the frames from VGG
        with torch.no_grad():
            x_vgg_feats = self.model_vgg(x.transpose(1, 2).reshape(b * ref_n, c, h, w))
        x_vgg_feats = x_vgg_feats[3].reshape(b, ref_n, -1, 16, 16).transpose(1, 2)

        # Update the parameters to the VGG features
        b, c, ref_n, h, w = x_vgg_feats.size()

        # Get target and reference values
        feats_t = x_vgg_feats[:, :, t]
        feats_ref = x_vgg_feats[:, :, r_list]
        v_t = 1 - m[:, :, t]
        v_ref = 1 - m[:, :, r_list]

        # Resize v_t and v_aligned to be h x w
        v_t = F.interpolate(v_t, size=(h, w), mode='nearest')
        v_ref = F.interpolate(
            v_ref.transpose(1, 2).reshape(-1, 1, v_ref.size(3), v_ref.size(4)), size=(h, w), mode='nearest'
        ).reshape(b, ref_n - 1, 1, h, w).transpose(1, 2)

        # Mask the features
        feats_t, feats_ref = feats_t * v_t, feats_ref * v_ref

        # Compute the correlation with target frame.
        # corr is (b, t, h, w, h, w)
        corr_1 = feats_t.reshape(b, c, -1).transpose(-1, -2).unsqueeze(1)
        corr_1 /= torch.norm(corr_1, dim=3).unsqueeze(3) + 1e-9
        corr_2 = feats_ref.reshape(b, c, ref_n - 1, -1).permute(0, 2, 1, 3)
        corr_2 /= torch.norm(corr_2, dim=2).unsqueeze(2) + 1e-9
        corr = torch.matmul(corr_1, corr_2).reshape(b, ref_n - 1, h, w, h, w)

        # Fill holes in the correlation matrix using a NN
        corr = self.conv(corr)

        # Compute the softmax over each pixel (b, t, h, w, h, w)
        return self.softmax(corr)


class CPNetMatching(nn.Module):

    def __init__(self, device):
        super(CPNetMatching, self).__init__()
        self.encoder = models.cpn_encoders.CPNEncoderDefault()
        self.correlation = CorrelationVGG(device)
        self.decoder = models.cpn_decoders.CPNDecoderDefault(in_c=256)
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
        # c1 = feats_ref.permute(0, 1, 3, 4, 2).reshape(b, c, h * w * ref_n).unsqueeze(3).permute(0, 1, 3, 2)
        # c2 = corr.permute(0, 2, 3, 4, 5, 1).reshape(b, h * w, h * w * ref_n).permute(0, 2, 1).unsqueeze(1)
        # feats = torch.matmul(c1, c2).reshape(b, c, h, w)

        # Mix the features using corr as weight
        decoder_input = torch.zeros((b, feats.size(1), feats.size(3), feats.size(4))).to(x.device)
        for i in range(corr.size(2)):
            for j in range(corr.size(3)):
                pixel_corr = F.interpolate(corr[:, :, i, j], size=(feats.size(3), feats.size(4))).unsqueeze(1)
                pixel_feats = torch.sum(feats[:, :, r_list] * pixel_corr, dim=(2, 3, 4)).view(b, feats.size(1), 1, 1)
                decoder_input[:, :, 4 * i:4 * i + 4, 4 * j:4 * j + 4] = pixel_feats.repeat(1, 1, 4, 4)

        # Decode the output
        y_hat = self.decoder(torch.cat((decoder_input, feats[:, :, t]), dim=1), None)
        y_hat = torch.clamp(y_hat * self.std.squeeze(4) + self.mean.squeeze(4), 0, 1)

        # Combine prediction with GT of the frame.
        y_hat_comp = y_hat * m[:, :, t] + y[:, :, t] * (1 - m[:, :, t])

        # Return everything
        return y_hat, y_hat_comp
