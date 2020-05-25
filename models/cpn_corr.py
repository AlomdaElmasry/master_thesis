import torch
import torch.nn as nn
import torch.nn.functional as F
import models.cpn_alignment
import models.cpn_encoders
import models.cpn_decoders
import models.cpn_matching
import models.test
import models.softmax3d


class CorrelationMatrix(nn.Module):

    def __init__(self):
        super(CorrelationMatrix, self).__init__()
        self.conv = models.test.SeparableConv4d()
        self.softmax = models.softmax3d.Softmax3d()

    def forward(self, feats_t, feats_ref, v_t, v_ref):
        b, c, ref_n, h, w = feats_ref.size()

        # Resize v_t and v_aligned to be h x w
        v_t = F.interpolate(v_t, size=(h, w), mode='nearest')
        v_ref = F.interpolate(
            v_ref.transpose(1, 2).reshape(-1, 1, v_ref.size(3), v_ref.size(4)), size=(h, w), mode='nearest'
        ).reshape(b, ref_n, 1, h, w).transpose(1, 2)

        # Mask the features
        feats_t, feats_ref = feats_t * v_t, feats_ref * v_ref

        # Compute the correlation with target frame.
        # corr is (b, t, h, w, h, w)
        corr_1 = feats_t.reshape(b, c, -1).transpose(-1, -2).unsqueeze(1)
        corr_2 = feats_ref.reshape(b, c, ref_n, -1).permute(0, 2, 1, 3)
        corr = torch.matmul(corr_1, corr_2).reshape(b, ref_n, h, w, h, w)

        # Fill holes in the correlation matrix using a NN
        corr = self.conv(corr)

        # Compute the softmax over each pixel (b, t, h, w, h, w)
        corr = self.softmax(corr)

        # Mix the features using corr as weight
        c1 = feats_ref.permute(0, 1, 3, 4, 2).reshape(b, c, h * w * ref_n).unsqueeze(3).permute(0, 1, 3, 2)
        c2 = corr.permute(0, 2, 3, 4, 5, 1).reshape(b, h * w, h * w * ref_n).permute(0, 2, 1).unsqueeze(1)
        return torch.matmul(c1, c2).reshape(b, c, h, w)


class CPNetMatching(nn.Module):

    def __init__(self):
        super(CPNetMatching, self).__init__()
        self.encoder = models.cpn_encoders.CPNEncoderPartialConv()
        self.context_matching = CorrelationMatrix()
        self.decoder = models.cpn_decoders.CPNDecoderPartialConv(in_c=128)
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

        # Apply Content-Matching Module feats_t, feats_ref, v_t, v_ref
        decoder_input = self.context_matching(
            feats[:, :, t], feats[:, :, r_list], 1 - m[:, :, t], 1 - m[:, :, r_list]
        )

        # Decode the output
        y_hat = torch.clamp(self.decoder(decoder_input, None) * self.std.squeeze(4) + self.mean.squeeze(4), 0, 1)

        # Combine prediction with GT of the frame.
        y_hat_comp = y_hat * m[:, :, t] + y[:, :, t] * (1 - m[:, :, t])

        # Return everything
        return y_hat, y_hat_comp
