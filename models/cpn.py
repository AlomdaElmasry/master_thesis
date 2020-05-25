import torch
import torch.nn as nn
import torch.nn.functional as F
import models.cpn_alignment
import models.cpn_encoders
import models.cpn_decoders
import models.cpn_matching


class CPNet(nn.Module):
    _modes_all = ['full', 'aligner', 'encdec']

    def __init__(self, mode, utils_alignment=None):
        super(CPNet, self).__init__()
        assert mode in self._modes_all
        self.mode = mode
        self.utils_alignment = utils_alignment
        if utils_alignment is None and mode in ['full', 'aligner']:
            self.alignment_encoder = models.cpn_alignment.CPNAlignmentEncoder()
            self.alignment_regressor = models.cpn_alignment.CPNAlignmentRegressor()
        if mode in ['full', 'encdec']:
            self.context_matching = models.cpn_matching.CorrelationMatrix()
            self._init_encoder_decoder()
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def _init_encoder_decoder(self, version='cpn'):
        # decoder_channels = 128 if single_frame else 257
        self.encoder = models.cpn_encoders.CPNEncoderDefault()
        self.decoder = models.cpn_decoders.CPNDecoderDefault()

    def align(self, x, m, y, t, r_list):
        b, c, f, h, w = x.size()  # B C H W

        # Get alignment features
        r_feats = self.alignment_encoder(x.transpose(1, 2).reshape(-1, c, h, w), m.transpose(1, 2).reshape(-1, 1, h, w))
        r_feats = r_feats.reshape(b, f, r_feats.size(1), r_feats.size(2), r_feats.size(3)).transpose(1, 2)

        # Get alignment grid
        theta_rt = self.alignment_regressor(
            r_feats[:, :, r_list].transpose(1, 2).reshape(-1, r_feats.size(1), r_feats.size(3), r_feats.size(4)),
            r_feats[:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1).transpose(1, 2).reshape(
                -1, r_feats.size(1), r_feats.size(3), r_feats.size(4)
            )
        )
        grid_rt = F.affine_grid(theta_rt, (theta_rt.size(0), c, h, w), align_corners=False)

        # Align x
        x_aligned = F.grid_sample(
            x[:, :, r_list].transpose(1, 2).reshape(-1, c, h, w), grid_rt, align_corners=False
        ).reshape(b, len(r_list), c, h, w).transpose(1, 2)

        # Align v
        v_aligned = (F.grid_sample(
            1 - m[:, :, r_list].transpose(1, 2).reshape(-1, 1, h, w), grid_rt, align_corners=False
        ).reshape(b, len(r_list), 1, h, w).transpose(1, 2) > 0.5).float()

        # Align y
        y_aligned = F.grid_sample(
            y[:, :, r_list].transpose(1, 2).reshape(-1, c, h, w), grid_rt, align_corners=False
        ).reshape(b, len(r_list), c, h, w).transpose(1, 2) if not self.training else None

        # Return stacked GTs
        return x_aligned, v_aligned, y_aligned

    def copy_and_paste(self, x_t, m_t, y_t, x_aligned, v_aligned):
        b, c, f_ref, h, w = x_aligned.size()

        # Get c_features of everything
        c_feats, c_feats_mid = self.encoder(
            torch.cat([x_t.unsqueeze(2), x_aligned], dim=2).transpose(1, 2).reshape(-1, c, h, w),
            torch.cat([1 - m_t.unsqueeze(2), v_aligned], dim=2).transpose(1, 2).reshape(-1, 1, h, w)
        )
        c_feats = c_feats.reshape(b, f_ref + 1, c_feats.size(1), c_feats.size(2), c_feats.size(3)).transpose(1, 2)

        # Reshape and average mid features, if available
        if c_feats_mid is not None:
            c_feats_mid = [
                c_mid.reshape(b, f_ref + 1, c_mid.size(1), c_mid.size(2), c_mid.size(3)).transpose(1, 2)
                for c_mid in c_feats_mid
            ]
            c_feats_mid = [F.avg_pool3d(c_mid, (f_ref + 1, 1, 1)).squeeze(2) for c_mid in c_feats_mid]

        # Apply Content-Matching Module
        p_in, c_mask, ref_importance = self.context_matching(c_feats, 1 - m_t, v_aligned, normalize=False)

        # Upscale c_mask to match the size of the mask
        c_mask = (F.interpolate(c_mask, size=(h, w), mode='bilinear', align_corners=False)).detach()

        # Obtain the predicted output y_hat. Clip the output to be between [0, 1]
        y_hat = torch.clamp(self.decoder(p_in, c_feats_mid) * self.std.squeeze(4) + self.mean.squeeze(4), 0, 1)

        # Combine prediction with GT of the frame.
        y_hat_comp = y_hat * m_t + y_t * (1 - m_t)

        # Return everything
        return y_hat, y_hat_comp, c_mask, ref_importance

    def forward(self, x, m, y, t, r_list):
        if self.utils_alignment is None:
            x = (x - self.mean) / self.std
            x_aligned, v_aligned, _ = self.align(x, m, y, t, r_list)
        elif self.utils_alignment.model_name == 'cpn':
            x = (x - self.mean) / self.std
            x_aligned, v_aligned, _ = self.utils_alignment.align(x, m, y, t, r_list)
        else:
            x_aligned, v_aligned, _ = self.utils_alignment.align(x, m, y, t, r_list)
            x_aligned = (x_aligned - self.mean) / self.std

        # Propagate using appropiate mode
        if self.mode == 'full':
            y_hat, y_hat_comp, c_mask, ref_importance = self.copy_and_paste(
                x[:, :, t], m[:, :, t], y[:, :, t], x_aligned, v_aligned
            )
        elif self.mode == 'encdec':
            c_feats = self.encoder(x[:, :, t], 1 - m[:, :, t])
            y_hat = self.decoder(c_feats)
            y_hat = torch.clamp((y_hat * self.std.squeeze(4)) + self.mean.squeeze(4), 0, 1)
            y_hat_comp = y_hat * m[:, :, t] + y[:, :, t] * (1 - m[:, :, t])
            ref_importance = None
            return y_hat, y_hat_comp, m.squeeze(2), (x, 1 - m)
        else:
            y_hat, y_hat_comp, c_mask, ref_importance = None, None, None, None

        # De-normalize x_aligned, which has been computed using normalized x
        x_aligned = x_aligned * self.std + self.mean if x_aligned is not None else x_aligned

        # Return data
        return y_hat, y_hat_comp, c_mask, ref_importance, (x_aligned, v_aligned)
