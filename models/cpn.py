import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class CPNAlignmentEncoder(nn.Module):
    def __init__(self):
        super(CPNAlignmentEncoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        self.convs.apply(init_weights)

    def forward(self, in_f, in_v):
        x = torch.cat([in_f, in_v], dim=1)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.convs(x)


class CPNAlignmentRegressor(nn.Module):
    def __init__(self):
        super(CPNAlignmentRegressor, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(512, 6)
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))
        self.convs.apply(init_weights)

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x = self.convs(x)
        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(-1, x.shape[1])
        return self.fc(x).view(-1, 2, 3)


class CPNEncoder(nn.Module):
    def __init__(self):
        super(CPNEncoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, in_f, in_v):
        return self.convs(torch.cat([in_f, in_v], dim=1))


class CPNContextMatching(nn.Module):
    def __init__(self):
        super(CPNContextMatching, self).__init__()

    def masked_softmax(self, vec, mask, dim):
        masked_vec = vec * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec - max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros = (masked_sums < 1e-4)
        masked_sums += zeros.float()
        return masked_exps / masked_sums

    def forward(self, c_feats, v_t, v_aligned):
        b, c_c, f, h, w = c_feats.size()

        # Resize the size of the target visibility map
        v_t = (F.interpolate(v_t, size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()

        # Compute visibility map and cosine similarity for each reference frame
        cos_sim, vr_map = [], []
        for r in range(f - 1):
            # Resize the size of the reference visibilty map
            v_r = (F.interpolate(v_aligned[:, :, r], size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()
            vr_map.append(v_r)

            # Computer visibility maps
            vmap = v_t * v_r
            v_sum = vmap[:, 0].sum(-1).sum(-1)
            v_sum_zeros = (v_sum < 1e-4)
            v_sum += v_sum_zeros.float()

            # Computer cosine similarity
            gs = (vmap * c_feats[:, :, 0] * c_feats[:, :, r + 1]).sum(-1).sum(-1).sum(-1) / (v_sum * c_c)
            gs[v_sum_zeros] = 0
            cos_sim.append(torch.ones((b, c_c, h, w)).to(c_feats.device) * gs.view(b, 1, 1, 1))

        # Stack lists into Tensors
        cos_sim = torch.stack(cos_sim, dim=2)
        vr_map = torch.stack(vr_map, dim=2)

        # weighted pixelwise masked softmax
        c_match = self.masked_softmax(cos_sim, vr_map, dim=2)
        c_out = torch.sum(c_feats[:, :, 1:] * c_match, dim=2)

        # c_mask
        c_mask = torch.sum(c_match * vr_map, 2)
        c_mask = 1 - torch.mean(c_mask, 1, keepdim=True)

        return torch.cat([c_feats[:, :, 0], c_out, c_mask], dim=1), c_mask


class CPNDecoder(nn.Module):
    def __init__(self, single_frame=False):
        super(CPNDecoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(128 if single_frame else 257, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=8, dilation=8), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=16, dilation=16), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(257, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.convs_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.convs_3 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.convs(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.convs_2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.convs_3(x)


class CPNet(nn.Module):
    _modes_all = ['full', 'aligner', 'encdec']

    def __init__(self, mode, utils_alignment=None):
        super(CPNet, self).__init__()
        assert mode in self._modes_all
        self.mode = mode
        self.utils_alignment = utils_alignment
        if utils_alignment is None and mode in ['full', 'aligner']:
            self.alignment_encoder = CPNAlignmentEncoder()
            self.alignment_regressor = CPNAlignmentRegressor()
        if mode in ['full', 'encdec']:
            self.encoder = CPNEncoder()
            self.context_matching = CPNContextMatching()
            self.decoder = CPNDecoder(mode == 'encdec')
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

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
        c_feats = self.encoder(
            torch.cat([x_t.unsqueeze(2), x_aligned], dim=2).transpose(1, 2).reshape(-1, c, h, w),
            torch.cat([1 - m_t.unsqueeze(2), v_aligned], dim=2).transpose(1, 2).reshape(-1, 1, h, w)
        )
        c_feats = c_feats.reshape(b, f_ref + 1, c_feats.size(1), c_feats.size(2), c_feats.size(3)).transpose(1, 2)

        # Apply Content-Matching Module
        p_in, c_mask = self.context_matching(c_feats, 1 - m_t, v_aligned)

        # Upscale c_mask to match the size of the mask
        c_mask = (F.interpolate(c_mask, size=(h, w), mode='bilinear', align_corners=False)).detach()

        # Obtain the predicted output y_hat. Clip the output to be between [0, 1]
        y_hat = torch.clamp(self.decoder(p_in) * self.std.squeeze(4) + self.mean.squeeze(4), 0, 1)

        # Combine prediction with GT of the frame.
        y_hat_comp = y_hat * m_t + y_t * (1 - m_t)

        # Return everything
        return y_hat, y_hat_comp, c_mask

    def forward(self, x, m, y, t, r_list):
        # Normalize input data
        x = (x - self.mean) / self.std

        # Propagate using appropiate mode
        if self.mode == 'full':
            y_hat, y_hat_comp, c_mask, (x_aligned, v_aligned) = self._forward_full(x, m, y, t, r_list)
        elif self.mode == 'aligner':
            y_hat, y_hat_comp, c_mask, (x_aligned, v_aligned) = self._forward_aligner(x, m, y, t, r_list)
        else:
            y_hat, y_hat_comp, c_mask, (x_aligned, v_aligned) = self._forward_encdec(x, m, y, t)

        # De-normalize x_aligned, which has been computed using normalized x
        x_aligned = x_aligned * self.std + self.mean if x_aligned is not None else x_aligned

        # Return data
        return y_hat, y_hat_comp, c_mask, (x_aligned, v_aligned)

    def _forward_full(self, x, m, y, t, r_list):
        if self.utils_alignment is None:
            x_aligned, v_aligned, _ = self.align(x, m, y, t, r_list)
        else:
            x_aligned, v_aligned, _ = self.utils_alignment.align(x, m, y, t, r_list)
        y_hat, y_hat_comp, c_mask = self.copy_and_paste(x[:, :, t], m[:, :, t], y[:, :, t], x_aligned, v_aligned)
        return y_hat, y_hat_comp, c_mask, (x_aligned, v_aligned)

    def _forward_aligner(self, x, m, y, t, r_list):
        if self.utils_alignment is None:
            x_aligned, v_aligned, _ = self.align(x, m, y, t, r_list)
        else:
            x_aligned, v_aligned, _ = self.utils_alignment.align(x, m, y, t, r_list)
        return None, None, None, (x_aligned, v_aligned)

    def _forward_encdec(self, x, m, y, t):
        c_feats = self.encoder(x[:, :, t], 1 - m[:, :, t])
        y_hat = self.decoder(c_feats)
        y_hat = torch.clamp((y_hat * self.std.squeeze(4)) + self.mean.squeeze(4), 0, 1)
        y_hat_comp = y_hat * m[:, :, t] + y[:, :, t] * (1 - m[:, :, t])
        return y_hat, y_hat_comp, m.squeeze(2), (x, 1 - m)
