from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, D=1, activation=nn.ReLU()):
        super(Conv2d, self).__init__()
        if activation:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=D),
                activation
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=D)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


def init_He(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array


# Alignment Encoder
class A_Encoder(nn.Module):
    def __init__(self):
        super(A_Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, kernel_size=5, stride=2, padding=2, activation=nn.ReLU())  # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 2
        self.conv23 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())  # 4
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 4
        self.conv34 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())  # 8
        self.conv4a = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 8
        self.conv4b = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 8
        init_He(self)
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        x = self.conv34(x)
        x = self.conv4a(x)
        x = self.conv4b(x)
        return x


# Alignment Regressor
class A_Regressor(nn.Module):
    def __init__(self):
        super(A_Regressor, self).__init__()
        self.conv45 = Conv2d(512, 512, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())  # 16
        self.conv5a = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 16
        self.conv5b = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 16
        self.conv56 = Conv2d(512, 512, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())  # 32
        self.conv6a = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 32
        self.conv6b = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 32
        init_He(self)

        self.fc = nn.Linear(512, 6)
        #self.fc.weight.data.zero_()
        #self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x = self.conv45(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv56(x)
        x = self.conv5a(x)
        x = self.conv5b(x)

        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(-1, x.shape[1])

        theta = self.fc(x)
        theta = theta.view(-1, 2, 3)

        return theta


# Encoder (Copy network)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, kernel_size=5, stride=2, padding=2, activation=nn.ReLU())  # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 2
        self.conv23 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.ReLU())  # 4
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 4
        self.value3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=None)  # 4
        init_He(self)
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        v = self.value3(x)
        return v


# Decoder (Paste network)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv4 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv5_1 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv5_2 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())

        # dilated convolution blocks
        self.convA4_1 = Conv2d(257, 257, kernel_size=3, stride=1, padding=2, D=2, activation=nn.ReLU())
        self.convA4_2 = Conv2d(257, 257, kernel_size=3, stride=1, padding=4, D=4, activation=nn.ReLU())
        self.convA4_3 = Conv2d(257, 257, kernel_size=3, stride=1, padding=8, D=8, activation=nn.ReLU())
        self.convA4_4 = Conv2d(257, 257, kernel_size=3, stride=1, padding=16, D=16, activation=nn.ReLU())

        self.conv3c = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 4
        self.conv3b = Conv2d(257, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 4
        self.conv3a = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 4
        self.conv32 = Conv2d(128, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())  # 2
        self.conv21 = Conv2d(64, 3, kernel_size=5, stride=1, padding=2, activation=None)  # 1

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = self.conv4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.convA4_1(x)
        x = self.convA4_2(x)
        x = self.convA4_3(x)
        x = self.convA4_4(x)

        x = self.conv3c(x)
        x = self.conv3b(x)
        x = self.conv3a(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 2
        x = self.conv32(x)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 2
        x = self.conv21(x)

        p = (x * self.std) + self.mean
        return p


# Context Matching Module
class CM_Module(nn.Module):
    def __init__(self):
        super(CM_Module, self).__init__()

    def masked_softmax(self, vec, mask, dim):
        masked_vec = vec * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec - max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros = (masked_sums < 1e-4)
        masked_sums += zeros.float()
        return masked_exps / masked_sums

    def forward(self, values, tvmap, rvmaps):
        B, C, T, H, W = values.size()
        # t_feat: target feature
        t_feat = values[:, :, 0]
        # r_feats: refetence features
        r_feats = values[:, :, 1:]

        B, Cv, T, H, W = r_feats.size()
        # vmap: visibility map
        # tvmap: target visibility map
        # rvmap: reference visibility map
        # gs: cosine similarity
        # c_m: c_match
        gs_, vmap_ = [], []
        tvmap_t = (F.interpolate(tvmap, size=(H, W), mode='bilinear', align_corners=False) > 0.5).float()
        for r in range(T):
            rvmap_t = (F.interpolate(rvmaps[:, :, r], size=(H, W), mode='bilinear') > 0.5).float()
            # vmap: visibility map
            vmap = tvmap_t * rvmap_t
            gs = (vmap * t_feat * r_feats[:, :, r]).sum(-1).sum(-1).sum(-1)
            # valid sum
            v_sum = vmap[:, 0].sum(-1).sum(-1)
            zeros = (v_sum < 1e-4)
            gs[zeros] = 0
            v_sum += zeros.float()
            gs = gs / v_sum / C
            gs = torch.ones(t_feat.shape).float().to(values.device) * gs.view(B, 1, 1, 1)
            gs_.append(gs)
            vmap_.append(rvmap_t)

        gss = torch.stack(gs_, dim=2)
        vmaps = torch.stack(vmap_, dim=2)

        # weighted pixelwise masked softmax
        c_match = self.masked_softmax(gss, vmaps, dim=2)
        c_out = torch.sum(r_feats * c_match, dim=2)

        # c_mask
        c_mask = (c_match * vmaps)
        c_mask = torch.sum(c_mask, 2)
        c_mask = 1. - (torch.mean(c_mask, 1, keepdim=True))

        return torch.cat([t_feat, c_out, c_mask], dim=1), c_mask


class CPNet(nn.Module):
    def __init__(self):
        super(CPNet, self).__init__()
        # Align Modules
        self.A_Encoder = A_Encoder()
        self.A_Regressor = A_Regressor()  # output: alignment network

        # Copy and Paste Modules
        self.Encoder = Encoder()
        self.CM_Module = CM_Module()
        self.Decoder = Decoder()

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('mean3d', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))

    def align(self, x, m, y, t, r_list):
        b, c, f, h, w = x.size()  # B C H W

        # Add padding to everything
        (x, m, y), pad = pad_divide_by([x, m, y], 8, (h, w))

        # Get alignment features
        r_feats = torch.stack([self.A_Encoder(x[:, :, i], m[:, :, i]) for i in range(f)], dim=2)

        # List to store the aligned GTs
        x_aligned = []
        v_aligned = []
        y_aligned = []

        # Iterate over the reference frames
        for r in r_list:
            # Predict Affine transformation and created grid
            theta_rt = self.A_Regressor(r_feats[:, :, t], r_feats[:, :, r])
            grid_rt = F.affine_grid(theta_rt, (b, c, h, w), align_corners=False)

            # Align x, v and y
            x_aligned.append(F.grid_sample(x[:, :, r], grid_rt, align_corners=False))
            v_aligned.append(F.grid_sample(1 - m[:, :, r], grid_rt, mode='nearest'))
            y_aligned.append(F.grid_sample(y[:, :, r], grid_rt, align_corners=False))

        # Return stacked GTs
        return torch.stack(x_aligned, dim=2), torch.stack(v_aligned, dim=2), torch.stack(y_aligned, dim=2)

    def copy_and_paste(self, x_t, m_t, y_t, x_aligned, v_aligned):
        b, c, h, w = x_t.size()  # B C H W

        # Get c_features of everything
        c_feats = torch.stack([self.Encoder(x_t, m_t)] + [
            self.Encoder(x_aligned[:, :, i], v_aligned[:, :, i]) for i in range(x_aligned.size(2))
        ], dim=2)

        # Apply Content-Matching Module
        p_in, c_mask = self.CM_Module(c_feats, 1 - m_t, v_aligned)

        # Upscale c_mask to match the size of the mask
        c_mask = (F.interpolate(c_mask, size=(h, w), mode='bilinear', align_corners=False))

        # Obtain the predicted output y_hat. Clip the output to be between [0, 1
        y_hat = torch.clamp(self.Decoder(p_in), 0, 1)

        # Combine prediction with GT of the frame. Limit the output range [0, 1].
        y_hat_comp = y_hat * m_t + y_t * (1 - m_t)

        # Return everything
        return y_hat, y_hat_comp, c_mask

    def forward(self, x, m, y, t, r_list):
        x_aligned, v_aligned, _ = self.align(x, m, y, t, r_list)
        y_hat, y_hat_comp, c_mask = self.copy_and_paste(x[:, :, t], m[:, :, t], y[:, :, t], x_aligned, v_aligned)
        return y_hat, y_hat_comp, c_mask, (x_aligned, v_aligned)
