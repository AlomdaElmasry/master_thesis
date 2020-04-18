import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = F.upsample(x, size=(224, 224), mode='bilinear', align_corners=False)
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
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

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


class CPNOriginalAligner(nn.Module):
    def __init__(self):
        super(CPNOriginalAligner, self).__init__()
        self.A_Encoder = A_Encoder()  # Align
        self.A_Regressor = A_Regressor()  # output: alignment network
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('mean3d', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1,1))

    def encoding(self, frames, holes):
        batch_size, _, num_frames, height, width = frames.size()
        # padding
        (frames, holes), pad = pad_divide_by([frames, holes], 8, (frames.size()[3], frames.size()[4]))

        feat_ = []
        for t in range(num_frames):
            feat = self.A_Encoder(frames[:, :, t], holes[:, :, t])
            feat_.append(feat)
        feats = torch.stack(feat_, dim=2)
        return feats

    def forward(self, x, m, y, t, r_list):
        rfeats = self.encoding(x, m)
        rfeats = rfeats[:, :, r_list]
        rframes = x[:, :, r_list]
        rholes = m[:, :, r_list]
        frame = x[:, :, t]
        hole = m[:, :, t]
        gt = 0

        # Parameters
        batch_size, _, height, width = frame.size()  # B C H W
        num_r = rfeats.size()[2]  # # of reference frames

        # padding
        (rframes, rholes, frame, hole), pad = pad_divide_by([rframes, rholes, frame, hole], 8, (height, width))

        # Target embedding
        tfeat = self.A_Encoder(frame, hole)

        # c_feat: Encoder(Copy Network) features
        # c_feat_ = [self.Encoder(frame, hole)]
        L_align = torch.zeros_like(frame)

        # aligned_r: aligned reference frames
        aligned_r_ = []

        # rvmap: aligned reference frames valid maps
        rvmap_ = []

        for r in range(num_r):
            theta_rt = self.A_Regressor(tfeat, rfeats[:, :, r])
            grid_rt = F.affine_grid(theta_rt, frame.size())

            # aligned_r: aligned reference frame
            # reference frame affine transformation
            aligned_r = F.grid_sample(rframes[:, :, r], grid_rt)

            # aligned_v: aligned reference visiblity map
            # reference mask affine transformation
            aligned_v = F.grid_sample(1 - rholes[:, :, r], grid_rt)
            aligned_v = (aligned_v > 0.5).float()

            aligned_r_.append(aligned_r)

            # intersection of target and reference valid map
            trvmap = (1 - hole) * aligned_v
            # compare the aligned frame - target frame

            # c_feat_.append(self.Encoder(aligned_r, aligned_v))

            rvmap_.append(aligned_v)

        aligned_rs = torch.stack(aligned_r_, 2)
        rvmap = torch.stack(rvmap_, 2)

        return aligned_rs, rvmap, None
