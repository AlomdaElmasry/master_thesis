import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class CPNEncoderDefault(nn.Module):
    def __init__(self):
        super(CPNEncoderDefault, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, v):
        # Input size: (b, 3, 256, 256)
        # Output size: (b, 128, 64, 64)
        return self.convs(torch.cat([x, v], dim=1))


class CPNEncoderU(nn.Module):
    def __init__(self):
        super(CPNEncoderU, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    def forward(self, x, v):
        x1 = F.relu(self.conv1(torch.cat([x, v], dim=1)))  # (b, 64, 128, 128)
        x2 = F.relu(self.conv2(x1))  # (b, 64, 128, 128)
        x3 = F.relu(self.conv3(x2))  # (b, 128, 64, 64)
        x4 = F.relu(self.conv4(x3))  # (b, 128, 64, 64)
        x5 = F.relu(self.conv5(x4))  # (b, 128, 64, 64)
        return x5, [x4, x3, x2, x1]


class CPNEncoderResidual(nn.Module):
    def __init__(self):
        super(CPNEncoderResidual, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    def forward(self, x, v):
        x1 = F.relu(self.conv1(torch.cat([x, v], dim=1)))  # (b, 64, 128, 128)
        x2 = F.relu(self.conv2(x1))  # (b, 64, 128, 128)
        x2 += x1
        x3 = F.relu(self.conv3(x2))  # (b, 128, 64, 64)
        x4 = F.relu(self.conv4(x3))  # (b, 128, 64, 64)
        x4 += x3
        x5 = F.relu(self.conv5(x4))  # (b, 128, 64, 64)
        return x5, None


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        print('ey')
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 4569616
        # 46267072

    def forward(self, x, v):
        a = self.get_n_params(self)
        fea = self.conv_first(torch.cat([x, v], dim=1))
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out

    def get_n_params(self, model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
