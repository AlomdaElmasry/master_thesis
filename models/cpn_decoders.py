import torch
import torch.nn as nn
import torch.nn.functional as F
import models.part_conv


class CPNDecoderDefault(nn.Module):
    def __init__(self, in_c=257):
        super(CPNDecoderDefault, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_c, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
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

    def forward(self, x, c_feats_mid):
        x = self.convs(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.convs_2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.convs_3(x)


class CPNDecoderPartialConv(nn.Module):
    def __init__(self, in_c=128):
        super(CPNDecoderPartialConv, self).__init__()
        self.convs = nn.Sequential(
            models.part_conv.PartialConv2d(in_c, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            models.part_conv.PartialConv2d(257, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            models.part_conv.PartialConv2d(257, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            models.part_conv.PartialConv2d(257, 257, kernel_size=3, stride=1, padding=2, dilation=2), nn.ReLU(),
            models.part_conv.PartialConv2d(257, 257, kernel_size=3, stride=1, padding=4, dilation=4), nn.ReLU(),
            models.part_conv.PartialConv2d(257, 257, kernel_size=3, stride=1, padding=8, dilation=8), nn.ReLU(),
            models.part_conv.PartialConv2d(257, 257, kernel_size=3, stride=1, padding=16, dilation=16), nn.ReLU(),
            models.part_conv.PartialConv2d(257, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            models.part_conv.PartialConv2d(257, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            models.part_conv.PartialConv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.convs_2 = nn.Sequential(
            models.part_conv.PartialConv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            models.part_conv.PartialConv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.convs_3 = models.part_conv.PartialConv2d(64, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, x, c_feats_mid):
        x = self.convs(x)
        x = F.interpolate(x, scale_factor=4, mode='nearest')
        x = self.convs_2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.convs_3(x)


class CPNDecoderU(nn.Module):

    def __init__(self, single_frame=False):
        super(CPNDecoderU, self).__init__()
        self.convs1 = nn.Sequential(
            nn.Conv2d(256 if single_frame else 385, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.convs2 = nn.Sequential(
            nn.Conv2d(385, 257, kernel_size=3, stride=1, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=8, dilation=8), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=16, dilation=16), nn.ReLU(),
            nn.Conv2d(257, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(257, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.convs3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.convs4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.convs5 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, x, c_feats_mid):
        x1 = self.convs1(torch.cat([x, c_feats_mid[0]], dim=1))
        x2 = self.convs2(torch.cat([x1, c_feats_mid[1]], dim=1))
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x3 = self.convs3(torch.cat([x2, c_feats_mid[2]], dim=1))
        x4 = self.convs4(torch.cat([x3, c_feats_mid[3]], dim=1))
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear')
        return self.convs5(x4)
