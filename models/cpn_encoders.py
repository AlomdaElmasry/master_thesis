import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Input size: (b, 3, 256, 256)
        # Output size: (b, 128, 64, 64)
        x1 = F.relu(self.conv1(torch.cat([x, v], dim=1)))  # (b, 64, 128, 128)
        x2 = F.relu(self.conv2(x1))  # (b, 64, 128, 128)
        x3 = F.relu(self.conv3(x2))  # (b, 128, 64, 64)
        x4 = F.relu(self.conv4(x3))  # (b, 128, 64, 64)
        x5 = F.relu(self.conv5(x4))  # (b, 128, 64, 64)
        return x5, [x4, x3, x2, x1]
