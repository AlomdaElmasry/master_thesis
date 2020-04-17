import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class AlignmentEncoder(nn.Module):
    def __init__(self):
        super(AlignmentEncoder, self).__init__()
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


class AlignmentRegressor(nn.Module):
    def __init__(self):
        super(AlignmentRegressor, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(512, 6)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))
        self.convs.apply(init_weights)

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x = self.convs(x)
        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(-1, x.shape[1])
        a = self.fc(x).view(-1, 2, 3)
        return a

class ThesisAligner(nn.Module):

    def __init__(self):
        super(ThesisAligner, self).__init__()
        self.A_Encoder = AlignmentEncoder()
        self.A_Regressor = AlignmentRegressor()
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, x, m, y, t, r_list):
        b, c, f, h, w = x.size()  # B C H W

        # Normalize Inputs
        x = (x - self.mean) / self.std

        # Get alignment features
        r_feats = self.A_Encoder(x.transpose(1, 2).reshape(-1, c, h, w), m.transpose(1, 2).reshape(-1, 1, h, w))
        r_feats = r_feats.reshape(b, f, r_feats.size(1), r_feats.size(2), r_feats.size(3)).transpose(1, 2)

        # Get alignment grid
        theta_rt = self.A_Regressor(
            r_feats[:, :, r_list].transpose(1, 2).reshape(-1, r_feats.size(1), r_feats.size(3), r_feats.size(4)),
            r_feats[:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1).transpose(1, 2).reshape(
                -1, r_feats.size(1), r_feats.size(3), r_feats.size(4)
            )
        )
        grid_rt = F.affine_grid(theta_rt, (theta_rt.size(0), c, h, w), align_corners=False)

        # Denormalize outputs
        x = (x * self.std) + self.mean

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
