import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, in_f, in_v):
        x = torch.cat([in_f, in_v], dim=1)
        return self.convs(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(128, 257, kernel_size=3, stride=1, padding=1), nn.ReLU(),
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
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = self.convs(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.convs_2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.convs_3(x)
        return (x * self.std) + self.mean


class CPNEncoderDecoder(nn.Module):
    use_aligner = None

    def __init__(self):
        super(CPNEncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))
        pass

    def forward(self, x, m, y):
        # Normalize input x
        x = (x - self.mean) / self.std

        # Encoder
        c_feats = self.encoder(x[:, :, 0], 1 - m[:, :, 0])

        # Decoder
        y_hat = self.decoder(c_feats)

        # Up-normalize output. Obtain the predicted output y_hat. Clip the output to be between [0, 1]
        y_hat = torch.clamp((y_hat.unsqueeze(2) * self.std) + self.mean, 0, 1)
        y_hat = y_hat.squeeze(2)

        # Combine prediction with GT of the frame. Limit the output range [0, 1].
        y_hat_comp = y_hat * m[:, :, 0] + y[:, :, 0] * (1 - m[:, :, 0])

        # Return everything
        return y_hat, y_hat_comp