import torch.nn as nn
import torch.nn.functional as F
import models.corr
import torch


class AlignmentCorrelationMixer(nn.Module):
    def __init__(self, corr_size=16):
        super(AlignmentCorrelationMixer, self).__init__()
        assert corr_size == 16
        self.mixer = nn.Sequential(
            nn.Conv2d(corr_size ** 2, corr_size ** 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(corr_size ** 2, corr_size, kernel_size=3, padding=1), nn.ReLU(),  # Out = 16
            nn.Conv2d(corr_size, corr_size // 2, kernel_size=3, padding=1), nn.ReLU(),  # Out = 8
            nn.Conv2d(corr_size // 2, corr_size // 4, kernel_size=3, padding=1), nn.ReLU(),  # Out = 4
            nn.Conv2d(corr_size // 4, corr_size // 8, kernel_size=3, padding=1), nn.Tanh(),  # Out = 2
        )

    def forward(self, corr):
        b, f, h, w, *_ = corr.size()

        # Reshape corr to be (b, t, h, w, h, w) -> (b * t, -1, h, w)
        corr = corr.reshape(b * f, -1, 16, 16)

        # Apply the mixer NN
        return self.mixer(corr).reshape(b, f, 2, h, w).permute(0, 1, 3, 4, 2)


class AlignmentCorrelationEncoder(nn.Module):
    def __init__(self, corr_size=16):
        super(AlignmentCorrelationEncoder, self).__init__()
        input_c = 3 + 3 + 1 + corr_size ** 2
        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, input_c, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(input_c, input_c * 2, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(input_c * 2, input_c * 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(input_c * 2, input_c * 4, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(input_c * 4, input_c * 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(input_c * 4, input_c * 8, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )
        self.decod_convs_1 = nn.Sequential(
            nn.Conv2d(input_c * 8, input_c * 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(input_c * 8, input_c * 4, kernel_size=3, stride=2), nn.ReLU()
        )

    def forward(self, target_c, reference_c):
        x = torch.cat([target_c, reference_c], dim=1)
        x_encoded = self.encoder(x)
        x_decod_1 = self.decod_convs_1(x_encoded)
        a = 1


class AlignmentCorrelation(nn.Module):

    def __init__(self, device):
        super(AlignmentCorrelation, self).__init__()
        self.corr = models.corr.CorrelationVGG(device)
        self.corr_mixer = AlignmentCorrelationMixer()
        # self.encoder = AlignmentCorrelationEncoder()
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, x, m, t, r_list):
        b, c, f, h, w = x.size()

        # Normalize the input
        x = (x - self.mean) / self.std

        # Apply the CorrelationVGG module. Corr is (b, t, h, w, h, w)
        corr = self.corr(x, m, t, r_list)

        # Mix the corr 4D volume to obtain a 16x16 dense flow estimation of size (b, t, h, w, 2)
        corr_mixed = self.corr_mixer(corr)

        # Return both corr and corr_mixed
        return corr, corr_mixed

        # Upscale the correlation to match (h, w)
        # corr = corr.reshape(b * (f - 1), -1, 16, 16)
        # corr = F.interpolate(corr, (h, w)).reshape(b, f - 1, corr.size(1), h, w).transpose(1, 2)
        #
        # # Concatenate
        # encoder_ref_input = torch.cat((x[:, :, r_list], m[:, :, r_list], corr), dim=1)
        #
        # # Encoder every target-reference pair
        # ref_flows = []
        # for ref_idx in range(encoder_ref_input.size(2)):
        #     ref_flows.append(self.encoder(x[:, :, t], encoder_ref_input[:, :, ref_idx]))
        #
        # a = 1
