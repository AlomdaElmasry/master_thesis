import torch
import torch.nn as nn
import torch.nn.functional as F


class SEEncoder(nn.Module):
    def __init__(self):
        super(SEEncoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, m):
        x = torch.cat([x, m], dim=1)
        return self.convs(x)


class SEAttention(nn.Module):
    def __init__(self):
        super(SEAttention, self).__init__()
        self.conv = torch.nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(5, 1, 1), padding=(0, 0, 0))
        self.linear = torch.nn.Linear(in_features=64, out_features=5)

    def forward(self, c):
        # Input c: (
        b = F.relu(self.conv(c)).squeeze(2)
        a = self.linear(b.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return F.softmax(a, dim=1).unsqueeze(1)


class SEDecoder(nn.Module):
    def __init__(self):
        super(SEDecoder, self).__init__()
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
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.convs_2(x)
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.convs_3(x)


class CorrelationLayer(nn.Module):

    def __init__(self, padding=20, kernel_size=1, max_displacement=20, stride_1=1, stride_2=2):
        # TODO generilize kernel size (right now just 1)
        # TODO generilize stride_1 (right now just 1), cause there is no downsample layer in pytorch
        super(CorrelationLayer, self).__init__()
        self.pad = padding
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride_1 = stride_1
        self.stride_2 = stride_2

    def forward(self, x_1, x_2):
        """
        Arguments
        ---------
        x_1 : 4D torch.Tensor (bathch channel height width)
        x_2 : 4D torch.Tensor (bathch channel height width)
        """
        x_1 = x_1.transpose(1, 2).transpose(2, 3)
        x_2 = F.pad(x_2, tuple([self.pad for _ in range(4)])).transpose(1, 2).transpose(2, 3)
        mean_x_1 = torch.mean(x_1, 3)
        mean_x_2 = torch.mean(x_2, 3)
        sub_x_1 = x_1.sub(mean_x_1.expand_as(x_1))
        sub_x_2 = x_2.sub(mean_x_2.expand_as(x_2))
        st_dev_x_1 = torch.std(x_1, 3)
        st_dev_x_2 = torch.std(x_2, 3)

        # TODO need optimize
        out_vb = torch.zeros(1)
        _y = 0
        _x = 0
        while _y < self.max_displacement * 2 + 1:
            while _x < self.max_displacement * 2 + 1:
                c_out = (torch.sum(sub_x_1 * sub_x_2[:, _x:_x + x_1.size(1),
                                             _y:_y + x_1.size(2), :], 3) /
                         (st_dev_x_1 * st_dev_x_2[:, _x:_x + x_1.size(1),
                                       _y:_y + x_1.size(2), :])).transpose(2, 3).transpose(1, 2)
                out_vb = torch.cat((out_vb, c_out), 1) if len(out_vb.size()) != 1 else c_out
                _x += self.stride_2
            _y += self.stride_2
        return out_vb


class SqueezeAndExcitationModel(nn.Module):
    use_aligner = None

    def __init__(self, use_aligner=True):
        super(SqueezeAndExcitationModel, self).__init__()
        self.encoder = SEEncoder()
        self.attention = SEAttention()
        self.decoder = SEDecoder()
        self.corr = CorrelationLayer()

        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def forward(self, x, m, y, t, r_list):
        b, c, f, h, w = x.size()

        # Normalize input
        x = (x - self.mean) / self.std

        # Correlation
        my_cor = self.corr(x[:, :, 2], x[:, :, 0])
        a = 1



        # Output c_feats: (b, 128, f, h, w)
        c_feats = self.encoder(x.transpose(1, 2).reshape(-1, c, h, w), m.transpose(1, 2).reshape(-1, 1, h, w))
        c_feats = c_feats.reshape(b, f, c_feats.size(1), c_feats.size(2), c_feats.size(3)).transpose(1, 2)

        # Output att_map: (b, 1, f, h, w). Softmax along dim=2
        att_map = self.attention(c_feats)

        # Decoder input: (b, 128, h, w)
        decoder_input = torch.sum(c_feats * att_map, dim=2)

        # Decode image: (b, h, 2)
        y_hat = self.decoder(decoder_input)

        # Return y_hat and the attention maps
        return y_hat, att_map
