import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.correlation


class FrameEncoder(nn.Module):

    def __init__(self, res):
        super(FrameEncoder, self).__init__()
        in_c = 4 if res == '16' else 5
        self.convs = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=4, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, enc_input):
        return self.convs(enc_input)


class FrameDecoder(nn.Module):
    def __init__(self, res):
        super(FrameDecoder, self).__init__()
        in_c = 256
        self.convs = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=16, dilation=16), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, dec_input):
        return self.convs(dec_input)


class FrameEncoderSkip(nn.Module):
    def __init__(self, res):
        super(FrameEncoderSkip, self).__init__()
        in_c = 4 if res == '16' else 5
        self.convs_1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, in_c * 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=7, padding=3), nn.ReLU(),
        )
        self.convs_2 = nn.Sequential(
            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c * 2, in_c * 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 2, in_c * 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 4, in_c * 4, kernel_size=3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(in_c * 4, in_c * 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 4, in_c * 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 4, in_c * 4, kernel_size=5, padding=2), nn.ReLU()
        )
        self.convs_3 = nn.Sequential(
            nn.Conv2d(in_c * 4, in_c * 4, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c * 4, in_c * 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 4, in_c * 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 8, in_c * 8, kernel_size=3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(in_c * 8, in_c * 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 8, in_c * 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c * 8, in_c * 8, kernel_size=5, padding=2), nn.ReLU()
        )

    def forward(self, enc_input):
        enc_out_x1 = self.convs_1(enc_input)
        enc_out_x2 = self.convs_2(enc_out_x1)
        enc_out_x3 = self.convs_3(enc_out_x2)
        return [enc_out_x1, enc_out_x2, enc_out_x3]


class FrameDecoderSkip(nn.Module):
    def __init__(self, res):
        super(FrameDecoderSkip, self).__init__()
        in_c = 64 if res == '16' else 80
        self.convs_1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(in_c // 2, in_c // 2, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=5, padding=2), nn.ReLU()
        )
        self.convs_2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(in_c // 2, in_c // 2, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c // 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c // 4, in_c // 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c // 4, in_c // 4, kernel_size=5, padding=2), nn.ReLU()
        )
        self.convs_3 = nn.Sequential(
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c // 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c // 4, in_c // 4, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c // 4, in_c // 4, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c // 4, in_c // 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c // 8, in_c // 8, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(in_c // 8, in_c // 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_c // 8, 3, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1), nn.ReLU()
        )

    def forward(self, dec_input):
        enc_out_x1 = self.convs_1(dec_input[2])
        enc_out_x2 = self.convs_2(torch.cat([enc_out_x1, dec_input[1]], dim=1))
        return self.convs_3(torch.cat([enc_out_x2, dec_input[0]], dim=1))


class ThesisInpaintingModel(nn.Module):
    enc_handlers = None

    def __init__(self, model_vgg):
        super(ThesisInpaintingModel, self).__init__()
        self.enc_16, self.enc_64, self.enc_256 = FrameEncoder('16'), FrameEncoder('64'), FrameEncoder('256')
        self.dec_16, self.dec_64, self.dec_256 = FrameDecoder('16'), FrameDecoder('64'), FrameDecoder('256')
        self.enc_handlers = {'16': self.enc_16, '64': self.enc_64, '256': self.enc_256}
        self.dec_handlers = {'16': self.dec_16, '64': self.dec_64, '256': self.dec_256}
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))

    def get_n_params(self, model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def forward(self, xs_target, vs_target, ys_target, xs_aligned, vs_aligned, v_maps):
        # Predict the inpainting at 16x16 resolution
        y_hat_16, y_hat_comp_16 = self.inpaint_resolution(
            xs_target[0], vs_target[0], ys_target[0], xs_aligned[0], vs_aligned[0], None, res='16'
        )

        # Interpolate y_hat_comp_16 to be 64x64
        y_hat_comp_64_pre = F.interpolate(y_hat_comp_16, (64, 64), mode='bilinear')

        # Predict the inpainting at 64x64 resolution
        y_hat_64, y_hat_comp_64 = self.inpaint_resolution(
            y_hat_comp_64_pre, vs_target[1], ys_target[1], xs_aligned[1], vs_aligned[1], v_maps[0], res='64'
        )

        # Interpolate y_hat_64 to be 256x256
        y_hat_comp_256_pre = F.interpolate(y_hat_comp_64, (256, 256), mode='bilinear')

        # Predict the inpainting at 64x64 resolution
        y_hat_256, y_hat_comp_256 = self.inpaint_resolution(
            y_hat_comp_256_pre, vs_target[2], ys_target[2], xs_aligned[2], vs_aligned[2], v_maps[1], res='256'
        )

        # Return inpainting at multiple resolutions
        return y_hat_16, y_hat_comp_16, y_hat_64, y_hat_comp_64, y_hat_256, y_hat_comp_256

    def inpaint_resolution(self, xs_target, vs_target, ys_target, xs_aligned, vs_aligned, v_maps, res):
        b, c, f, h, w = xs_aligned.size()

        # Normalize input
        xs_target = (xs_target - self.mean.squeeze(2)) / self.std.squeeze(2)
        xs_aligned = (xs_aligned - self.mean) / self.std

        # Prepare the input of the encoder
        enc_input = torch.cat([
            torch.cat([xs_target.unsqueeze(2), xs_aligned], dim=2),
            torch.cat([vs_target.unsqueeze(2), vs_aligned], dim=2)
        ], dim=1)
        enc_input = torch.cat([
            enc_input, torch.cat([torch.zeros_like(vs_target).unsqueeze(2), v_maps], dim=2),
        ], dim=1) if v_maps is not None else enc_input

        # Encode input frames
        enc_output = self.enc_handlers[res](enc_input.transpose(1, 2).reshape(b * (f + 1), -1, h, w)) \
            .reshape(b, f + 1, -1, h // 4, w // 4).transpose(1, 2)

        # Interpolate masks to perform the matching
        vs_target_res = F.interpolate(vs_target, (h // 4, w // 4))
        vs_aligned_res = F.interpolate(
            vs_aligned.transpose(1, 2).reshape(b * f, -1, h, w), (h // 4, w // 4)
        ).reshape(b, f, -1, h // 4, w // 4).transpose(1, 2)

        # Compute the matching
        matching = ThesisInpaintingModel.compute_matching(
            enc_output[:, :, 0], enc_output[:, :, 1:], vs_target_res, vs_aligned_res
        )

        # Mix the features of the reference frames
        dec_input = torch.cat([enc_output[:, :, 0], (enc_output[:, :, 1:] * matching).sum(2)], dim=1)

        # Predict the inpainted frame and de-normalize
        y_hat = self.dec_handlers[res](dec_input) * self.std.squeeze(2) + self.mean.squeeze(2)
        y_hat_comp = ys_target * vs_target + y_hat * (1 - vs_target)

        # Return both y_hat and y_hat_comp
        return y_hat, y_hat_comp

    def inpaint_resolution_skip(self, xs_target, vs_target, ys_target, xs_aligned, vs_aligned, v_maps, res):
        b, c, f, h, w = xs_aligned.size()

        # Normalize input
        xs_target = (xs_target - self.mean.squeeze(2)) / self.std.squeeze(2)
        xs_aligned = (xs_aligned - self.mean) / self.std

        # Prepare the input of the encoder
        enc_input = torch.cat([
            torch.cat([xs_target.unsqueeze(2), xs_aligned], dim=2),
            torch.cat([vs_target.unsqueeze(2), vs_aligned], dim=2)
        ], dim=1)
        enc_input = torch.cat([
            enc_input, torch.cat([torch.zeros_like(vs_target).unsqueeze(2), v_maps], dim=2),
        ], dim=1) if v_maps is not None else enc_input

        # Encode input frames
        enc_output = self.enc_handlers[res](enc_input.transpose(1, 2).reshape(b * (f + 1), -1, h, w))
        enc_output[0] = enc_output[0].reshape(b, f + 1, -1, h, w).transpose(1, 2)
        enc_output[1] = enc_output[1].reshape(b, f + 1, -1, h // 2, w // 2).transpose(1, 2)
        enc_output[2] = enc_output[2].reshape(b, f + 1, -1, h // 4, w // 4).transpose(1, 2)

        # Interpolate masks to perform the matching
        vs_target_x1, vs_aligned_x1 = vs_target, vs_aligned
        vs_target_x2 = F.interpolate(vs_target, (h // 2, w // 2))
        vs_aligned_x2 = F.interpolate(
            vs_aligned.transpose(1, 2).reshape(b * f, -1, h, w), (h // 2, w // 2)
        ).reshape(b, f, -1, h // 2, w // 2).transpose(1, 2)
        vs_target_x3 = F.interpolate(vs_target, (h // 4, w // 4))
        vs_aligned_x3 = F.interpolate(
            vs_aligned.transpose(1, 2).reshape(b * f, -1, h, w), (h // 4, w // 4)
        ).reshape(b, f, -1, h // 4, w // 4).transpose(1, 2)

        # Compute the matching in the different sub-resolutions
        matching = list()
        matching.append(ThesisInpaintingModel.compute_matching(
            enc_output[0][:, :, 0], enc_output[0][:, :, 1:], vs_target_x1, vs_aligned_x1)
        )
        matching.append(ThesisInpaintingModel.compute_matching(
            enc_output[1][:, :, 0], enc_output[1][:, :, 1:], vs_target_x2, vs_aligned_x2)
        )
        matching.append(ThesisInpaintingModel.compute_matching(
            enc_output[2][:, :, 0], enc_output[2][:, :, 1:], vs_target_x3, vs_aligned_x3)
        )

        # Mix the features in the different sub-resolutions
        dec_input = list()
        dec_input.append(torch.cat([enc_output[0][:, :, 0], (enc_output[0][:, :, 1:] * matching[0]).sum(2)], dim=1))
        dec_input.append(torch.cat([enc_output[1][:, :, 0], (enc_output[1][:, :, 1:] * matching[1]).sum(2)], dim=1))
        dec_input.append(torch.cat([enc_output[2][:, :, 0], (enc_output[2][:, :, 1:] * matching[2]).sum(2)], dim=1))

        # Predict the inpainted frame and de-normalize
        y_hat = self.dec_handlers[res](dec_input) * self.std.squeeze(2) + self.mean.squeeze(2)
        y_hat_comp = ys_target * vs_target + y_hat * (1 - vs_target)

        # Return both y_hat and y_hat_comp
        return y_hat, y_hat_comp

    @staticmethod
    def compute_matching(feats_target, feats_references, mask_target, mask_references):
        matching_maps = []
        for i in range(feats_references.size(2)):
            matching_maps.append(utils.correlation.compute_masked_correlation(
                feats_target, feats_references[:, :, i], mask_target, mask_references[:, :, i]
            ))
        return F.softmax(torch.stack(matching_maps, dim=2), dim=2)
