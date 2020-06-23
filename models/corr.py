import torch
import torch.nn as nn
import torch.nn.functional as F
import models.vgg_16
import utils.correlation






# class CorrelationModel(nn.Module):
#
#     def __init__(self, device):
#         super(CorrelationModel, self).__init__()
#         self.encoder = models.cpn_encoders.CPNEncoderDefault()
#         self.correlation = CorrelationVGG(device)
#         self.decoder = models.cpn_decoders.CPNDecoderDefault(in_c=128)
#         self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
#         self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))
#
#     def forward(self, x, m, y, t, r_list):
#         b, c, f, h, w = x.size()
#
#         # Normalize the input
#         x = (x - self.mean) / self.std
#
#         # Get features of the frames
#         feats, feats_mid = self.encoder(
#             x.transpose(1, 2).reshape(-1, c, h, w),
#             (1 - m).transpose(1, 2).reshape(-1, 1, h, w)
#         )
#         feats = feats.reshape(b, f, feats.size(1), feats.size(2), feats.size(3)).transpose(1, 2)
#
#         # Apply the CorrelationVGG module
#         corr = self.correlation(x, m, t, r_list)
#
#         # Mix the features using corr as weight
#         c = 128
#         h = w = 64
#         c1 = feats[:, :, r_list].permute(0, 1, 3, 4, 2).reshape(b, c, h * w * (f - 1)).unsqueeze(3).permute(0, 1, 3, 2)
#         c2 = corr.permute(0, 2, 3, 4, 5, 1).reshape(b, h * w, h * w * (f - 1)).permute(0, 2, 1).unsqueeze(1)
#         decoder_input = torch.matmul(c1, c2).reshape(b, c, h, w)
#
#         # Mix the features using corr as weight
#         # decoder_input = torch.zeros((b, feats.size(1), feats.size(3), feats.size(4))).to(x.device)
#         # for i in range(corr.size(2)):
#         #     for j in range(corr.size(3)):
#         #         pixel_corr = F.interpolate(corr[:, :, i, j], size=(feats.size(3), feats.size(4))).unsqueeze(1) / 4 ** 2
#         #         pixel_feats = torch.sum(feats[:, :, r_list] * pixel_corr, dim=(2, 3, 4)).view(b, feats.size(1), 1, 1)
#         #         decoder_input[:, :, 4 * i:4 * i + 4, 4 * j:4 * j + 4] = pixel_feats.repeat(1, 1, 4, 4)
#
#         # Decode the output
#         y_hat = self.decoder(decoder_input, None)
#         y_hat = torch.clamp(y_hat * self.std.squeeze(4) + self.mean.squeeze(4), 0, 1)
#
#         # Combine prediction with GT of the frame.
#         y_hat_comp = y_hat * m[:, :, t] + y[:, :, t] * (1 - m[:, :, t])
#
#         # Return everything
#         return y_hat, y_hat_comp, corr
