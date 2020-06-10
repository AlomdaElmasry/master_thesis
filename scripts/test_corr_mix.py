import torch
import torch.nn.functional as F

b, c, t, h, w = 2, 128, 5, 64, 64
r_list = [0, 1, 3, 4]
feats = torch.zeros((b, c, t, h, w))
corr = torch.zeros((b, t - 1, 16, 16, 16, 16))

# Force the features to be equal to the index they are in
feats[:, :, 1].fill_(1)
feats[:, :, 2].fill_(2)
feats[:, :, 3].fill_(3)
feats[:, :, 4].fill_(4)

# Focus on the features h=w=0.
# Assume a weight of 0.5 for t=1, h'=3 and w'=3 & a weight of 0.5 for t=4, h'=6 and w'=6 (t's adapted to r_list)
# The features of the decoder at h=w=0 should be 0.5*1 + 0.5*4 in all the channels
corr[:, 1, 0, 0, 3, 3].fill_(0.5)
corr[:, 3, 0, 0, 6, 6].fill_(0.5)

# Apply mixing
decoder_input = torch.zeros((b, c, h, w))
for i in range(corr.size(2)):
    for j in range(corr.size(3)):
        # When we upscale the correlation weights from 16x16 to 64x64, we have to normalize them again dividing by 4**2
        pixel_corr = F.interpolate(corr[:, :, i, j], size=(feats.size(3), feats.size(4))).unsqueeze(1) / 4 ** 2

        # Sum the product of the normalized correlation weights by the features. r_list contains the indexes of ref frames
        pixel_feats = torch.sum(feats[:, :, r_list] * pixel_corr, dim=(2, 3, 4)).view(b, feats.size(1), 1, 1)

        # We have now 16x16 different feature maps, instead of 64x64
        decoder_input[:, :, 4 * i:4 * i + 4, 4 * j:4 * j + 4] = pixel_feats.repeat(1, 1, 4, 4)

# Check values of the decoder input at h=w=0
decoder_input_vals = print(decoder_input[0, :, 0, 0])
