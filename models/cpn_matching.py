import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.transforms
import matplotlib.pyplot as plt


def masked_softmax(vec, mask, dim):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec - max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros = (masked_sums < 1e-4)
    masked_sums += zeros.float()
    return masked_exps / masked_sums


def add_match_border(v_t, c_match):
    v_t_border = ((utils.transforms.ImageTransforms.dilatate(
        1 - v_t.transpose(0, 1).cpu(), (3, 3), 1
    ).transpose(0, 1) - (1 - v_t)) > 0).float()
    ref_importance = F.interpolate(
        c_match[:, 0], size=(v_t.size(2), v_t.size(3)), mode='bilinear', align_corners=False
    )
    ref_importance = torch.cat(
        (ref_importance[:, :c_match.size(2) // 2], torch.zeros((v_t.size(0), 1, v_t.size(2), v_t.size(3))),
         ref_importance[:, c_match.size(2) // 2:]), dim=1
    ).unsqueeze(1).repeat(1, 3, 1, 1, 1)
    b1 = ref_importance * (1 - v_t_border.unsqueeze(2))
    b2 = v_t_border.unsqueeze(2).repeat(1, 3, 1, 1, 1) * torch.tensor([1, 0, 0]).view(1, 3, 1, 1, 1)
    return b1 + b2


def add_border(x, mask):
    border = ((utils.transforms.ImageTransforms.dilatate(mask.transpose(0, 1).cpu(), (3, 3), 1)
               .transpose(0, 1).to(x.device) - mask) > 0).float()
    b1 = x * (1 - border.unsqueeze(2))
    b2 = border.unsqueeze(2).repeat(1, 3, 1, 1, 1) * torch.tensor([1, 0, 0]).view(1, 3, 1, 1, 1).to(x.device)
    return b1 + b2


class CPNContextMatchingDefault(nn.Module):
    def __init__(self):
        super(CPNContextMatchingDefault, self).__init__()

    def forward(self, c_feats, v_t, v_aligned, normalize=True):
        b, c_c, f, h, w = c_feats.size()

        # Resize the size of the target visibility map
        v_t_resized = (F.interpolate(v_t, size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()

        # Compute visibility map and cosine similarity for each reference frame
        cos_sim, vr_map = [], []
        for r in range(f - 1):
            # Resize the size of the reference visibilty map
            v_r = (F.interpolate(v_aligned[:, :, r], size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()
            vr_map.append(v_r)

            # Computer visibility maps
            vmap = v_t_resized * v_r

            v_sum = vmap[:, 0].sum(-1).sum(-1)
            v_sum_zeros = (v_sum < 1e-4)
            v_sum += v_sum_zeros.float()

            # Computer cosine similarity
            gs_norm = (v_sum * c_c) if normalize else 1
            gs = (vmap * c_feats[:, :, 0] * c_feats[:, :, r + 1]).sum(-1).sum(-1).sum(-1) / gs_norm
            gs[v_sum_zeros] = 0
            cos_sim.append(torch.ones((b, c_c, h, w)).to(c_feats.device) * gs.view(b, 1, 1, 1))

        # Stack lists into Tensors
        cos_sim = torch.stack(cos_sim, dim=2)
        vr_map = torch.stack(vr_map, dim=2)

        # weighted pixelwise masked softmax
        c_match = masked_softmax(cos_sim, vr_map, dim=2)
        c_out = torch.sum(c_feats[:, :, 1:] * c_match, dim=2)

        # Lala
        ref_importance = add_match_border(v_t, c_match)

        # c_mask
        c_mask = torch.sum(c_match * vr_map, 2)  # The multiplication * vr_map is useless
        c_mask = 1 - torch.mean(c_mask, 1, keepdim=True)  # Used to reduce the channel dimension.

        return torch.cat([c_feats[:, :, 0], c_out, c_mask], dim=1), c_mask, ref_importance


class CPNContextMatchingComplete(nn.Module):
    def __init__(self):
        super(CPNContextMatchingComplete, self).__init__()

    def forward(self, c_feats, v_t, v_aligned, normalize=True):
        b, c_c, f, h, w = c_feats.size()

        # Resize the size of the target visibility map
        v_t_resized = (F.interpolate(v_t, size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()

        # Compute visibility map and cosine similarity for each reference frame
        cos_sim, vr_map = [], []
        for r in range(f - 1):
            # Resize the size of the reference visibilty map
            v_r = (F.interpolate(v_aligned[:, :, r], size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()
            vr_map.append(v_r)

            # Computer visibility maps
            vmap = v_t_resized * v_r

            v_sum = vmap[:, 0].sum(-1).sum(-1)
            v_sum_zeros = (v_sum < 1e-4)
            v_sum += v_sum_zeros.float()

            # Computer cosine similarity
            gs_norm = (v_sum * c_c) if normalize else 1
            gs = (vmap * c_feats[:, :, 0] * c_feats[:, :, r + 1]).sum(-1).sum(-1).sum(-1) / gs_norm
            gs[v_sum_zeros] = 0
            cos_sim.append(torch.ones((b, c_c, h, w)).to(c_feats.device) * gs.view(b, 1, 1, 1))

        # Stack lists into Tensors
        cos_sim = torch.stack(cos_sim, dim=2)
        vr_map = torch.stack(vr_map, dim=2)

        # Weighted pixelwise masked softmax. Use only inside the region of the hole
        c_match = masked_softmax(cos_sim, vr_map, dim=2)
        c_match *= (1 - v_t_resized).unsqueeze(2).repeat(1, 1, c_match.size(2), 1, 1)
        c_out = torch.sum(c_feats[:, :, 1:] * c_match, dim=2)

        # Lala
        ref_importance = F.interpolate(
            c_match[:, 0], size=(v_t.size(2), v_t.size(3)), mode='bilinear', align_corners=False
        )
        ref_importance = torch.cat(
            (ref_importance[:, :c_match.size(2) // 2], v_t, ref_importance[:, c_match.size(2) // 2:]), dim=1
        ).unsqueeze(1).repeat(1, 3, 1, 1, 1)

        # Add the border to ref_importance
        ref_importance = add_border(ref_importance, 1 - v_t)

        # c_mask
        c_mask = torch.sum(c_match * vr_map, 2)  # The multiplication * vr_map is useless
        c_mask = 1 - torch.mean(c_mask, 1, keepdim=True)  # Used to reduce the channel dimension.

        # Combine reference frames with the ones from auxilliary frames
        decoder_input = c_feats[:, :, 0] * v_t_resized + c_out

        return torch.cat([decoder_input, c_mask], dim=1), c_mask, ref_importance


# class CorrelationMatrix(nn.Module):
#
#     def __init__(self):
#         super(CorrelationMatrix, self).__init__()
#         self.conv = models.test.SeparableConv4d()
#         self.softmax = models.softmax3d.Softmax3d()
#
#     def forward(self, c_feats, v_t, v_aligned, normalize=True):
#         b, c, f, h, w = c_feats.size()
#
#         # Resize v_t and v_aligned to be h x w
#         v_t = F.interpolate(v_t, size=(h, w), mode='nearest')
#         v_aligned = F.interpolate(
#             v_aligned.transpose(1, 2).reshape(-1, 1, v_aligned.size(3), v_aligned.size(4)), size=(h, w), mode='nearest'
#         ).reshape(b, f - 1, 1, h, w).transpose(1, 2)
#
#         # Mask the features
#         c_feats[:, :, 0] = c_feats[:, :, 0] * v_t
#         c_feats[:, :, 1:] = c_feats[:, :, 1:] * v_aligned
#
#         # Compute the correlation with target frame. Mask references first (v_aligned).
#         # The features of the target frame are at c_feats[b,c,0]
#         # corr is (b, t, h, w, h, w)
#         corr_1 = c_feats[:, :, 0].reshape(b, c, -1).transpose(-1, -2)
#         corr_2 = c_feats[:, :, 1:].reshape(b, c, f - 1, -1).permute(0, 2, 1, 3)
#         corr = torch.matmul(corr_1, corr_2).reshape(b, f - 1, h, w, h, w)
#
#         # Fill holes in the correlation matrix using a NN
#         corr = self.conv(corr)
#
#         # Compute the softmax over each pixel (b, t, h, w, h, w)
#         corr = self.softmax(corr)
#
#         # Verify that the implementation of Softmax is correct. Sum over b=0, h=0, w=0
#         # sum_pixelwise = torch.sum(corr[0, :, 0, 0])  # Output: 1.0000
#
#         # Expand c_feats to be (b, c, t, h, w, h, w)
#         c1 = c_feats[:, :, 1:].permute(0, 1, 3, 4, 2).reshape(b, c, h * w * (f - 1)).unsqueeze(3).permute(0, 1, 3, 2)
#         c2 = corr.permute(0, 2, 3, 4, 5, 1).reshape(b, h * w, h * w * (f - 1)).permute(0, 2, 1).unsqueeze(1)
#         decoder_input = torch.matmul(c1, c2).reshape(b, c, h, w)
#
#         # Combine the features using corr to get the input of the decoder
#         # decoder_input = torch.zeros((b, c, h, w))
#         # for i in range(h):
#         #     for j in range(w):
#         #         decoder_input[:, :, i, j] = torch.sum(
#         #             c_feats[:, :, 1:] * corr[:, :, i, j].unsqueeze(1), dim=(2, 3, 4)
#         #         )
#
#         # Return decoder_input (b, c, h, w)
#         return decoder_input
