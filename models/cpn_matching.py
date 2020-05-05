import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(vec, mask, dim):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec - max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros = (masked_sums < 1e-4)
    masked_sums += zeros.float()
    return masked_exps / masked_sums


class CPNContextMatchingDefault(nn.Module):
    def __init__(self):
        super(CPNContextMatchingDefault, self).__init__()

    def forward(self, c_feats, v_t, v_aligned):
        b, c_c, f, h, w = c_feats.size()

        # Resize the size of the target visibility map
        v_t = (F.interpolate(v_t, size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()

        # Compute visibility map and cosine similarity for each reference frame
        cos_sim, vr_map = [], []
        for r in range(f - 1):
            # Resize the size of the reference visibilty map
            v_r = (F.interpolate(v_aligned[:, :, r], size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()
            vr_map.append(v_r)

            # Computer visibility maps
            vmap = v_t * v_r

            v_sum = vmap[:, 0].sum(-1).sum(-1)
            v_sum_zeros = (v_sum < 1e-4)
            v_sum += v_sum_zeros.float()

            # Computer cosine similarity
            gs = (vmap * c_feats[:, :, 0] * c_feats[:, :, r + 1]).sum(-1).sum(-1).sum(-1) / (v_sum * c_c)
            gs[v_sum_zeros] = 0
            cos_sim.append(torch.ones((b, c_c, h, w)).to(c_feats.device) * gs.view(b, 1, 1, 1))

        # Stack lists into Tensors
        cos_sim = torch.stack(cos_sim, dim=2)
        vr_map = torch.stack(vr_map, dim=2)

        # weighted pixelwise masked softmax
        c_match = masked_softmax(cos_sim, vr_map, dim=2)
        c_out = torch.sum(c_feats[:, :, 1:] * c_match, dim=2)

        # c_mask
        c_mask = torch.sum(c_match * vr_map, 2)  # The multiplication * vr_map is useless
        c_mask = 1 - torch.mean(c_mask, 1, keepdim=True)  # Used to reduce the channel dimension.

        return torch.cat([c_feats[:, :, 0], c_out, c_mask], dim=1), c_mask


class CPNContextMatchingComplete(nn.Module):
    def __init__(self):
        super(CPNContextMatchingComplete, self).__init__()

    def forward(self, c_feats, v_t, v_aligned):
        b, c_c, f, h, w = c_feats.size()

        # Resize the size of the target visibility map
        v_t = (F.interpolate(v_t, size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()

        # Compute visibility map and cosine similarity for each reference frame
        cos_sim, vr_map = [], []
        for r in range(f - 1):
            # Resize the size of the reference visibilty map
            v_r = (F.interpolate(v_aligned[:, :, r], size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()
            vr_map.append(v_r)

            # Computer visibility maps
            vmap = v_t * v_r

            v_sum = vmap[:, 0].sum(-1).sum(-1)
            v_sum_zeros = (v_sum < 1e-4)
            v_sum += v_sum_zeros.float()

            # Computer cosine similarity
            gs = (vmap * c_feats[:, :, 0] * c_feats[:, :, r + 1]).sum(-1).sum(-1).sum(-1) / (v_sum * c_c)
            gs[v_sum_zeros] = 0
            cos_sim.append(torch.ones((b, c_c, h, w)).to(c_feats.device) * gs.view(b, 1, 1, 1))

        # Stack lists into Tensors
        cos_sim = torch.stack(cos_sim, dim=2)
        vr_map = torch.stack(vr_map, dim=2)

        # weighted pixelwise masked softmax
        c_match = masked_softmax(cos_sim, vr_map, dim=2)
        c_out = torch.sum(c_feats[:, :, 1:] * c_match, dim=2)

        # c_mask
        c_mask = torch.sum(c_match * vr_map, 2)  # The multiplication * vr_map is useless
        c_mask = 1 - torch.mean(c_mask, 1, keepdim=True)  # Used to reduce the channel dimension.

        # Combine reference frames with the ones from auxilliary frames
        decoder_input = c_feats[:, :, 0] * v_t + c_out * (1 - v_t)

        return torch.cat([decoder_input, c_mask], dim=1), c_mask
