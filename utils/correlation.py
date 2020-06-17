import torch


def compute_masked_correlation(features, masks, t, r_list):
    """Computes the normalized correlation between the feature maps of t and r_list.

    Args:
        features (torch.FloatTensor): tensor of size (B,C,F,H,W) containing the feature maps.
        masks (torch.FloatTensor): tensor of size (B,1,F,H,W) containing the masks for each feature map.
        t (int): index of the target feature map.
        r_list (list): list of indexes of the reference feature maps.

    Returns:
        torch.FloatTensor: 4D correlation volume of size (B,F-1,H,W,H,W).
    """
    b, c, f, h, w = features.size()

    # Mask the features
    features *= masks

    # Compute the correlation with target frame.
    corr_1 = features[:, :, t].reshape(b, c, -1).transpose(-1, -2).unsqueeze(1)
    corr_1 /= torch.norm(corr_1, dim=3).unsqueeze(3) + 1e-9
    corr_2 = features[:, :, r_list].reshape(b, c, f - 1, -1).permute(0, 2, 1, 3)
    corr_2 /= torch.norm(corr_2, dim=2).unsqueeze(2) + 1e-9
    corr = torch.matmul(corr_1, corr_2).reshape(b, f - 1, h, w, h, w)

    # Return 4D volume corr
    return corr
