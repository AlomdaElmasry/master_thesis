import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def corr_to_flow(corr):
    """Given an input 4D Volume corr, computes the flow using argmax.

    Args:
        corr (torch.FloatTensor): tensor of size (B,F,S,S,S,S) containing ground-truth images.

    Returns:
        torch.FloatTensor: dense flow tensor of size (B,F,S,S,2) in absolute positions between [-1, 1]
    """
    b, f, s, *_ = corr.size()

    # Get maximum similarity block for each pixel
    corr_max = corr.reshape(b * f * s * s, s * s).argmax(dim=1).reshape(b, f, s, s)

    # Transform maximum similarity block (b,f,h,w) to (b,f,h,w,2) -> (x,y)
    corr_max_pos = torch.stack((corr_max % s, corr_max // s), dim=4)

    # Return normalized position between [-1, 1]
    return corr_max_pos * (2 / (s - 1)) - 1


def flow_abs_to_relative(flow):
    """Given a normalized flow between [-1, 1], returns the relative flow between [-2, 2]

    Args:
        flow (torch.FloatTensor): tensor of size (B,F,S,S,2) containing absolute position flows.

    Returns:
        torch.FloatTensor: relative dense flow tensor of size (B,F,S,S,2).
    """
    b, f, h, w, _ = flow.size()

    # Compute identity position between [-1, 1]
    flow_pos_identity = F.affine_grid(
        torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]).unsqueeze(0), [1, 1, h, w], align_corners=True
    ).view(1, 1, h, w, 2)

    # Subtract flow - identity to obtain relative flow between [-2, 2]
    return flow - flow_pos_identity.repeat(b, f, 1, 1, 1)


def flow_relative_to_abs(flow_rel):
    """Given a relative flow between [-2, 2], returns the absolute flow between [-1, 1]

    Args:
        flow (torch.FloatTensor): tensor of size (B,F,S,S,2) containing relative position flows.

    Returns:
        torch.FloatTensor: absolute dense flow tensor of size (B,F,S,S,2).
    """
    b, f, h, w, _ = flow_rel.size()

    # Compute identity position between [-1, 1]
    flow_pos_identity = F.affine_grid(
        torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]).unsqueeze(0), [1, 1, h, w], align_corners=True
    ).view(1, 1, h, w, 2)

    # Add flow + identity to obtain absolute flow between [-1, 1]
    return flow_rel + flow_pos_identity.repeat(b, f, 1, 1, 1)


def crop_flow(flow, crop_size, crop_position):
    """Cuts an absolute flow between at the position `crop_position`.

    Args:
        flow (torch.FloatTensor): tensor of size (B,F,H,W,2) containing absolute position flows.

    Returns:
        torch.FloatTensor: cropped dense flow tensor of size (B,F,H',W',2).
    """
    flow_rel = flow_abs_to_relative(flow)
    flow_rel_cut = flow_rel[:, :, crop_position[0]:crop_position[0] + crop_size[0],
                   crop_position[1]:crop_position[1] + crop_size[1]]
    return flow_relative_to_abs(flow_rel_cut)


def resize_flow(flow, size):
    b, f, h, w, _ = flow.size()
    flow_resized = F.interpolate(flow.reshape(b * f, h, w, 2).permute(0, 3, 1, 2), size)
    return flow_resized.reshape(b, f, 2, size[0], size[1]).permute(0, 1, 3, 4, 2)


# def plot_relative_flow(flow_rel):
#     h, w, _ = flow_rel.size()
#     mag, ang = cv2.cartToPolar(flow_rel[:, :, 0].numpy(), flow_rel[:, :, 1].numpy())
#     hsv = np.zeros((h, w, 3), dtype=np.float32)
#     hsv[..., 0] = ang * 180 / np.pi / 2
#     hsv[..., 1] = 255
#     hsv[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#     plt.imshow(bgr)
#     plt.show()
