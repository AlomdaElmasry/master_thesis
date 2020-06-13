import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    b, f, s, *_ = flow.size()

    # Compute identity position between [-1, 1]
    flow_pos_steps = torch.linspace(-1, 1, s)
    flow_pos_grid_y, flow_pos_grid_x = torch.meshgrid(flow_pos_steps, flow_pos_steps)
    flow_pos_identity = torch.stack((flow_pos_grid_x, flow_pos_grid_y), dim=2).view(1, 1, s, s, 2)

    # Subtract flow - identity to obtain relative flow between [-2, 2]
    return flow - flow_pos_identity.repeat(b, f, 1, 1, 1)


def plot_relative_flow(flow_rel):
    mag, ang = cv2.cartToPolar(flow_rel[:, :, 0].numpy(), flow_rel[:, :, 1].numpy())
    hsv = np.zeros((16, 16, 3), dtype=np.float32)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt.imshow(bgr)
    plt.show()
    a = 1
