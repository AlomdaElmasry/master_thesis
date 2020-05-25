import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv4d(nn.Module):
    def __init__(self, kernel_size=3, input_dim=1, inter_dim=7, output_dim=1, bias=True, padding=None):
        super().__init__()
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.weight1 = nn.Parameter(torch.zeros(inter_dim, input_dim, *kernel_size), requires_grad=True)
        self.weight2 = nn.Parameter(torch.zeros(output_dim, inter_dim, *kernel_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_dim), requires_grad=True) if bias else None
        self.padding = [k // 2 for k in kernel_size] if padding is None else padding

    def forward(self, x):
        x = x.unsqueeze(4)
        b, t, h, w, c, *_ = x.size()

        # reshape (b*t*H*W, c, H, W)
        # shape is b, t, H*W, inter_dim, H*W then permute
        x2 = F.conv2d(x.reshape(-1, c, h, w), self.weight1, padding=self.padding)
        x2 = x2.reshape(b, t, h * w, x2.size(1), h * w).permute(0, 1, 4, 3, 2)

        # reshape (b*t*H*W, inter_dim, H, W)
        x3 = F.conv2d(x2.reshape(-1, x2.size(3), h, w), self.weight2, bias=self.bias, padding=self.padding)
        x3 = x3.reshape(b, t, h, w, x3.size(1), h, w)  # reshape (b, t, H, W, output_dim, H, W)

        return x3.squeeze(4)
