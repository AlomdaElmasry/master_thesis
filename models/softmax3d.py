import torch.nn
import torch.nn.functional as F


class Softmax3d(torch.nn.Module):

    def __init__(self):
        super(Softmax3d, self).__init__()

    def forward(self, input):
        assert input.dim() == 6  # Expect (B,T,H1,W1,H,W)

        # Get dimensions
        b, t, h, w, _, _ = input.size()

        # Transform input to be (B,H,W,H*W*T)
        input = input.permute(0, 2, 3, 4, 5, 1).reshape(b, h, w, -1)

        # Apply Softmax
        input = F.softmax(input, dim=3)

        # Restore original dimensions
        return input.reshape(b, h, w, h, w, t).permute(0, 5, 1, 2, 3, 4)
