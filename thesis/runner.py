import skeltorch
from .model import SqueezeAndExcitationModel
import torch
import torch.nn.functional as F


class ThesisRunner(skeltorch.Runner):
    aligner = None

    def init_model(self, device):
        self.model = SqueezeAndExcitationModel()

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train_step(self, it_data, device):
        # Decompose iteration data
        (x, m), y, info = it_data

        # Move data to the correct device
        x = x.to(device)
        m = m.to(device)
        y = y.to(device)

        # Get target and reference frames
        t = x.size(2) // 2
        r_list = list(range(x.size(2)))
        r_list.pop(t)

        # Propagate through the model
        y_hat, att_map = self.model(x, m, y, t, r_list)

        # Return loss
        return self.compute_loss(y[:, :, t], m[:, :, t], y_hat, att_map)

    def compute_loss(self, y, m, t, y_hat, att_map):
        # Get y_t and m_t
        y_t, m_t = y[:, :, t], m[:, :, t]

        # Loss 1: Non-Hole
        loss_nh = F.l1_loss((1 - m_t) * y_hat, (1 - m_t) * y_t, reduction='sum') / torch.sum(1 - m_t)

        # Loss 2: Hole
        loss_nh = F.l1_loss(m_t * y_hat, m_t * y_t, reduction='sum') / torch.sum(m_t)

        # Loss 3: Attention Maps
        a = 1


    def test(self, epoch, device):
        pass
