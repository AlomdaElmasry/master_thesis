import skeltorch
import torch.optim
import models.alignment_corr
import torch.nn.functional as F
import utils.losses


class ThesisAlignmentRunner(skeltorch.Runner):
    utils_losses = None

    def init_model(self, device):
        self.model = models.alignment_corr.AlignmentCorrelation(device).to(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.experiment.configuration.get('training', 'lr')
        )

    def init_others(self, device):
        self.utils_losses = utils.losses.LossesUtils(device)

    def train_step(self, it_data, device):
        # Decompose iteration data
        (x, m), y, info = it_data
        b, c, f, h, w = x.size()

        # Move Tensors to correct device
        x = x.to(device)
        m = m.to(device)
        y = y.to(device)

        # Propagate through the model
        t = x.size(2) // 2
        r_list = list(range(x.size(2)))
        r_list.pop(t)
        corr, corr_mixed = self.model(x, m, t, r_list)

        # Interpolate data to be 16x16
        x_down = F.interpolate(
            x.transpose(1, 2).reshape(-1, c, h, w), (16, 16)
        ).reshape(b, f, c, 16, 16).transpose(1, 2)
        m_down = F.interpolate(
            m.transpose(1, 2).reshape(-1, 1, h, w), (16, 16)
        ).reshape(b, f, 1, 16, 16).transpose(1, 2)

        # Apply dense flow to both the reference frames and their masks
        x_down_aligned = F.grid_sample(
            x_down[:, :, r_list].transpose(1, 2).reshape(-1, c, 16, 16), corr_mixed.reshape(-1, 16, 16, 2)
        ).reshape(b, -1, 3, 16, 16).transpose(1, 2)

        m_down_aligned = F.grid_sample(
            m_down[:, :, r_list].transpose(1, 2).reshape(-1, 1, 16, 16), corr_mixed.reshape(-1, 16, 16, 2)
        ).reshape(b, -1, 1, 16, 16).transpose(1, 2)

        # Return the loss
        return self.loss_function(x_down[:, :, t], m_down[:, :, t], x_down_aligned, m_down_aligned)

    def test(self, epoch, device):
        pass

    def loss_function(self, x_down, m_down, x_down_aligned, m_down_aligned):
        x_down_extended = x_down.unsqueeze(2).repeat(1, 1, x_down_aligned.size(2), 1, 1)
        return self.utils_losses.masked_l1(x_down_extended, x_down_aligned, 1 - m_down_aligned, 'mean', 1)