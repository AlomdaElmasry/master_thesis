import skeltorch
import models.align_masks
import torch


class ThesisRunner(skeltorch.Runner):
    aligner = None

    def init_model(self, device):
        self.model = models.align_masks.AlignMasksModel()

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
        y_hat, y_hat_comp, c_mask, (x_aligned, v_aligned) = self.model(x, m, y, t, r_list)

    def test(self, epoch, device):
        pass
