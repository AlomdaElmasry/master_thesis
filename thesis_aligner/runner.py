from thesis.runner import ThesisRunner
import torch
import torch.nn.functional as F
from .model import ThesisAligner


class AlignerRunner(ThesisRunner):
    losses_items_ids = ['alignment']

    def init_model(self, device):
        self.model = ThesisAligner()

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-5
        )

    def init_others(self, device):
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.experiment.configuration.get('training', 'lr_scheduler_step_size'),
            gamma=self.experiment.configuration.get('training', 'lr_scheduler_gamma')
        )

    def train_step(self, it_data, device):
        # Decompose iteration data
        (x, m), y, info = it_data

        # Move Tensors to correct device
        x = x.to(device)
        m = m.to(device)
        y = y.to(device)

        # Set target and auxilliary frames indexes
        t = x.size(2) // 2
        r_list = list(range(x.size(2)))
        r_list.pop(t)

        # Propagate through the model
        x_aligned, v_aligned, y_aligned = self.model(x, m, y, t, r_list)

        # Get visibility map of aligned frames and target frame
        visibility_maps = (1 - m[:, :, t].unsqueeze(2)) * v_aligned

        # Return the loss
        return self._compute_loss(x[:, :, t], x_aligned, visibility_maps)

    def _compute_loss(self, x_t, x_aligned, v_map):
        alignment_input = x_aligned * v_map
        alignment_target = x_t.unsqueeze(2).repeat(1, 1, x_aligned.size(2), 1, 1) * v_map
        return F.l1_loss(alignment_input, alignment_target)
