import skeltorch
import torch.optim
import models.alignment_corr
import torch.nn.functional as F
import utils.losses
import numpy as np
import matplotlib.pyplot as plt


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
        self.test(None, device)
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

        # Interpolate data to be 16x16 and align
        x_down, m_down = self.resize_to_16(x, m)
        x_down_aligned, m_down_aligned = self.align_16(corr_mixed, x_down[:, :, r_list], m_down[:, :, r_list])

        # Return the loss
        return self.loss_function(x_down[:, :, t], m_down[:, :, t], x_down_aligned, m_down_aligned)

    def train_after_epoch_tasks(self, device):
        super().train_after_epoch_tasks(device)
        self.test(None, device)

    def test(self, epoch, device):
        # Load state if epoch is set
        if epoch is not None:
            self.load_states(epoch, device)

        # Set model in evaluation mode
        self.model.eval()

        # Create a Subset using self.experiment.data.test_frames_indexes defined frames
        subset_dataset = torch.utils.data.Subset(
            self.experiment.data.datasets['validation'], self.experiment.data.validation_objective_measures_indexes
        )
        loader = torch.utils.data.DataLoader(
            subset_dataset, self.experiment.configuration.get('training', 'batch_size')
        )

        # Create variables with the images to log inside TensorBoard -> (b,c,h,w)
        x_down_tbx, x_down_aligned_tbx = [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, info = it_data
            x, m = x.to(device), m.to(device)

            # Propagate through the model
            t, r_list = x.size(2) // 2, list(range(x.size(2)))
            r_list.pop(t)
            with torch.no_grad():
                corr, corr_mixed = self.model(x, m, t, r_list)

            # Interpolate data to be 16x16 and align
            x_down, m_down = self.resize_to_16(x, m)
            x_down_aligned, m_down_aligned = self.align_16(corr_mixed, x_down[:, :, r_list], m_down[:, :, r_list])

            # Add items to the lists
            x_down_tbx.append(x_down.cpu().numpy())
            x_down_aligned_tbx.append(x_down_aligned.cpu().numpy())

        # Concatenate the results along dim=0
        x_down_tbx = np.concatenate(x_down_tbx)
        x_down_aligned_tbx = np.concatenate(x_down_aligned_tbx)

        # Add samples to TensorBoard
        for b in range(x_down_tbx.shape[0]):
            x_down_sample = x_down_tbx[b].transpose(1, 0, 2, 3)
            x_down_aligned_sample = np.insert(
                x_down_aligned_tbx[b].transpose(1, 0, 2, 3), x_down_tbx[b].shape[1] // 2, x_down_tbx[b, :, t], 0
            )
            sample = np.concatenate((x_down_sample, x_down_aligned_sample), axis=2)
            self.experiment.tbx.add_images(
                '{}_alignment_16/{}'.format('validation', b + 1), sample, global_step=self.counters['epoch']
            )

    def loss_function(self, x_down, m_down, x_down_aligned, m_down_aligned):
        x_down_extended = x_down.unsqueeze(2).repeat(1, 1, x_down_aligned.size(2), 1, 1)
        return self.utils_losses.masked_l1(x_down_extended, x_down_aligned, 1 - m_down_aligned, 'mean', 1)

    def resize_to_16(self, x, m):
        b, c, f, h, w = x.size()
        x_down = F.interpolate(x.transpose(1, 2).reshape(-1, c, h, w), (16, 16), mode='bilinear'). \
            reshape(b, f, c, 16, 16).transpose(1, 2)
        m_down = F.interpolate(m.transpose(1, 2).reshape(-1, 1, h, w), (16, 16)). \
            reshape(b, f, 1, 16, 16).transpose(1, 2)
        return x_down, m_down

    def align_16(self, corr_mixed, x_down, m_down):
        b, c, f, h, w = x_down.size()
        x_down_aligned = F.grid_sample(x_down.transpose(1, 2).reshape(-1, c, 16, 16), corr_mixed.reshape(-1, 16, 16, 2)
                                       ).reshape(b, -1, 3, 16, 16).transpose(1, 2)
        m_down_aligned = F.grid_sample(m_down.transpose(1, 2).reshape(-1, 1, 16, 16), corr_mixed.reshape(-1, 16, 16, 2)
                                       ).reshape(b, -1, 1, 16, 16).transpose(1, 2)
        return x_down_aligned, m_down_aligned
