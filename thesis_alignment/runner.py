import skeltorch
import torch.optim
import models.alignment_corr
import torch.nn.functional as F
import utils.losses
import numpy as np
import matplotlib.pyplot as plt
import utils.draws


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
        corr, flow_16, flow_64, flow_256, vmap_64, vmap_256 = self.model(x, m, t, r_list)

        # Compute losses over 16x16
        y_16, m_16 = self.resize_data(y, m, 16)
        y_16_aligned, m_16_aligned = self.align_data(y_16[:, :, r_list], m_16[:, :, r_list], flow_16)
        x_16_vmap = (1 - m_16[:, :, t].unsqueeze(2)) * (1 - m_16_aligned)
        loss_recons_16 = self.utils_losses.masked_l1(
            y_16[:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), y_16_aligned, x_16_vmap, 'mean', 10
        )

        # Compute losses over 64x64
        y_64, m_64 = self.resize_data(y, m, 64)
        y_64_aligned, m_64_aligned = self.align_data(y_64[:, :, r_list], m_64[:, :, r_list], flow_64)
        x_64_vmap = (1 - m_64[:, :, t].unsqueeze(2)) * (1 - m_64_aligned)
        loss_recons_64 = self.utils_losses.masked_l1(
            y_64[:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), y_64_aligned, x_64_vmap, 'mean', 10
        )

        # Compute losses over 256x256
        y_256, m_256 = y, m
        y_256_aligned, m_256_aligned = self.align_data(y_256[:, :, r_list], m_256[:, :, r_list], flow_256)
        x_256_vmap = (1 - m_256[:, :, t].unsqueeze(2)) * (1 - m_256_aligned)
        loss_recons_256 = self.utils_losses.masked_l1(
            y_256[:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), y_256_aligned, x_256_vmap, 'mean', 10
        )

        # Return the loss
        return loss_recons_16 + loss_recons_64 + loss_recons_256

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
            self.experiment.data.datasets['validation'], self.experiment.data.validation_frames_indexes
        )
        loader = torch.utils.data.DataLoader(
            subset_dataset, self.experiment.configuration.get('training', 'batch_size')
        )

        # Create variables with the images to log inside TensorBoard -> (b,c,h,w)
        y_256_tbx, m_256_tbx, y_256_aligned_tbx = [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, info = it_data
            x, m = x.to(device), m.to(device)

            # Propagate through the model
            t, r_list = x.size(2) // 2, list(range(x.size(2)))
            r_list.pop(t)
            with torch.no_grad():
                corr, flow_16, flow_64, flow_256, vmap_64, vmap_256 = self.model(x, m, t, r_list)

            # Compute losses over 256x256
            y_256, m_256 = y, m
            y_256_aligned, m_256_aligned = self.align_data(y_256[:, :, r_list], m_256[:, :, r_list], flow_256)

            # Add items to the lists
            y_256_tbx.append(y_256.cpu().numpy())
            m_256_tbx.append(m_256.cpu().numpy())
            y_256_aligned_tbx.append(y_256_aligned.cpu().numpy())

        # Concatenate the results along dim=0
        y_256_tbx = np.concatenate(y_256_tbx)
        m_256_tbx = np.concatenate(m_256_tbx)
        y_256_aligned_tbx = np.concatenate(y_256_aligned_tbx)

        # Add samples to TensorBoard
        for b in range(y_256_tbx.shape[0]):
            y_256_sample = y_256_tbx[b].transpose(1, 0, 2, 3)
            y_256_aligned_sample = np.insert(
                arr=y_256_aligned_tbx[b], obj=y_256_tbx[b].shape[1] // 2, values=y_256_tbx[b, :, t], axis=1
            )
            y_256_aligned_sample = utils.draws.add_border(y_256_aligned_sample, m_256_tbx[b, :, t]) \
                .transpose(1, 0, 2, 3)
            sample = np.concatenate((y_256_sample, y_256_aligned_sample), axis=2)
            self.experiment.tbx.add_images(
                '{}_alignment_256/{}'.format('validation', b + 1), sample, global_step=self.counters['epoch']
            )

    def resize_data(self, x, m, size):
        b, c, f, h, w = x.size()
        x_down = F.interpolate(x.transpose(1, 2).reshape(-1, c, h, w), (size, size), mode='bilinear'). \
            reshape(b, f, c, size, size).transpose(1, 2)
        m_down = F.interpolate(m.transpose(1, 2).reshape(-1, 1, h, w), (size, size)). \
            reshape(b, f, 1, size, size).transpose(1, 2)
        return x_down, m_down

    def align_data(self, x, m, flow):
        b, c, f, h, w = x.size()
        x_aligned = F.grid_sample(
            x.transpose(1, 2).reshape(-1, c, h, w), flow.reshape(-1, h, w, 2)
        ).reshape(b, -1, 3, h, w).transpose(1, 2)
        m_aligned = F.grid_sample(
            m.transpose(1, 2).reshape(-1, 1, h, w), flow.reshape(-1, h, w, 2), mode='nearest'
        ).reshape(b, -1, 1, h, w).transpose(1, 2)
        return x_aligned, m_aligned
