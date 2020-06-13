import torch.optim
import models.alignment_corr
import torch.nn.functional as F
import utils.losses
import numpy as np
import utils.draws
import utils.flow
import thesis.runner
import matplotlib.pyplot as plt


class ThesisAlignmentRunner(thesis.runner.ThesisRunner):
    utils_losses = None
    losses_items_ids = None

    def init_model(self, device):
        self.model = models.alignment_corr.AlignmentCorrelation(device).to(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.experiment.configuration.get('training', 'lr')
        )

    def init_others(self, device):
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.experiment.configuration.get('training', 'lr_scheduler_step_size'),
            gamma=self.experiment.configuration.get('training', 'lr_scheduler_gamma')
        )
        self.utils_losses = utils.losses.LossesUtils(device)
        self.losses_items_ids = ['recons_16', 'recons_64', 'recons_256']
        super().init_others(device)

    def train_step(self, it_data, device):
        # Decompose iteration data and move data to proper device
        (x, m), y, info = it_data
        x, m, y = x.to(device), m.to(device), y.to(device)

        # Compute t and r_list
        t, r_list = x.size(2) // 2, list(range(x.size(2)))
        r_list.pop(t)

        # Propagate through the model
        corr, flows, xs, ms, xs_aligned, ms_aligned, v_maps = self.train_step_propagate(x, m, t, r_list)

        # Get both total loss and loss items
        loss, loss_items = self.compute_loss(xs, xs_aligned, v_maps, t, r_list)

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        for i, loss_item in enumerate(self.losses_items_ids):
            e_losses_items[loss_item].append(loss_items[i].item())

        # Return total loss
        return loss

    def train_step_propagate(self, x, m, t, r_list):
        corr, flow_16, flow_64, flow_256, _, _ = self.model(x, m, t, r_list)
        (x_16, m_16), (x_64, m_64), (x_256, m_256) = self.resize_data(x, m, 16), self.resize_data(x, m, 64), (x, m)
        x_16_aligned, m_16_aligned = self.align_data(x_16[:, :, r_list], m_16[:, :, r_list], flow_16)
        x_64_aligned, m_64_aligned = self.align_data(x_64[:, :, r_list], m_64[:, :, r_list], flow_64)
        x_256_aligned, m_256_aligned = self.align_data(x_256[:, :, r_list], m_256[:, :, r_list], flow_256)
        # v_map_16 = (1 - m_16[:, :, t].unsqueeze(2)) * (1 - m_16_aligned)
        # v_map_64 = (1 - m_64[:, :, t].unsqueeze(2)) * (1 - m_64_aligned)
        # v_map_256 = (1 - m_256[:, :, t].unsqueeze(2)) * (1 - m_256_aligned)
        v_map_16, v_map_64, v_map_256 = 1, 1, 1
        return corr, (flow_16, flow_64, flow_256), (x_16, x_64, x_256), (m_16, m_64, m_256), \
               (x_16_aligned, x_64_aligned, x_256_aligned), (m_16_aligned, m_64_aligned, m_256_aligned), \
               (v_map_16, v_map_64, v_map_256)

    def compute_loss(self, xs, xs_aligned, v_maps, t, r_list):
        loss_recons_16 = self.utils_losses.masked_l1(
            xs[0][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), xs_aligned[0], v_maps[0]
        )
        loss_recons_64 = self.utils_losses.masked_l1(
            xs[1][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), xs_aligned[1], v_maps[1]
        )
        loss_recons_256 = self.utils_losses.masked_l1(
            xs[2][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), xs_aligned[2], v_maps[2]
        )
        return loss_recons_16 + loss_recons_64 + loss_recons_256, [loss_recons_16, loss_recons_64, loss_recons_256]

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
        x_256_tbx, m_256_tbx, x_256_aligned_tbx = [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, info = it_data
            x, y, m = x.to(device), y.to(device), m.to(device)

            # Propagate through the model
            t, r_list = x.size(2) // 2, list(range(x.size(2)))
            r_list.pop(t)
            with torch.no_grad():
                corr, flows, xs, ms, xs_aligned, ms_aligned, v_maps = self.train_step_propagate(x, m, t, r_list)

            # Add items to the lists
            x_256_tbx.append(xs[2].cpu().numpy())
            m_256_tbx.append(ms[2].cpu().numpy())
            x_256_aligned_tbx.append(xs_aligned[2].cpu().numpy())

        # Concatenate the results along dim=0
        x_256_tbx = np.concatenate(x_256_tbx)
        m_256_tbx = np.concatenate(m_256_tbx)
        x_256_aligned_tbx = np.concatenate(x_256_aligned_tbx)

        # Add samples to TensorBoard
        for b in range(x_256_tbx.shape[0]):
            x_256_sample = x_256_tbx[b].transpose(1, 0, 2, 3)
            x_256_aligned_sample = np.insert(
                arr=x_256_aligned_tbx[b], obj=x_256_tbx[b].shape[1] // 2, values=x_256_tbx[b, :, t], axis=1
            )
            x_256_aligned_sample = utils.draws.add_border(x_256_aligned_sample, m_256_tbx[b, :, t]) \
                .transpose(1, 0, 2, 3)
            sample = np.concatenate((x_256_sample, x_256_aligned_sample), axis=2)
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
