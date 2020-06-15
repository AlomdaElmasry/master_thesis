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
        # self.losses_items_ids = ['recons_16', 'recons_64', 'recons_256']
        self.losses_items_ids = ['flow_16', 'flow_64', 'flow_256']
        super().init_others(device)

    def train_step(self, it_data, device):
        # Decompose iteration data and move data to proper device
        (x, m), y, info = it_data
        x, m, y, gt_movement = x.to(device), m.to(device), y.to(device), info[5].to(device)

        # Compute t and r_list
        t, r_list = x.size(2) // 2, list(range(x.size(2)))
        r_list.pop(t)

        # Propagate through the model
        corr, flows, xs, ms, gt_movements, xs_aligned, ms_aligned, v_maps = self.train_step_propagate(
            x, m, gt_movement, t, r_list
        )

        # Get both total loss and loss items
        loss, loss_items = self.compute_loss(xs, xs_aligned, flows, gt_movements, v_maps, t, r_list)

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        for i, loss_item in enumerate(self.losses_items_ids):
            e_losses_items[loss_item].append(loss_items[i].item())

        # Return total loss
        return loss

    def train_step_propagate(self, x, m, gt_movement, t, r_list):
        corr, flow_16, flow_64, flow_256, _, _ = self.model(x, m, t, r_list)

        # Resize the data to multiple resolutions
        (x_16, m_16), (x_64, m_64), (x_256, m_256) = self.resize_data(x, m, 16), self.resize_data(x, m, 64), (x, m)

        # Align the data in multiple resolutions
        x_16_aligned, m_16_aligned = self.align_data(x_16[:, :, r_list], m_16[:, :, r_list], flow_16)
        x_64_aligned, m_64_aligned = self.align_data(x_64[:, :, r_list], m_64[:, :, r_list], flow_64)
        x_256_aligned, m_256_aligned = self.align_data(x_256[:, :, r_list], m_256[:, :, r_list], flow_256)

        # Pack variables to return
        flows = (flow_16, flow_64, flow_256)
        xs = (x_16, x_64, x_256)
        ms = (m_16, m_64, m_256)
        gt_movements = utils.flow.resize_flow(gt_movement, (16, 16)), utils.flow.resize_flow(
            gt_movement, (64, 64)), gt_movement
        xs_aligned = (x_16_aligned, x_64_aligned, x_256_aligned)
        ms_aligned = (m_16_aligned, m_64_aligned, m_256_aligned)
        v_maps = 1, 1, 1

        # Return packed data
        return corr, flows, xs, ms, gt_movements, xs_aligned, ms_aligned, v_maps

    def compute_loss(self, xs, xs_aligned, flows, gt_movements, v_maps, t, r_list):
        flow_loss_16 = self.utils_losses.masked_l1(flows[0], gt_movements[0][:, r_list], 1)
        flow_loss_64 = self.utils_losses.masked_l1(flows[1], gt_movements[1][:, r_list], 1)
        flow_loss_256 = self.utils_losses.masked_l1(flows[2], gt_movements[2][:, r_list], 1)
        loss_recons_16 = self.utils_losses.masked_l1(
            xs[0][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), xs_aligned[0], v_maps[0]
        )
        loss_recons_64 = self.utils_losses.masked_l1(
            xs[1][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), xs_aligned[1], v_maps[1]
        )
        loss_recons_256 = self.utils_losses.masked_l1(
            xs[2][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), xs_aligned[2], v_maps[2]
        )
        total_loss = flow_loss_16 + flow_loss_64 + flow_loss_256
        return total_loss, [flow_loss_16, flow_loss_64, flow_loss_256]

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
        x_64_tbx, m_64_tbx, x_64_aligned_tbx, x_64_aligned_gt_tbx = [], [], [], []
        x_256_tbx, m_256_tbx, x_256_aligned_tbx, x_256_aligned_gt_tbx = [], [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, info = it_data
            x, m, y, gt_movement = x.to(device), m.to(device), y.to(device), info[5].to(device)

            # Propagate through the model
            t, r_list = x.size(2) // 2, list(range(x.size(2)))
            r_list.pop(t)
            with torch.no_grad():
                corr, flows, xs, ms, gt_movements, xs_aligned, ms_aligned, v_maps = self.train_step_propagate(
                    x, m, gt_movement, t, r_list
                )

            # Get GT alignment
            x_64_aligned_gt, _ = self.align_data(xs[1], ms[1], gt_movements[1])
            x_256_aligned_gt, _ = self.align_data(xs[2], ms[2], gt_movements[2])

            # Add items to the lists
            x_64_tbx.append(xs[1].cpu().numpy())
            m_64_tbx.append(ms[1].cpu().numpy())
            x_64_aligned_tbx.append(xs_aligned[1].cpu().numpy())
            x_64_aligned_gt_tbx.append(x_64_aligned_gt.cpu().numpy())
            x_256_tbx.append(xs[2].cpu().numpy())
            m_256_tbx.append(ms[2].cpu().numpy())
            x_256_aligned_tbx.append(xs_aligned[2].cpu().numpy())
            x_256_aligned_gt_tbx.append(x_256_aligned_gt.cpu().numpy())

        # Concatenate the results along dim=0
        x_64_tbx = np.concatenate(x_64_tbx)
        m_64_tbx = np.concatenate(m_64_tbx)
        x_64_aligned_tbx = np.concatenate(x_64_aligned_tbx)
        x_64_aligned_gt_tbx = np.concatenate(x_64_aligned_gt_tbx)
        x_256_tbx = np.concatenate(x_256_tbx)
        m_256_tbx = np.concatenate(m_256_tbx)
        x_256_aligned_tbx = np.concatenate(x_256_aligned_tbx)
        x_256_aligned_gt_tbx = np.concatenate(x_256_aligned_gt_tbx)

        # Define a function to add images to TensorBoard
        def add_to_tbx(x, m, x_aligned, x_aligned_gt, res_size):
            for b in range(x_256_tbx.shape[0]):
                x_sample = x[b].transpose(1, 0, 2, 3)
                x_aligned_sample = np.insert(arr=x_aligned[b], obj=x[b].shape[1] // 2, values=x[b, :, t], axis=1)
                x_aligned_sample = utils.draws.add_border(x_aligned_sample, m[b, :, t]).transpose(1, 0, 2, 3)
                sample = np.concatenate((x_sample, x_aligned_gt[b].transpose(1, 0, 2, 3), x_aligned_sample), axis=2)
                self.experiment.tbx.add_images(
                    '{}_alignment_{}/{}'.format('validation', res_size, b + 1), sample,
                    global_step=self.counters['epoch']
                )

        # Add different resolutions to TensorBoard
        add_to_tbx(x_64_tbx, m_64_tbx, x_64_aligned_tbx, x_64_aligned_gt_tbx, '64')
        add_to_tbx(x_256_tbx, m_256_tbx, x_256_aligned_tbx, x_256_aligned_gt_tbx, '256')

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
            x.transpose(1, 2).reshape(-1, c, h, w), flow.reshape(-1, h, w, 2), align_corners=True
        ).reshape(b, -1, 3, h, w).transpose(1, 2)
        m_aligned = F.grid_sample(
            m.transpose(1, 2).reshape(-1, 1, h, w), flow.reshape(-1, h, w, 2), align_corners=True, mode='nearest'
        ).reshape(b, -1, 1, h, w).transpose(1, 2)
        return x_aligned, m_aligned
