import torch.optim
import models.alignment_corr
import torch.nn.functional as F
import utils.losses
import numpy as np
import utils.draws
import utils.flow
import thesis.runner
import models.vgg_16
import matplotlib.pyplot as plt
import utils.correlation


class ThesisAlignmentRunner(thesis.runner.ThesisRunner):
    model_vgg = None
    utils_losses = None
    losses_items_ids = None

    def init_model(self, device):
        self.model_vgg = models.vgg_16.get_pretrained_model(device)
        self.model = models.alignment_corr.AlignmentCorrelation(self.model_vgg).to(device)

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
        self.losses_items_ids = ['corr_loss', 'flow_16', 'flow_64', 'flow_256', 'alignment_recons_16',
                                 'alignment_recons_64', 'alignment_recons_256', 'v_map_loss_64', 'v_map_loss_256']
        super().init_others(device)

    def train_step(self, it_data, device):
        (x, m), y, info = it_data
        x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)

        # Compute t and r_list
        t, r_list = x.size(2) // 2, list(range(x.size(2)))
        r_list.pop(t)

        # Propagate through the model
        corr, xs, vs, ys, xs_aligned, xs_aligned_gt, vs_aligned, vs_aligned_gt, flows, flows_gt, flows_use, v_maps, \
        v_maps_gt = self.train_step_propagate(x, m, y, flow_gt, flows_use, t, r_list)

        # Get both total loss and loss items
        loss, loss_items = self.compute_loss(
            corr, xs, vs, ys, xs_aligned, xs_aligned_gt, vs_aligned, vs_aligned_gt, flows, flows_gt, flows_use, v_maps,
            v_maps_gt, t, r_list
        )

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        for i, loss_item in enumerate(self.losses_items_ids):
            e_losses_items[loss_item].append(loss_items[i].item())

        # Return total loss
        return loss

    def train_step_propagate(self, x, m, y, flow_gt, flows_use, t, r_list):
        corr, flow_16, flow_64, flow_256, v_map_64, v_map_256 = self.model(x, m, t, r_list)

        # Resize the data to multiple resolutions
        (x_16, v_16, y_16), (x_64, v_64, y_64), (x_256, v_256, y_256) = \
            self.resize_data(x, 1 - m, y, 16), self.resize_data(x, 1 - m, y, 64), (x, 1 - m, y)
        flow_16_gt, flow_64_gt, flow_256_gt = utils.flow.resize_flow(flow_gt[:, r_list], (16, 16)), \
                                              utils.flow.resize_flow(flow_gt[:, r_list], (64, 64)), flow_gt[:, r_list]

        # Align the data in multiple resolutions with GT dense flow
        x_16_aligned_gt, v_16_aligned_gt = self.align_data(x_16[:, :, r_list], v_16[:, :, r_list], flow_16_gt)
        x_64_aligned_gt, v_64_aligned_gt = self.align_data(x_64[:, :, r_list], v_64[:, :, r_list], flow_64_gt)
        x_256_aligned_gt, v_256_aligned_gt = self.align_data(x_256[:, :, r_list], v_256[:, :, r_list], flow_256_gt)

        # Compute target v_maps
        v_map_64_gt, v_map_256_gt = torch.zeros_like(v_64_aligned_gt), torch.zeros_like(v_256_aligned_gt)
        v_map_64_gt[flows_use] = (
                v_64_aligned_gt[flows_use] - v_64[flows_use, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1)
        ).clamp(0, 1)
        v_map_256_gt[flows_use] = (
                v_256_aligned_gt[flows_use] - v_256[flows_use, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1)
        ).clamp(0, 1)

        # Align the data in multiple resolutions
        x_16_aligned, v_16_aligned = self.align_data(x_16[:, :, r_list], v_16[:, :, r_list], flow_16)
        x_64_aligned, v_64_aligned = self.align_data(x_64[:, :, r_list], v_64[:, :, r_list], flow_64)
        x_256_aligned, v_256_aligned = self.align_data(x_256[:, :, r_list], v_256[:, :, r_list], flow_256)

        # Pack variables to return
        xs, vs, ys = (x_16, x_64, x_256), (v_16, v_64, v_256), (y_16, y_64, y_256)
        xs_aligned = (x_16_aligned, x_64_aligned, x_256_aligned)
        xs_aligned_gt = (x_16_aligned_gt, x_64_aligned_gt, x_256_aligned_gt)
        vs_aligned = (v_16_aligned, v_64_aligned, v_256_aligned)
        vs_aligned_gt = (v_16_aligned_gt, v_64_aligned_gt, v_256_aligned_gt)
        flows, flows_gt = (flow_16, flow_64, flow_256), (flow_16_gt, flow_64_gt, flow_256_gt)
        v_maps, v_maps_gt = (v_map_64, v_map_256), (v_map_64_gt, v_map_256_gt)

        # Return packed data
        return corr, xs, vs, ys, xs_aligned, xs_aligned_gt, vs_aligned, vs_aligned_gt, flows, flows_gt, flows_use, \
            v_maps, v_maps_gt

    def compute_loss(self, corr, xs, vs, ys, xs_aligned, xs_aligned_gt, vs_aligned, vs_aligned_gt, flows, flows_gt,
                     flows_use, v_maps, v_maps_gt, t, r_list):

        # Get the features of the frames from VGG
        b, c, f, h, w = ys[2].size()
        with torch.no_grad():
            x_vgg_feats = self.model_vgg(ys[2].transpose(1, 2).reshape(b * f, c, h, w), normalize_input=True)
        y_vgg_feats = x_vgg_feats[3].reshape(b, f, -1, 16, 16).transpose(1, 2)
        corr_y = utils.correlation.compute_masked_correlation(y_vgg_feats, torch.ones_like(y_vgg_feats), t, r_list)

        # Compute L1 loss between correlation volumes
        corr_loss = F.l1_loss(corr, corr_y)

        # Compute flow losses
        flow_loss_16 = self.utils_losses.masked_l1(flows[0], flows_gt[0], torch.ones_like(flows[0]), flows_use)
        flow_loss_64 = self.utils_losses.masked_l1(flows[1], flows_gt[1], torch.ones_like(flows[1]), flows_use)
        flow_loss_256 = self.utils_losses.masked_l1(flows[2], flows_gt[2], torch.ones_like(flows[2]), flows_use)

        # Compute out-of-frame regions from flows
        mask_out_16 = ((flows[0] < -1).float() + (flows[0] > 1).float()).sum(4).clamp(0, 1).unsqueeze(1)
        mask_out_64 = ((flows[1] < -1).float() + (flows[1] > 1).float()).sum(4).clamp(0, 1).unsqueeze(1)
        mask_out_256 = ((flows[2] < -1).float() + (flows[2] > 1).float()).sum(4).clamp(0, 1).unsqueeze(1)

        # Compute alignment reconstruction losses
        alignment_recons_16 = self.utils_losses.masked_l1(
            xs[0][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), xs_aligned[0],
            vs[0][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1) * (1 - mask_out_16)
        )
        alignment_recons_64 = self.utils_losses.masked_l1(
            xs[1][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), xs_aligned[1],
            vs[1][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1) * (1 - mask_out_64)
        )
        alignment_recons_256 = self.utils_losses.masked_l1(
            xs[2][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1), xs_aligned[2],
            vs[2][:, :, t].unsqueeze(2).repeat(1, 1, len(r_list), 1, 1) * (1 - mask_out_256)
        )

        # Compute visual map loss
        v_map_loss_64 = self.utils_losses.bce(v_maps[0], v_maps_gt[0], torch.ones_like(v_maps_gt[0]), flows_use)
        v_map_loss_256 = self.utils_losses.bce(v_maps[1], v_maps_gt[1], torch.ones_like(v_maps_gt[1]), flows_use)

        # Compute sum of losses and return them
        total_loss = corr_loss + flow_loss_16 + flow_loss_64 + flow_loss_256
        total_loss += alignment_recons_16 + alignment_recons_64 + alignment_recons_256
        total_loss += v_map_loss_64 + v_map_loss_256
        return total_loss, [corr_loss, flow_loss_16, flow_loss_64, flow_loss_256, alignment_recons_16,
                            alignment_recons_64, alignment_recons_256, v_map_loss_64, v_map_loss_256]

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
        v_map_64_tbx, v_map_64_gt_tbx, v_map_256_tbx, v_map_256_gt_tbx = [], [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, info = it_data
            x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)

            # Propagate through the model
            t, r_list = x.size(2) // 2, list(range(x.size(2)))
            r_list.pop(t)
            with torch.no_grad():
                corr, xs, vs, ys, xs_aligned, xs_aligned_gt, vs_aligned, vs_aligned_gt, flows, flows_gt, flows_use, \
                v_maps, v_maps_gt = self.train_step_propagate(x, m, y, flow_gt, flows_use, t, r_list)

            # Get GT alignment
            x_64_aligned_gt, _ = self.align_data(xs[1][:, :, r_list], 1 - vs[1][:, :, r_list], flows_gt[1])
            x_256_aligned_gt, _ = self.align_data(xs[2][:, :, r_list], 1 - vs[2][:, :, r_list], flows_gt[2])

            # Add items to the lists
            x_64_tbx.append(xs[1].cpu().numpy())
            m_64_tbx.append(1 - vs[1].cpu().numpy())
            x_64_aligned_tbx.append(xs_aligned[1].cpu().numpy())
            x_64_aligned_gt_tbx.append(x_64_aligned_gt.cpu().numpy())
            x_256_tbx.append(xs[2].cpu().numpy())
            m_256_tbx.append(1 - vs[2].cpu().numpy())
            x_256_aligned_tbx.append(xs_aligned[2].cpu().numpy())
            x_256_aligned_gt_tbx.append(x_256_aligned_gt.cpu().numpy())
            v_map_64_tbx.append((v_maps[0] > 0.5).float().cpu().numpy())
            v_map_64_gt_tbx.append(v_maps_gt[0].cpu().numpy())
            v_map_256_tbx.append((v_maps[1] > 0.5).float().cpu().numpy())
            v_map_256_gt_tbx.append(v_maps_gt[1].cpu().numpy())

        # Concatenate the results along dim=0
        x_64_tbx = np.concatenate(x_64_tbx)
        m_64_tbx = np.concatenate(m_64_tbx)
        x_64_aligned_tbx = np.concatenate(x_64_aligned_tbx)
        x_64_aligned_gt_tbx = np.concatenate(x_64_aligned_gt_tbx)
        v_map_64_tbx = np.concatenate(v_map_64_tbx)
        v_map_64_gt_tbx = np.concatenate(v_map_64_gt_tbx)
        x_256_tbx = np.concatenate(x_256_tbx)
        m_256_tbx = np.concatenate(m_256_tbx)
        x_256_aligned_tbx = np.concatenate(x_256_aligned_tbx)
        x_256_aligned_gt_tbx = np.concatenate(x_256_aligned_gt_tbx)
        v_map_256_tbx = np.concatenate(v_map_256_tbx)
        v_map_256_gt_tbx = np.concatenate(v_map_256_gt_tbx)

        # Define a function to add images to TensorBoard
        def add_alignment_tbx(x, m, x_aligned, x_aligned_gt, res_size):
            for b in range(x.shape[0]):
                x_sample = x[b].transpose(1, 0, 2, 3)
                x_aligned_gt_sample = np.insert(arr=x_aligned_gt[b], obj=t, values=x[b, :, t], axis=1)
                x_aligned_gt_sample = utils.draws.add_border(x_aligned_gt_sample, m[b, :, t]).transpose(1, 0, 2, 3)
                x_aligned_sample = np.insert(arr=x_aligned[b], obj=t, values=x[b, :, t], axis=1)
                x_aligned_sample = utils.draws.add_border(x_aligned_sample, m[b, :, t]).transpose(1, 0, 2, 3)
                sample = np.concatenate((x_sample, x_aligned_gt_sample, x_aligned_sample), axis=2)
                self.experiment.tbx.add_images(
                    '{}_alignment_{}/{}'.format('validation', res_size, b + 1), sample,
                    global_step=self.counters['epoch']
                )

        # Define a function to add v_maps to TensorBoard
        def add_v_map_tbx(x, m, v_map, v_map_gt, res_size):
            for b in range(x.shape[0]):
                x_sample = x[b]
                v_map_rep, v_map_gt_rep, m_rep = v_map[b].repeat(3, axis=0), v_map_gt[b].repeat(3, axis=0), \
                                                 m[b, :, t].repeat(3, axis=0)
                v_map_sample = np.insert(arr=v_map_rep, obj=t, values=m_rep, axis=1)
                v_map_gt_sample = np.insert(arr=v_map_gt_rep, obj=t, values=m_rep, axis=1)
                sample = np.concatenate((x_sample, v_map_gt_sample, v_map_sample), axis=2).transpose(1, 0, 2, 3)
                self.experiment.tbx.add_images(
                    '{}_v_map_{}/{}'.format('validation', res_size, b + 1), sample,
                    global_step=self.counters['epoch']
                )

        # Add different resolutions to TensorBoard
        add_alignment_tbx(x_64_tbx, m_64_tbx, x_64_aligned_tbx, x_64_aligned_gt_tbx, '64')
        add_alignment_tbx(x_256_tbx, m_256_tbx, x_256_aligned_tbx, x_256_aligned_gt_tbx, '256')
        add_v_map_tbx(x_64_tbx, m_64_tbx, v_map_64_tbx, v_map_64_gt_tbx, '64')
        add_v_map_tbx(x_256_tbx, m_256_tbx, v_map_256_tbx, v_map_256_gt_tbx, '256')

    def resize_data(self, x, m, y, size):
        b, c, f, h, w = x.size()
        x_down = F.interpolate(x.transpose(1, 2).reshape(-1, c, h, w), (size, size), mode='bilinear'). \
            reshape(b, f, c, size, size).transpose(1, 2)
        m_down = F.interpolate(m.transpose(1, 2).reshape(-1, 1, h, w), (size, size)). \
            reshape(b, f, 1, size, size).transpose(1, 2)
        y_down = F.interpolate(y.transpose(1, 2).reshape(-1, c, h, w), (size, size), mode='bilinear'). \
            reshape(b, f, c, size, size).transpose(1, 2)
        return x_down, m_down, y_down

    def align_data(self, x, m, flow):
        b, c, f, h, w = x.size()
        x_aligned = F.grid_sample(
            x.transpose(1, 2).reshape(-1, c, h, w), flow.reshape(-1, h, w, 2), align_corners=True
        ).reshape(b, -1, 3, h, w).transpose(1, 2)
        m_aligned = F.grid_sample(
            m.transpose(1, 2).reshape(-1, 1, h, w), flow.reshape(-1, h, w, 2), align_corners=True, mode='nearest'
        ).reshape(b, -1, 1, h, w).transpose(1, 2)
        return x_aligned, m_aligned
