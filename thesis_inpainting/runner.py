import thesis.runner
import models.vgg_16
import models.thesis_alignment
import models.thesis_inpainting
import torch
import utils.losses
import thesis_alignment.runner
import torch.nn.functional as F
import utils.flow
import numpy as np
import utils.draws


class ThesisInpaintingRunner(thesis.runner.ThesisRunner):
    # checkpoint_path = '/home/ubuntu/ebs/master_thesis/experiments/hard_flow_9/checkpoints/80.checkpoint.pkl'
    checkpoint_path = '/Users/DavidAlvarezDLT/Documents/PyCharm/master_thesis/experiments/test/checkpoints/58.checkpoint.pkl'
    model_vgg = None
    model_alignment = None
    utils_losses = None

    def init_model(self, device):
        torch.autograd.set_detect_anomaly(True)
        self.model_vgg = models.vgg_16.get_pretrained_model(device)
        self.model_alignment = models.thesis_alignment.ThesisAlignmentModel(self.model_vgg).to(device)
        self.model = models.thesis_inpainting.ThesisInpaintingModel(self.model_vgg).to(device)
        self.load_alignment_state(self.checkpoint_path, device)

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

    def load_alignment_state(self, checkpoint_path, device):
        with open(checkpoint_path, 'rb') as checkpoint_file:
            self.model_alignment.load_state_dict(torch.load(checkpoint_file, map_location=device)['model'])

    def train_step(self, it_data, device):
        (x, m), y, info = it_data
        x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)

        # Compute t and r_list
        t, r_list = x.size(2) // 2, list(range(x.size(2)))
        r_list.pop(t)

        # Propagate through alignment network
        with torch.no_grad():
            corr, xs, vs, ys, xs_aligned, _, vs_aligned, _, _, _, _, v_maps, _ = \
                thesis_alignment.runner.ThesisAlignmentRunner.train_step_propagate(
                    self.model_alignment, x, m, y, flow_gt, flows_use, t, r_list
                )

        # Propagate through inpainting network
        ys_hat, ys_hat_comp = ThesisInpaintingRunner.train_step_propagate(
            self.model, xs, vs, ys, xs_aligned, vs_aligned, v_maps, t, r_list
        )

        # Get both total loss and loss items
        loss, loss_items = ThesisInpaintingRunner.compute_loss(self.utils_losses, ys_hat, ys_hat_comp, ys, t, r_list)

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        for i, loss_item in enumerate(self.losses_items_ids):
            e_losses_items[loss_item].append(loss_items[i].item())

        # Return total loss
        return loss

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
        x_64_tbx, m_64_tbx, y_64_tbx, x_64_aligned_tbx, y_hat_64_tbx, y_hat_comp_64_tbx = [], [], [], [], [], []
        x_256_tbx, m_256_tbx, y_256_tbx, x_256_aligned_tbx, y_hat_256_tbx, y_hat_comp_256_tbx = [], [], [], [], [], []
        v_map_64_tbx, v_map_256_tbx = [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, info = it_data
            x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)

            # Propagate through the model
            t, r_list = x.size(2) // 2, list(range(x.size(2)))
            r_list.pop(t)
            with torch.no_grad():
                corr, xs, vs, ys, xs_aligned, xs_aligned_gt, vs_aligned, vs_aligned_gt, flows, flows_gt, flows_use, \
                v_maps, v_maps_gt = thesis_alignment.runner.ThesisAlignmentRunner.train_step_propagate(
                    self.model_alignment, x, m, y, flow_gt, flows_use, t, r_list
                )
                ys_hat, ys_hat_comp = ThesisInpaintingRunner.train_step_propagate(
                    self.model, xs, vs, ys, xs_aligned, vs_aligned, v_maps, t, r_list
                )

            # Add items to the lists
            x_64_tbx.append(xs[1].cpu().numpy())
            m_64_tbx.append(1 - vs[1].cpu().numpy())
            y_64_tbx.append(ys[1].cpu().numpy())
            x_64_aligned_tbx.append(xs_aligned[1].cpu().numpy())
            y_hat_64_tbx.append(ys_hat[1].cpu().numpy())
            y_hat_comp_64_tbx.append(ys_hat_comp[1].cpu().numpy())
            x_256_tbx.append(xs[2].cpu().numpy())
            m_256_tbx.append(1 - vs[2].cpu().numpy())
            y_256_tbx.append(ys[2].cpu().numpy())
            x_256_aligned_tbx.append(xs_aligned[2].cpu().numpy())
            v_map_64_tbx.append((v_maps[0] > 0.5).float().cpu().numpy())
            v_map_256_tbx.append((v_maps[1] > 0.5).float().cpu().numpy())
            y_hat_256_tbx.append(ys_hat[2].cpu().numpy())
            y_hat_comp_256_tbx.append(ys_hat_comp[2].cpu().numpy())

        # Concatenate the results along dim=0
        x_64_tbx = np.concatenate(x_64_tbx)
        m_64_tbx = np.concatenate(m_64_tbx)
        y_64_tbx = np.concatenate(y_64_tbx)
        x_64_aligned_tbx = np.concatenate(x_64_aligned_tbx)
        v_map_64_tbx = np.concatenate(v_map_64_tbx)
        y_hat_64_tbx = np.concatenate(y_hat_64_tbx)
        y_hat_comp_64_tbx = np.concatenate(y_hat_comp_64_tbx)
        x_256_tbx = np.concatenate(x_256_tbx)
        m_256_tbx = np.concatenate(m_256_tbx)
        y_256_tbx = np.concatenate(y_256_tbx)
        x_256_aligned_tbx = np.concatenate(x_256_aligned_tbx)
        v_map_256_tbx = np.concatenate(v_map_256_tbx)
        y_hat_256_tbx = np.concatenate(y_hat_256_tbx)
        y_hat_comp_256_tbx = np.concatenate(y_hat_comp_256_tbx)

        # Define a function to add images to TensorBoard
        def add_sample_tbx(x, m, y, x_aligned, v_map, y_hat, y_hat_comp, t, res_size):
            for b in range(x.shape[0]):
                x_aligned_sample = np.insert(arr=x_aligned[b], obj=t, values=x[b, :, t], axis=1)
                x_aligned_sample = utils.draws.add_border(x_aligned_sample, m[b, :, t])
                v_map_rep, m_rep = v_map[b].repeat(3, axis=0), m[b, :, t].repeat(3, axis=0)
                v_map_sample = np.insert(arr=v_map_rep, obj=t, values=m_rep, axis=1)
                y_sample = np.insert(arr=np.zeros_like(x_aligned[b]), obj=t, values=y_hat[b], axis=1)
                y_sample[:, t - 1] = y[b, :, t]
                y_sample[:, t + 1] = y_hat_comp[b]
                sample = np.concatenate(
                    (x[b], x_aligned_sample, v_map_sample, y_sample), axis=2
                ).transpose(1, 0, 2, 3)
                self.experiment.tbx.add_images(
                    '{}_sample_{}/{}'.format('validation', res_size, b + 1), sample, global_step=self.counters['epoch']
                )

        # Add different resolutions to TensorBoard
        add_sample_tbx(
            x_64_tbx, m_64_tbx, y_64_tbx, x_64_aligned_tbx, v_map_64_tbx, y_hat_64_tbx, y_hat_comp_64_tbx, t, '64'
        )
        add_sample_tbx(
            x_256_tbx, m_256_tbx, y_256_tbx, x_256_aligned_tbx, v_map_256_tbx, y_hat_256_tbx, y_hat_comp_256_tbx, t, '256'
        )

    @staticmethod
    def train_step_propagate(model, xs, vs, ys, xs_aligned, vs_aligned, v_maps, t, r_list):
        y_hat_16, y_hat_comp_16, y_hat_64, y_hat_comp_64, y_hat_256, y_hat_comp_256 = model(
            [xs_item[:, :, t] for xs_item in xs], [vs_item[:, :, t] for vs_item in vs],
            [ys_item[:, :, t] for ys_item in ys], xs_aligned, vs_aligned, v_maps
        )
        return (y_hat_16, y_hat_64, y_hat_256), (y_hat_comp_16, y_hat_comp_64, y_hat_comp_256)

    @staticmethod
    def compute_loss(utils_losses, ys_hat, ys_hat_comp, ys, t, r_list):
        recons_16 = F.l1_loss(ys_hat[0], ys[0][:, :, t])
        recons_64 = F.l1_loss(ys_hat[1], ys[1][:, :, t])
        recons_256 = F.l1_loss(ys_hat[2], ys[2][:, :, t])
        loss = recons_16 + recons_64 + recons_256
        return loss, [recons_16, recons_64, recons_256]
