import thesis.runner
import models.vgg_16
import models.thesis_alignment
import models.thesis_inpainting
import models.thesis_inpainting
import torch
import utils.losses
import thesis_alignment.runner
import torch.nn.functional as F
import utils.flow
import numpy as np
import utils.draws
import matplotlib.pyplot as plt


class ThesisInpaintingRunner(thesis.runner.ThesisRunner):
    checkpoint_path_cuda = '/home/ubuntu/ebs/master_thesis/experiments/align_v3_1/checkpoints/45.checkpoint.pkl'
    checkpoint_path_cpu = '/Users/DavidAlvarezDLT/Documents/PyCharm/master_thesis/experiments/test/checkpoints/45.checkpoint.pkl'
    model_vgg = None
    model_alignment = None
    utils_losses = None

    def init_model(self, device):
        self.checkpoint_path = self.checkpoint_path_cuda if device == 'cuda' else self.checkpoint_path_cpu
        self.model_vgg = models.vgg_16.get_pretrained_model(device)
        self.model_alignment = models.thesis_alignment.ThesisAlignmentModel(self.model_vgg).to(device)
        self.model = models.thesis_inpainting.ThesisInpaintingVisible().to(device)
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
        self.losses_items_ids = ['loss_nh', 'loss_vh', 'loss_nvh']
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
            x_aligned, v_aligned = thesis_alignment.runner.ThesisAlignmentRunner.infer_step_propagate(
                self.model_alignment, x, m, t, r_list
            )

        # Propagate through inpainting network
        y_hat, y_hat_comp, v_map = self.model(x[:, :, t], (1 - m)[:, :, t], y[:, :, t], x_aligned, v_aligned)

        # Get both total loss and loss items
        loss, loss_items = ThesisInpaintingRunner.compute_loss(
            self.utils_losses, y[:, :, t], (1 - m)[:, :, t], y_hat, y_hat_comp, v_map
        )

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
        x_tbx, m_tbx, y_tbx, x_aligned_tbx, y_hat_tbx, y_hat_comp_tbx, v_map_tbx = [], [], [], [], [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, info = it_data
            x, m, y, flows_use, flow_gt = x.to(device), m.to(device), y.to(device), info[2], info[5].to(device)

            # Propagate through the model
            t, r_list = x.size(2) // 2, list(range(x.size(2)))
            r_list.pop(t)
            with torch.no_grad():
                x_aligned, v_aligned = thesis_alignment.runner.ThesisAlignmentRunner.infer_step_propagate(
                    self.model_alignment, x, m, t, r_list
                )
                y_hat, y_hat_comp, v_map = self.model(x[:, :, t], (1 - m)[:, :, t], y[:, :, t], x_aligned, v_aligned)

            # Add items to the lists
            x_tbx.append(x.cpu().numpy())
            m_tbx.append(m.cpu().numpy())
            y_tbx.append(y.cpu().numpy())
            x_aligned_tbx.append(x_aligned.cpu().numpy())
            y_hat_tbx.append(y_hat.cpu().numpy())
            y_hat_comp_tbx.append(y_hat_comp.cpu().numpy())
            v_map_tbx.append(v_map.cpu().numpy())

        # Concatenate the results along dim=0
        x_tbx = np.concatenate(x_tbx)
        m_tbx = np.concatenate(m_tbx)
        y_tbx = np.concatenate(y_tbx)
        x_aligned_tbx = np.concatenate(x_aligned_tbx)
        y_hat_tbx = np.concatenate(y_hat_tbx)
        y_hat_comp_tbx = np.concatenate(y_hat_comp_tbx)
        v_map_tbx = np.concatenate(v_map_tbx)

        # Define a function to add images to TensorBoard
        def add_sample_tbx(x, m, y, x_aligned, v_map, y_hat, y_hat_comp, t, res_size):
            for b in range(x.shape[0]):
                x_aligned_sample = np.insert(arr=x_aligned[b], obj=t, values=x[b, :, t], axis=1)
                x_aligned_sample = utils.draws.add_border(x_aligned_sample, m[b, :, t])
                v_map_rep, m_rep = v_map[b].repeat(3, axis=0), m[b, :, t].repeat(3, axis=0)
                v_map_sample = np.insert(arr=v_map_rep, obj=t, values=m_rep, axis=1)
                y_hat_sample = np.insert(arr=y_hat[b], obj=t, values=y[b, :, t], axis=1)
                y_hat_sample = utils.draws.add_border(y_hat_sample, m[b, :, t])
                y_hat_comp_sample = np.insert(arr=y_hat_comp[b], obj=t, values=y[b, :, t], axis=1)
                # y_sample = np.insert(arr=np.zeros_like(x_aligned[b]), obj=t, values=y_hat[b], axis=1)
                # y_sample[:, t - 1] = y[b, :, t]
                # y_sample[:, t + 1] = y_hat_comp[b]
                sample = np.concatenate(
                    (x[b], x_aligned_sample, v_map_sample, y_hat_sample, y_hat_comp_sample), axis=2
                ).transpose(1, 0, 2, 3)
                self.experiment.tbx.add_images(
                    '{}_sample_{}/{}'.format('validation', res_size, b + 1), sample, global_step=self.counters['epoch']
                )

        # Add different resolutions to TensorBoard
        add_sample_tbx(x_tbx, m_tbx, y_tbx, x_aligned_tbx, v_map_tbx, y_hat_tbx, y_hat_comp_tbx, t, '256')

    @staticmethod
    def compute_loss(utils_losses, y_target, v_target, y_hat, y_hat_comp, v_map):
        target_img = y_target.unsqueeze(2).repeat(1, 1, y_hat.size(2), 1, 1)
        nh_mask = v_target.unsqueeze(2).repeat(1, 1, y_hat.size(2), 1, 1)
        vh_mask = v_map
        nvh_mask = (1 - nh_mask) - vh_mask
        loss_nh = utils_losses.masked_l1(y_hat, target_img, nh_mask, reduction='sum')
        loss_vh = utils_losses.masked_l1(y_hat, target_img, vh_mask, reduction='sum')
        loss_nvh = utils_losses.masked_l1(y_hat, target_img, nvh_mask, reduction='sum', weight=0.5)
        loss = loss_nh + loss_vh + loss_nvh
        return loss, [loss_nh, loss_vh, loss_nvh]
