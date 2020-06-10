import torch.optim
import numpy as np
import thesis.runner
import models.cpn
import torch.utils.data
import utils.measures
import utils.losses
import utils.alignment
import utils.draws
import utils.losses
import models.corr
import matplotlib.pyplot as plt


class ThesisCorrelationRunner(thesis.runner.ThesisRunner):
    scheduler = None
    utils_losses = None
    utils_measures = None
    losses_items_ids = None

    def init_model(self, device):
        self.model = models.corr.CorrelationModel(device).to(device)

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
        self.utils_losses.init_vgg(device)
        self.losses_items_ids = ['h', 'nh', 'perceptual']
        super().init_others(device)

    def train_step(self, it_data, device):
        self.test(None, device)
        self.experiment.tbx.flush()
        exit()
        # Decompose iteration data
        (x, m), y, info = it_data

        # Move Tensors to correct device
        x = x.to(device)
        m = m.to(device)
        y = y.to(device)

        # Propagate through the model
        t = x.size(2) // 2
        r_list = list(range(x.size(2)))
        r_list.pop(t)
        y_hat, y_hat_comp, _ = self.model(x, m, y, t, r_list)

        # Compute loss and return
        loss, loss_items = self.loss_function(y[:, :, t], y_hat, y_hat_comp, m[:, :, t])

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        for i, loss_item in enumerate(self.losses_items_ids):
            e_losses_items[loss_item].append(loss_items[i].item())

        # Return the total loss
        return loss

    def test(self, epoch, device):
        # Load state if epoch is set
        if epoch is not None:
            self.load_states(epoch, device)

        # Set model in evaluation mode
        self.model.eval()

        # Test frames from the validation split
        self._test_frames(
            self.experiment.data.datasets['validation'], self.experiment.data.validation_frames_indexes, 'validation',
            device
        )

        # Test frames from the test split
        # self._test_frames(
        #     self.experiment.data.datasets['test'], self.experiment.data.test_frames_indexes, 'test', device
        # )

    def _test_frames(self, samples_dataset, samples_indexes, label, device):
        # Set training parameters to the original test dataset
        samples_dataset.frames_n = self.experiment.configuration.get('data', 'frames_n')

        # Create a Subset using self.experiment.data.test_frames_indexes defined frames
        subset_dataset = torch.utils.data.Subset(samples_dataset, samples_indexes)
        loader = torch.utils.data.DataLoader(
            subset_dataset, self.experiment.configuration.get('training', 'batch_size')
        )

        # Create variables with the images to log inside TensorBoard -> (b,c,h,w)
        x_tbx, y_tbx, y_hat_tbx, y_hat_comp_tbx, corr_tbx = [], [], [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, info = it_data
            x = x.to(device)
            m = m.to(device)
            y = y.to(device)
            with torch.no_grad():
                t = x.size(2) // 2
                r_list = list(range(x.size(2)))
                r_list.pop(t)
                y_hat, y_hat_comp, corr = self.model(x, m, y, t, r_list)
            x_tbx.append(x.cpu().numpy())
            y_tbx.append(y.cpu().numpy())
            y_hat_tbx.append(y_hat.cpu().numpy())
            y_hat_comp_tbx.append(y_hat_comp.cpu().numpy())
            corr_tbx.append(corr.cpu().numpy())
            break

        # Concatenate the results along dim=0
        x_tbx = np.concatenate(x_tbx) if len(x_tbx) > 0 else x_tbx
        y_tbx = np.concatenate(y_tbx) if len(y_tbx) > 0 else y_tbx
        y_hat_tbx = np.concatenate(y_hat_tbx) if len(y_hat_tbx) > 0 else y_hat_tbx
        y_hat_comp_tbx = np.concatenate(y_hat_comp_tbx) if len(y_hat_comp_tbx) > 0 else y_hat_comp_tbx
        corr_tbx = np.concatenate(corr_tbx) if len(corr_tbx) > 0 else corr_tbx

        # Save group image for each sample
        for b in range(x_tbx.shape[0]):
            test_frames = [x_tbx[b, :, t], y_hat_tbx[b], y_hat_comp_tbx[b], y_tbx[b, :, t]]
            self.experiment.tbx.add_images(
                '{}_frames/{}'.format(label, b + 1), test_frames, global_step=self.counters['epoch'], dataformats='CHW'
            )

        # Save correlation visualization for each sample
        h_index = w_index = 7
        cmap = plt.get_cmap('jet')
        for b in range(corr_tbx.shape[0]):
            corr_frames = []
            for t in range(corr_tbx.shape[1]):
                rgba_img = np.transpose(cmap(corr_tbx[b, t, h_index, w_index]), (2, 0, 1))
                corr_frames.append(rgba_img[:3, :, :])
            self.experiment.tbx.add_images(
                '{}_corr/{}'.format(label, b + 1), corr_frames, global_step=self.counters['epoch'], dataformats='CHW'
            )

        # Add the histogram of the correlations
        self.experiment.tbx.add_histogram(
            '{}_corr_bins'.format(label), corr_tbx, global_step=self.counters['epoch'] + 1
        )

    def loss_function(self, y_t, y_hat, y_hat_comp, m_t):
        reduction = 'mean'
        loss_weights = self.experiment.configuration.get('model', 'loss_lambdas')
        loss_h = self.utils_losses.masked_l1(y_t, y_hat, m_t, reduction, loss_weights[0])
        loss_nh = self.utils_losses.masked_l1(y_t, y_hat, 1 - m_t, reduction, loss_weights[1])
        loss_perceptual, vgg_y, vgg_y_hat_comp = self.utils_losses.perceptual(y_t, y_hat_comp, 1)
        loss = loss_h + loss_nh + loss_perceptual
        return loss, [loss_h, loss_nh, loss_perceptual]
