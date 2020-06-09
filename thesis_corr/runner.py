import torch.optim
import numpy as np
import thesis.runner
import models.cpn
import torch.utils.data
import copy
import utils.measures
import utils.losses
import utils.alignment
import utils.draws
import matplotlib.pyplot as plt
import skeltorch
import utils.losses
import models.corr


class ThesisCorrelationRunner(skeltorch.Runner):

    def init_model(self, device):
        self.model = models.corr.CPNetMatching(device).to(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.experiment.configuration.get('training', 'lr')
        )

    def init_others(self, device):
        self.utils_losses = utils.losses.LossesUtils(device)

    def train_step(self, it_data, device):
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
        y_hat, y_hat_comp, *_ = self.model(x, m, y, t, r_list)

        # Compute loss and return
        loss, loss_items = self.loss_function(y[:, :, t], y_hat, y_hat_comp, m[:, :, t])

        # Return the total loss
        return loss

    def test(self, epoch, device):
        # Load state if epoch is set
        if epoch is not None:
            self.load_states(epoch, device)

        # Set model in evaluation mode
        self.model.eval()

        # Inpaint individual frames given by self.experiment.data.test_frames_indexes
        self._test_frames(
            self.experiment.data.datasets['validation'], self.experiment.data.validation_frames_indexes, 'validation',
            device
        )
        self._test_frames(
            self.experiment.data.datasets['test'], self.experiment.data.test_frames_indexes, 'test', device
        )

    def _test_frames(self, samples_dataset, samples_indexes, label, device):
        # Set training parameters to the original test dataset
        samples_dataset.frames_n = self.experiment.configuration.get('data', 'frames_n')

        # Create a Subset using self.experiment.data.test_frames_indexes defined frames
        subset_dataset = torch.utils.data.Subset(samples_dataset, samples_indexes)
        loader = torch.utils.data.DataLoader(
            subset_dataset, self.experiment.configuration.get('training', 'batch_size')
        )

        # Create variables with the images to log inside TensorBoard -> (b,c,h,w)
        x_tbx, y_tbx, y_hat_tbx, y_hat_comp_tbx = [], [], [], []

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
                y_hat, y_hat_comp = self.model(x, m, y, t, r_list)
            x_tbx.append(x.cpu().numpy())
            y_tbx.append(y.cpu().numpy())
            y_hat_tbx.append(y_hat.cpu().numpy())
            y_hat_comp_tbx.append(y_hat_comp.cpu().numpy())

        # Concatenate the results along dim=0
        x_tbx = np.concatenate(x_tbx) if len(x_tbx) > 0 else x_tbx
        y_tbx = np.concatenate(y_tbx) if len(y_tbx) > 0 else y_tbx
        y_hat_tbx = np.concatenate(y_hat_tbx) if len(y_hat_tbx) > 0 else y_hat_tbx
        y_hat_comp_tbx = np.concatenate(y_hat_comp_tbx) if len(y_hat_comp_tbx) > 0 else y_hat_comp_tbx

        # Save group image for each sample
        for b in range(len(self.experiment.data.test_frames_indexes)):
            if self.experiment.configuration.get('model', 'mode') in ['full', 'encdec']:
                test_frames = [x_tbx[b, :, t], y_hat_tbx[b], y_hat_comp_tbx[b], y_tbx[b, :, t]]
                self.experiment.tbx.add_images(
                    '{}_frames/{}'.format(label, b + 1), test_frames, global_step=self.counters['epoch'],
                    dataformats='CHW'
                )

    def loss_function(self, y_t, y_hat, y_hat_comp, m_t):
        reduction = 'mean'
        loss_h = self.utils_losses.masked_l1(y_t, y_hat, m_t, reduction, 1)
        loss_nh = self.utils_losses.masked_l1(y_t, y_hat, 1 - m_t, reduction, 1)
        loss = loss_h + loss_nh
        return loss, [loss_h, loss_nh]
