import thesis.runner
import torch.optim
import torch
import torch.nn.functional as F
import models.unet_attention
import matplotlib.pyplot as plt
import skeltorch
import torch.utils.data
import numpy as np


class ThesisAttentionRunner(skeltorch.Runner):

    def init_model(self, device):
        self.model = models.unet_attention.UNet(4, n_classes=1).to(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)

    def init_others(self, device):
        self.losses_items_ids = ['alignment', 'vh', 'nvh', 'nh', 'perceptual', 'style', 'tv', 'grad']

    def train_step(self, it_data, device):
        # Decompose iteration data
        (x, m), y, info = it_data

        # Move the data to the device
        x = x.to(device)
        m = m.to(device)

        # Compute target segmentation
        attention_target = self.compute_objective_section(m, info[5], info[6])

        # Propagate both x and m through the model
        attention = self.model(x[:, :, 0], m[:, :, 0], x[:, :, 1], m[:, :, 1])

        plt.imshow(np.concatenate((x[0, :, 0].permute(1, 2, 0), x[0, :, 1].permute(1, 2, 0)), axis=1))
        plt.show()

        plt.imshow(np.concatenate((m[0, 0, 0], m[0, 0, 1]), axis=1))
        plt.show()

        plt.imshow(attention_target[0, 0])
        plt.show()
        # plt.imshow(np.concatenate((attention[0, 0].detach().numpy(), attention_target[0, 0].numpy()), axis=1))
        # plt.show()

        # Lala
        return F.binary_cross_entropy_with_logits(attention, attention_target)

    def compute_objective_section(self, m, gt_movement, m_movement):
        return (m[:, 0, 0] - m[:, 0, 1]).clamp(0, 1).unsqueeze(1)

    def train_after_epoch_tasks(self, device):
        super().train_after_epoch_tasks(device)
        self.test(None, device)

    def test(self, epoch, device):
        if epoch is not None:
            self.load_states(epoch, device)

        subset_dataset = torch.utils.data.Subset(
            self.experiment.data.datasets['train'], self.experiment.data.validation_frames_indexes
        )
        loader = torch.utils.data.DataLoader(
            subset_dataset, self.experiment.configuration.get('training', 'batch_size')
        )

        # Create variables with the images to log inside TensorBoard -> (b,c,h,w)
        x_tbx, m_tbx, attention_tbx, attention_target_tbx = [], [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, info = it_data
            x = x.to(device)
            m = m.to(device)
            attention_target = self.compute_objective_section(m, info[5], info[6])
            with torch.no_grad():
                attention = self.model(x[:, :, 0], m[:, :, 0], x[:, :, 1], m[:, :, 1])
                loss = F.binary_cross_entropy_with_logits(attention, attention_target)
                self.logger.info('Loss {}'.format(loss))
            x_tbx.append(x.cpu().numpy())
            m_tbx.append(m.repeat(1, 3, 1, 1, 1).cpu().numpy())
            attention_tbx.append(attention.repeat(1, 3, 1, 1).cpu().numpy())
            attention_target_tbx.append(attention_target.repeat(1, 3, 1, 1).cpu().numpy())

        # Concatenate the results along dim=0
        x_tbx = np.concatenate(x_tbx)
        m_tbx = np.concatenate(m_tbx)
        attention_tbx = np.concatenate(attention_tbx)
        attention_target_tbx = np.concatenate(attention_target_tbx)

        # Save samples
        for b in range(len(self.experiment.data.test_frames_indexes)):
            first_row = np.concatenate((x_tbx[b, :, 0], x_tbx[b, :, 1], attention_target_tbx[b]), axis=2)
            second_row = np.concatenate((m_tbx[b, :, 0], m_tbx[b, :, 1], attention_tbx[b]), axis=2)
            test_frames = np.concatenate((first_row, second_row), axis=1)
            self.experiment.tbx.add_images(
                'frames/{}'.format(b + 1), test_frames, global_step=self.counters['epoch'], dataformats='CHW'
            )
