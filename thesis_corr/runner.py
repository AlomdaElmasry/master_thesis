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
import models.cpn_corr


class ThesisCorrelationRunner(skeltorch.Runner):

    def init_model(self, device):
        self.model = models.cpn_corr.CPNetMatching().to(device)

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
        pass

    def loss_function(self, y_t, y_hat, y_hat_comp, m_t):
        reduction = 'mean'
        loss_v = self.utils_losses.masked_l1(y_t, y_hat_comp, 1 - m_t, reduction, 1)
        loss_nv = self.utils_losses.masked_l1(y_t, y_hat_comp, m_t, reduction, 1)
        loss = loss_v + loss_nv
        return loss, [loss_v, loss_nv]
