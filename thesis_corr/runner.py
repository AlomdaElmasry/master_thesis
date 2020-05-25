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
import models.cpn_corr


class ThesisCorrelationRunner(skeltorch.Runner):

    def init_model(self, device):
        self.model = models.cpn_corr.CPNetMatching().to(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.experiment.configuration.get('training', 'lr')
        )

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
        y_hat, y_hat_comp = self.model(x, m, y, t, r_list)

        print('one step')

    def test(self, epoch, device):
        pass
