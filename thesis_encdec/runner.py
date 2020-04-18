import skeltorch
from .model import CPNEncoderDecoder
import torch.optim
import torch.nn.functional as F
from models.model_vgg import get_pretrained_model
import torch.utils.data


class EncoderDecoderRunner(skeltorch.Runner):
    model_vgg = None
    e_train_losses_items = None
    e_validation_losses_items = None
    losses_it_items = None
    losses_epoch_items = None
    vgg_mean = None
    vgg_std = None
    loss_constant_normalization = None
    loss_weights = None
    scheduler = None

    def init_model(self, device):
        self.model = CPNEncoderDecoder().to(device)
        self.model_vgg = get_pretrained_model(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.experiment.configuration.get('training', 'lr'))

    def init_others(self, device):
        self.loss_weights = self.experiment.configuration.get('model', 'loss_lambdas')
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.experiment.configuration.get('training', 'lr_scheduler_step_size'),
            gamma=self.experiment.configuration.get('training', 'lr_scheduler_gamma')
        )

    def train_step(self, it_data, device):
        # Decompose iteration data
        (x, m), y, info = it_data

        # Move data to the correct device
        x = x.to(device)
        m = m.to(device)
        y = y.to(device)

        # Propagate through the model
        y_hat, y_hat_comp = self.model(x, m, y)

        # Compute loss and return
        loss, loss_items = self._compute_loss(y[:, :, 0], y_hat, y_hat_comp, x[:, :, 0], m[:, :, 0])

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        e_losses_items['h'].append(loss_items[0].item())
        e_losses_items['nh'].append(loss_items[1].item())
        e_losses_items['perceptual'].append(loss_items[2].item())
        e_losses_items['style'].append(loss_items[3].item())
        e_losses_items['tv'].append(loss_items[4].item())

        # Return combined loss
        return loss

    def test(self, epoch, device):
        pass

    def _compute_loss(self, y_t, y_hat, y_hat_comp, x_t, m):
        # Loss 1: Hole
        h_input = m * y_hat
        h_target = m * y_t
        if self.loss_constant_normalization:
            loss_h = F.l1_loss(h_input, h_target)
        else:
            loss_h = F.l1_loss(h_input, h_target, reduction='sum') / torch.sum(m)
        loss_h *= self.loss_weights[0]

        # Loss 2: Non-Hole
        nh_input = (1 - m) * y_hat
        nh_target = (1 - m) * y_t
        if self.loss_constant_normalization:
            loss_nh = F.l1_loss(nh_input, nh_target)
        else:
            loss_nh = F.l1_loss(nh_input, nh_target, reduction='sum') / torch.sum(1 - m)
        loss_nh *= self.loss_weights[1]

        # User VGG-16 to compute features of both the estimation and the target
        with torch.no_grad():
            vgg_y = self.model_vgg(y_t.contiguous())
            vgg_y_hat = self.model_vgg(y_hat.contiguous())

        # Loss 3: Perceptual
        loss_perceptual = 0
        for p in range(len(vgg_y)):
            loss_perceptual += F.l1_loss(vgg_y_hat[p], vgg_y[p])
        loss_perceptual /= len(vgg_y)
        loss_perceptual *= self.loss_weights[2]

        # Loss 4: Style
        loss_style = 0
        for p in range(len(vgg_y)):
            b, c, h, w = vgg_y[p].size()
            g_y = torch.mm(vgg_y[p].view(b * c, h * w), vgg_y[p].view(b * c, h * w).t())
            g_y_comp = torch.mm(vgg_y_hat[p].view(b * c, h * w), vgg_y_hat[p].view(b * c, h * w).t())
            loss_style += F.l1_loss(g_y_comp / (b * c * h * w), g_y / (b * c * h * w))
        loss_style /= len(vgg_y)
        loss_style *= self.loss_weights[3]

        # Loss 5: Smoothing Checkerboard Effect
        loss_tv_h = (y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]).pow(2).sum()
        loss_tv_w = (y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]).pow(2).sum()
        loss_tv = (loss_tv_h + loss_tv_w) / (y_hat.size(0) * y_hat.size(1) * y_hat.size(2) * y_hat.size(3))
        loss_tv *= self.loss_weights[4]

        # Compute combined loss
        loss = loss_h + loss_nh + loss_perceptual + loss_style + loss_tv

        # Return combination of the losses
        return loss, [loss_h, loss_nh, loss_perceptual, loss_style, loss_tv]
