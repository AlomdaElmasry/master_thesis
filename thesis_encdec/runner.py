import skeltorch
from .model import CPNEncoderDecoder
import torch.optim
import numpy as np
from PIL import Image
import os.path
import utils
import torch.nn.functional as F
from thesis.model_vgg import get_pretrained_model
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

    def __init__(self):
        super().__init__()
        self.losses_it_items = {
            'train': {'h': {}, 'nh': {}, 'perceptual': {}, 'style': {}, 'tv': {}},
            'validation': {'h': {}, 'nh': {}, 'perceptual': {}, 'style': {}, 'tv': {}}
        }
        self.losses_epoch_items = {
            'train': {'h': {}, 'nh': {}, 'perceptual': {}, 'style': {}, 'tv': {}},
            'validation': {'h': {}, 'nh': {}, 'perceptual': {}, 'style': {}, 'tv': {}}
        }

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

    def train_before_epoch_tasks(self, device):
        super().train_before_epoch_tasks(device)
        self.e_train_losses_items = {'h': [], 'nh': [], 'perceptual': [], 'style': [], 'tv': []}
        self.e_validation_losses_items = {'h': [], 'nh': [], 'perceptual': [], 'style': [], 'tv': []}
        self.experiment.tbx.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.counters['epoch'])

    def train_iteration_log(self, e_train_losses, log_period, device):
        super().train_iteration_log(e_train_losses, log_period, device)
        self._log_iteration(self.counters['train_it'], self.losses_it_items['train'], self.e_train_losses_items,
                            log_period, 'iteration', 'train')

    def validation_iteration_log(self, e_validation_losses, log_period, device):
        super().validation_iteration_log(e_validation_losses, log_period, device)
        self._log_iteration(self.counters['validation_it'], self.losses_it_items['validation'],
                            self.e_validation_losses_items, log_period, 'iteration', 'validation')

    def train_after_epoch_tasks(self, device):
        super().train_after_epoch_tasks(device)
        self._log_iteration(self.counters['epoch'], self.losses_epoch_items['train'], self.e_train_losses_items, 0,
                            'epoch', 'train')
        self._log_iteration(self.counters['epoch'], self.losses_epoch_items['validation'],
                            self.e_validation_losses_items, 0, 'epoch', 'validation')
        self._generate_random_samples(device)
        self.experiment.data.load_loaders(None, self.experiment.data.loaders['train'].num_workers)
        self.scheduler.step(epoch=self.counters['epoch'])

    def _log_iteration(self, it_n, losses_it_items_split, e_losses_items, log_period, ot, split):
        losses_it_items_split['h'][it_n] = np.mean(e_losses_items['h'][-log_period:])
        losses_it_items_split['nh'][it_n] = np.mean(e_losses_items['nh'][-log_period:])
        losses_it_items_split['perceptual'][it_n] = np.mean(e_losses_items['perceptual'][-log_period:])
        losses_it_items_split['style'][it_n] = np.mean(e_losses_items['style'][-log_period:])
        losses_it_items_split['tv'][it_n] = np.mean(e_losses_items['tv'][-log_period:])
        self.experiment.tbx.add_scalar(
            'loss_hole/{}/{}'.format(ot, split), losses_it_items_split['h'][it_n], it_n
        )
        self.experiment.tbx.add_scalar(
            'loss_non_hole/{}/{}'.format(ot, split), losses_it_items_split['nh'][it_n], it_n
        )
        self.experiment.tbx.add_scalar(
            'loss_perceptual/{}/{}'.format(ot, split), losses_it_items_split['perceptual'][it_n], it_n
        )
        self.experiment.tbx.add_scalar(
            'loss_style/{}/{}'.format(ot, split), losses_it_items_split['style'][it_n], it_n
        )
        self.experiment.tbx.add_scalar(
            'loss_tv/{}/{}'.format(ot, split), losses_it_items_split['tv'][it_n], it_n
        )

    def test(self, epoch, device):
        raise NotImplemented

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

    def _generate_random_samples(self, device):
        b = self.experiment.configuration.get('training', 'batch_size')
        loader = torch.utils.data.DataLoader(self.experiment.data.datasets['train'], shuffle=True, batch_size=b)
        (x, m), y, _ = next(iter(loader))

        # Move data to the correct device
        x = x.to(device)
        m = m.to(device)
        y = y.to(device)

        # Get target and reference frames
        t = x.size(2) // 2

        # Propagate the data through the model
        with torch.no_grad():
            y_hat, y_hat_comp = self.model(x, m, y)

        # Expand 1->3 dimensions of the masks
        m = m.repeat(1, 3, 1, 1, 1)

        # Save group image for each sample
        for b in range(x.size(0)):
            tensor_list = [y[b, :, t], x[b, :, t], m[b, :, t], y_hat[b, :], y_hat_comp[b, :]]
            self.experiment.tbx.add_images(
                'samples/epoch-{}'.format(self.counters['epoch']), tensor_list, global_step=b + 1, dataformats='CHW'
            )

    def _save_samples(self, y_hat_comp, folder_name, file_name, save_as_video=True):
        """Saves a set of samples.

        Args:
            y_hat_comp (np.Array): array of size (B,F,H,W,C) containing the frames to save.
            folder_name (str): name of the folder where the results should be stored inside.
            file_name (list[str]): list containing the names of the sequences to be stored.
            save_as_video (bool): whether to store the sequence as a video.
        """
        for b in range(y_hat_comp.shape[0]):
            # Define path where the sequences will be stored
            b_path = os.path.join(
                self.experiment.paths['results'], folder_name, 'epoch-{}'.format(self.counters['epoch'])
            )
            if not os.path.exists(b_path):
                os.makedirs(b_path)

            # Save sequence as video
            if save_as_video:
                frames_to_video = utils.FramesToVideo(0, 10, None)
                frames_to_video.add_sequence(y_hat_comp[b])
                frames_to_video.save(b_path, file_name[b])

            # Save sequence as frames
            else:
                for t in range(y_hat_comp.shape[1]):
                    bt_path = os.path.join(b_path, file_name[b])
                    if not os.path.exists(bt_path):
                        os.makedirs(bt_path)
                    pil_img = Image.fromarray(y_hat_comp[b, t])
                    pil_img.save(os.path.join(bt_path, '{}.jpg'.format(t)))

            # Log progress
            self.logger.info('Epoch {} | Test generated as {} for sequence {}'.format(
                self.counters['epoch'], 'video' if save_as_video else 'frames', file_name[b]
            ))

    def load_states_others(self, checkpoint_data):
        self.scheduler.load_state_dict(checkpoint_data['scheduler'])
        self.losses_it_items = checkpoint_data['losses_it_items']
        self.losses_epoch_items = checkpoint_data['losses_epoch_items']

    def save_states_others(self):
        return {'scheduler': self.scheduler.state_dict(), 'losses_it_items': self.losses_it_items,
                'losses_epoch_items': self.losses_epoch_items}
