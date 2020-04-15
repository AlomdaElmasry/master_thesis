import skeltorch
import numpy as np
import os
import utils
from PIL import Image


class ThesisRunner(skeltorch.Runner):
    losses_items_ids = None
    losses_it_items = None
    losses_epoch_items = None
    e_train_losses_items = None
    e_validation_losses_items = None
    scheduler = None

    def __init__(self):
        super().__init__()
        self.losses_it_items = {
            'train': {loss_item_id: {} for loss_item_id in self.losses_items_ids},
            'validation': {loss_item_id: {} for loss_item_id in self.losses_items_ids}
        }
        self.losses_epoch_items = {
            'train': {loss_item_id: {} for loss_item_id in self.losses_items_ids},
            'validation': {loss_item_id: {} for loss_item_id in self.losses_items_ids}
        }

    def init_model(self, device):
        raise NotImplemented

    def init_optimizer(self, device):
        raise NotImplemented

    def load_states_others(self, checkpoint_data):
        self.scheduler.load_state_dict(checkpoint_data['scheduler'])
        self.losses_it_items = checkpoint_data['losses_it_items']
        self.losses_epoch_items = checkpoint_data['losses_epoch_items']

    def save_states_others(self):
        return {'scheduler': self.scheduler.state_dict(), 'losses_it_items': self.losses_it_items,
                'losses_epoch_items': self.losses_epoch_items}

    def train_step(self, it_data, device):
        raise NotImplemented

    def train_before_epoch_tasks(self, device):
        super().train_before_epoch_tasks(device)
        self.e_train_losses_items = {loss_item_id: [] for loss_item_id in self.losses_items_ids}
        self.e_validation_losses_items = {loss_item_id: [] for loss_item_id in self.losses_items_ids}
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
        self.test(None, device)
        self.scheduler.step(epoch=self.counters['epoch'])
        self.experiment.data.load_loaders(None, self.experiment.data.loaders['train'].num_workers)

    def _log_iteration(self, it_n, losses_it_items_split, e_losses_items, log_period, ot, split):
        for loss_item_id in self.losses_items_ids:
            losses_it_items_split[loss_item_id][it_n] = np.mean(e_losses_items[loss_item_id][-log_period:])
            self.experiment.tbx.add_scalar(
                'loss_items_{}_{}/{}'.format(split, ot, loss_item_id), losses_it_items_split[loss_item_id][it_n], it_n
            )

    def test(self, epoch, device):
        raise NotImplemented

    def _generate_random_samples(self, device):
        raise NotImplemented

    def _save_sample(self, y_hat_comp, folder_name, file_name, save_as_video=True):
        """Saves a set of samples.

        Args:
            y_hat_comp (np.Array): array of size (F,H,W,C) containing the frames to save.
            folder_name (str): name of the folder where the results should be stored inside.
            file_name (list[str]): list containing the names of the sequences to be stored.
            save_as_video (bool): whether to store the sequence as a video.
        """
        path = os.path.join(self.experiment.paths['results'], folder_name, 'epoch-{}'.format(self.counters['epoch']))
        if not os.path.exists(path):
            os.makedirs(path)

        # Permute Tensor
        y_hat_comp = y_hat_comp.transpose(1, 2, 3, 0)

        # Save sequence as video
        if save_as_video:
            frames_to_video = utils.FramesToVideo(0, 10, None)
            frames_to_video.add_sequence(y_hat_comp)
            frames_to_video.save(path, file_name)

        # Save sequence as frames
        else:
            for t in range(y_hat_comp.shape[1]):
                bt_path = os.path.join(path, file_name)
                if not os.path.exists(bt_path):
                    os.makedirs(bt_path)
                pil_img = Image.fromarray(y_hat_comp[t])
                pil_img.save(os.path.join(bt_path, '{}.jpg'.format(t)))

        # Log progress
        self.logger.info('Epoch {} | Test generated as {} for sequence {}'.format(
            self.counters['epoch'], 'video' if save_as_video else 'frames', file_name
        ))
