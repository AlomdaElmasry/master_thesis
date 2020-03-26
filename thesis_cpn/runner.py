import skeltorch
from .model import CPNet
import torch.optim
import numpy as np
from PIL import Image
import os.path
import utils
import torch.nn.functional as F
from .model_vgg import get_pretrained_model
import torch.utils.data


class CopyPasteRunner(skeltorch.Runner):
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
            'train': {'alignment': {}, 'vh': {}, 'nvh': {}, 'nh': {}, 'perceptual': {}, 'style': {}},
            'validation': {'alignment': {}, 'vh': {}, 'nvh': {}, 'nh': {}, 'perceptual': {}, 'style': {}}
        }
        self.losses_epoch_items = {
            'train': {'alignment': {}, 'vh': {}, 'nvh': {}, 'nh': {}, 'perceptual': {}, 'style': {}},
            'validation': {'alignment': {}, 'vh': {}, 'nvh': {}, 'nh': {}, 'perceptual': {}, 'style': {}}
        }

    def init_model(self, device):
        use_aligner = self.experiment.configuration.get('model', 'use_aligner')
        use_aligner = use_aligner if use_aligner is not None else True
        self.model = CPNet(use_aligner=use_aligner).to(device)
        self.model_vgg = get_pretrained_model(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def init_others(self, device):
        self.loss_constant_normalization = self.experiment.configuration.get('model', 'loss_constant_normalization')
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

        # Get target and reference frames
        t = x.size(2) // 2
        r_list = list(range(x.size(2)))
        r_list.pop(t)

        # Propagate through the model
        y_hat, y_hat_comp, c_mask, (x_aligned, v_aligned) = self.model(x, m, y, t, r_list)

        # Get visibility map of aligned frames and target frame
        visibility_maps = (1 - m[:, :, t].unsqueeze(2)) * v_aligned

        # Compute loss and return
        loss, loss_items = self._compute_loss(
            y[:, :, t], y_hat, y_hat_comp, x[:, :, t], x_aligned, visibility_maps, m[:, :, t], c_mask
        )

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        e_losses_items['alignment'].append(loss_items[0].item())
        e_losses_items['vh'].append(loss_items[1].item())
        e_losses_items['nvh'].append(loss_items[2].item())
        e_losses_items['nh'].append(loss_items[3].item())
        e_losses_items['perceptual'].append(loss_items[4].item())
        e_losses_items['style'].append(loss_items[5].item())

        # Return combined loss
        return loss

    def train_before_epoch_tasks(self, device):
        super().train_before_epoch_tasks(device)
        self.e_train_losses_items = {'alignment': [], 'vh': [], 'nvh': [], 'nh': [], 'perceptual': [], 'style': []}
        self.e_validation_losses_items = {'alignment': [], 'vh': [], 'nvh': [], 'nh': [], 'perceptual': [], 'style': []}
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
        losses_it_items_split['alignment'][it_n] = np.mean(e_losses_items['alignment'][-log_period:])
        losses_it_items_split['vh'][it_n] = np.mean(e_losses_items['vh'][-log_period:])
        losses_it_items_split['nvh'][it_n] = np.mean(e_losses_items['nvh'][-log_period:])
        losses_it_items_split['nh'][it_n] = np.mean(e_losses_items['nh'][-log_period:])
        losses_it_items_split['perceptual'][it_n] = np.mean(e_losses_items['perceptual'][-log_period:])
        losses_it_items_split['style'][it_n] = np.mean(e_losses_items['style'][-log_period:])
        self.experiment.tbx.add_scalar(
            'loss_alignment/{}/{}'.format(ot, split), losses_it_items_split['alignment'][it_n], it_n
        )
        self.experiment.tbx.add_scalar(
            'loss_visible_hole/{}/{}'.format(ot, split), losses_it_items_split['vh'][it_n], it_n
        )
        self.experiment.tbx.add_scalar(
            'loss_non_visible_hole/{}/{}'.format(ot, split), losses_it_items_split['nvh'][it_n], it_n
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

    def test(self, epoch, save_as_video, device):
        self.load_states(epoch, device)
        self.model.eval()
        for it_data in self.experiment.data.loaders['test']:
            (x, m), y, info = it_data
            b, c, n, h, w = x.size()

            # Create a matrix to store inpainted frames. Size (B, 2, C, F, H, W), where the 2 is due to double direction
            y = y.to(device)
            y_hat_comp = np.zeros((b, 2, c, n, h, w), dtype=np.float32)

            # Use the model twice: forward (0) and backward (1)
            for d in range(2):
                x_copy = x.clone().to(device)
                m_copy = m.clone().to(device)

                # Iterate over all the frames of the video
                for t in (list(range(n)) if d == 0 else reversed(list(range(n)))):
                    r_list = CopyPasteRunner.get_reference_frame_indexes(t, n)

                    # Replace input_frames and input_masks with previous predictions to improve quality
                    with torch.no_grad():
                        _, x_copy[:, :, t], _, _ = self.model(x_copy, m_copy, y, t, r_list)
                        m_copy[:, :, t] = 0
                        y_hat_comp[:, d, :, t] = x_copy[:, :, t].detach().cpu().numpy()

            # Combine forward and backward predictions
            forward_factor = np.arange(start=0, stop=y_hat_comp.shape[3]) / n
            backward_factor = (n - np.arange(start=0, stop=y_hat_comp.shape[3])) / n
            y_hat_comp = (
                    y_hat_comp[:, 0].transpose(0, 1, 3, 4, 2) * forward_factor * 255 +
                    y_hat_comp[:, 1].transpose(0, 1, 3, 4, 2) * backward_factor * 255
            ).transpose(0, 4, 2, 3, 1).astype(np.uint8)

            # Save the samples of the batch
            self._save_samples(y_hat_comp, 'test', info[0], save_as_video)

    def test_alignment(self, epoch, save_as_video, device):
        self.load_states(epoch, device)
        self.model.eval()
        for it_data in self.experiment.data.loaders['test']:
            (x, m), y, info = it_data
            b, c, n, h, w = x.size()

            # Force t=0 and obtain predicted aligned frames
            t = 0
            r_list = CopyPasteRunner.get_reference_frame_indexes(t, n)
            _, _, y_aligned = self.model.align(x, m, y, t, r_list)

            # Create a numpy array of size (B,F,H,W,C)
            y_aligned = (y_aligned.detach().cpu().permute(0, 2, 3, 4, 1).numpy() * 255).astype(np.uint8)

            # Save the samples of the batch
            self._save_samples(y_aligned, 'test_alignment', info[0], save_as_video)

    def test_inpainting(self, epoch, save_as_video, device):
        self.load_states(epoch, device)
        self.model.eval()
        for it_data in self.experiment.data.loaders['test_inpainting']:
            (x, m), y, (info, affine_matrices) = it_data
            b, c, n, h, w = x.size()

            # Compute inverse transformations
            inverse_affine_matrices = [torch.inverse(affine_matrices[0, i]) for i in range(n)]
            inverse_theta_matrices = torch.stack([
                utils.MovementSimulator.affine2theta(iam, h, w) for iam in inverse_affine_matrices
            ])
            inverse_grid = F.affine_grid(inverse_theta_matrices, [n, c, h, w])

            # Compute the aligned version of everything
            x_aligned = F.grid_sample(x[0].transpose(0, 1), inverse_grid).transpose(0, 1).unsqueeze(0)
            m_aligned = F.grid_sample(m[0].transpose(0, 1), inverse_grid).transpose(0, 1).unsqueeze(0)

            # Create a matrix to store inpainted frames. Size (B, 2, C, F, H, W), where the 2 is due to double direction
            y = y.to(device)
            y_hat_comp = np.zeros((b, 2, c, n, h, w), dtype=np.float32)

            # Use the model twice: forward (0) and backward (1)
            for d in range(2):
                x_copy = x.clone().to(device)
                m_copy = m.clone().to(device)

                # Iterate over all the frames of the video
                for t in (list(range(n)) if d == 0 else reversed(list(range(n)))):
                    with torch.no_grad():
                        _, x_copy[:, :, t], _ = self.model.copy_and_paste(
                            x_copy[:, :, t], m_copy[:, :, t], y[:, :, t], x_aligned, m_aligned
                        )
                        m_copy[:, :, t] = 0
                        y_hat_comp[:, d, :, t] = x_copy[:, :, t].detach().cpu().numpy()
                        print(t)

            # Combine forward and backward predictions
            forward_factor = np.arange(start=0, stop=y_hat_comp.shape[3]) / len(index)
            backward_factor = (len(index) - np.arange(start=0, stop=y_hat_comp.shape[3])) / len(index)
            y_hat_comp = (
                    y_hat_comp[:, 0].transpose(0, 1, 3, 4, 2) * forward_factor * 255 +
                    y_hat_comp[:, 1].transpose(0, 1, 3, 4, 2) * backward_factor * 255
            ).transpose(0, 4, 2, 3, 1).astype(np.uint8)

            # Save the samples of the batch
            # self._save_samples(y_aligned, 'test_inpainting', info[0], save_as_video)

    def _compute_loss(self, y_t, y_hat, y_hat_comp, x_t, x_aligned, v_map, m, c_mask):

        # Loss 1: Alignment Loss
        if self.experiment.configuration.get('model', 'use_aligner'):
            alignment_input = x_aligned * v_map
            alignment_target = x_t.unsqueeze(2).repeat(1, 1, x_aligned.size(2), 1, 1) * v_map
            if self.loss_constant_normalization:
                loss_alignment = F.l1_loss(alignment_input, alignment_target)
            else:
                loss_alignment = F.l1_loss(alignment_input, alignment_target, reduction='sum') / torch.sum(v_map)
            loss_alignment *= self.loss_weights[0]
        else:
            loss_alignment = torch.zeros(1).to(y_hat.device)

        # Loss 2: Visible Hole
        vh_input = m * (1 - c_mask) * y_hat
        vh_target = m * (1 - c_mask) * y_t
        if self.loss_constant_normalization:
            loss_vh = F.l1_loss(vh_input, vh_target)
        else:
            loss_vh = F.l1_loss(vh_input, vh_target, reduction='sum') / torch.sum(m * (1 - c_mask))
        loss_vh *= self.loss_weights[1]

        # Loss 3: Non-Visible Hole
        nvh_input = m * c_mask * y_hat
        nvh_target = m * c_mask * y_t
        if self.loss_constant_normalization:
            loss_nvh = F.l1_loss(nvh_input, nvh_target)
        else:
            loss_nvh = F.l1_loss(nvh_input, nvh_target, reduction='sum') / torch.sum(m * c_mask)
        loss_nvh *= self.loss_weights[2]

        # Loss 4: Non-Hole
        nh_input = (1 - m) * y_hat
        nh_target = (1 - m) * y_t
        if self.loss_constant_normalization:
            loss_nh = F.l1_loss(nh_input, nh_target)
        else:
            loss_nh = F.l1_loss(nh_input, nh_target, reduction='sum') / torch.sum(1 - m)
        loss_nh *= self.loss_weights[3]

        # User VGG-16 to compute features of both the estimation and the target
        with torch.no_grad():
            _, vgg_y = self.model_vgg(y_t.contiguous())
            _, vgg_y_hat_comp = self.model_vgg(y_hat_comp.contiguous())

        # Loss 5: Perceptual
        loss_perceptual = 0
        for p in range(len(vgg_y)):
            loss_perceptual += F.l1_loss(vgg_y_hat_comp[p], vgg_y[p])
        loss_perceptual /= len(vgg_y)
        loss_perceptual *= self.loss_weights[4]

        # Loss 6: Style
        loss_style = 0
        for p in range(len(vgg_y)):
            b, c, h, w = vgg_y[p].size()
            g_y = torch.mm(vgg_y[p].view(b * c, h * w), vgg_y[p].view(b * c, h * w).t())
            g_y_comp = torch.mm(vgg_y_hat_comp[p].view(b * c, h * w), vgg_y_hat_comp[p].view(b * c, h * w).t())
            loss_style += F.l1_loss(g_y_comp / (b * c * h * w), g_y / (b * c * h * w))
        loss_style /= len(vgg_y)
        loss_style *= self.loss_weights[5]

        # Loss 7: Smoothing Checkerboard Effect
        loss_smoothing = 0
        loss_smoothing *= self.loss_weights[6]

        # Compute combined loss
        loss = loss_alignment + loss_vh + loss_nvh + loss_nh + loss_perceptual + loss_style + loss_smoothing

        # Return combination of the losses
        return loss, [loss_alignment, loss_vh, loss_nvh, loss_nh, loss_perceptual, loss_style]

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
        r_list = list(range(x.size(2)))
        r_list.pop(t)

        # Propagate the data through the model
        with torch.no_grad():
            y_hat, y_hat_comp, _, _ = self.model(x, m, y, t, r_list)

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

    @staticmethod
    def get_reference_frame_indexes(t, n_frames, p=2, r_list_max_length=120):
        # Set start and end frames
        start = t - r_list_max_length
        end = t + r_list_max_length

        # Adjust frames in case they are not in the limit
        if t - r_list_max_length < 0:
            end = (t + r_list_max_length) - (t - r_list_max_length)
            end = n_frames - 1 if end > n_frames else end
            start = 0
        elif t + r_list_max_length > n_frames:
            start = (t - r_list_max_length) - (t + r_list_max_length - n_frames)
            start = 0 if start < 0 else start
            end = n_frames - 1

        # Return list of reference_frames every n=2 frames
        return [i for i in range(start, end, p) if i != t]
