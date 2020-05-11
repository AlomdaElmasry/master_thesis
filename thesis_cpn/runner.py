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


class ThesisCPNRunner(thesis.runner.ThesisRunner):
    scheduler = None
    utils_losses = None
    utils_measures = None
    losses_items_ids = None

    def init_model(self, device):
        trained_aligner = self.experiment.configuration.get('model', 'trained_aligner')
        utils_alignment = utils.alignment.AlignmentUtils(trained_aligner, device) if trained_aligner is not None else \
            None
        self.model = models.cpn.CPNet(self.experiment.configuration.get('model', 'mode'), utils_alignment).to(device)
        self.utils_losses = utils.losses.LossesUtils(device)
        self.utils_losses.init_vgg(device)
        self.utils_measures = utils.measures.UtilsMeasures()

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
        self._init_mode_params()
        super().init_others(device)

    def _init_mode_params(self):
        if self.experiment.configuration.get('model', 'mode') == 'full':
            self.losses_items_ids = ['alignment', 'vh', 'nvh', 'nh', 'perceptual', 'style', 'tv', 'grad']
            self.loss_function = self._compute_loss_full
        elif self.experiment.configuration.get('model', 'mode') == 'aligner':
            self.losses_items_ids = ['alignment']
            self.loss_function = self._compute_loss_aligner
        else:
            self.losses_items_ids = ['vh', 'nvh', 'nh', 'perceptual', 'style', 'tv']
            self.loss_function = self._compute_loss_encdec

    def train_step(self, it_data, device):
        # Decompose iteration data
        (x, m), y, info = it_data

        # Move Tensors to correct device
        x = x.to(device)
        m = m.to(device)
        y = y.to(device)

        # Propagate through the model
        (y_hat, y_hat_comp, c_mask, _, (x_aligned, v_aligned)), t, r_list = self.train_step_propagate(
            x, m, y
        )

        # Get visibility map of aligned frames and target frame
        visibility_maps = (1 - m[:, :, t].unsqueeze(2)) * v_aligned

        # Compute loss and return
        loss, loss_items = self.loss_function(
            y[:, :, t], y_hat, y_hat_comp, x[:, :, t], x_aligned, visibility_maps, m[:, :, t], c_mask
        )

        # Append loss items to epoch dictionary
        e_losses_items = self.e_train_losses_items if self.model.training else self.e_validation_losses_items
        for i, loss_item in enumerate(self.losses_items_ids):
            e_losses_items[loss_item].append(loss_items[i].item())

        # Return combined loss
        return loss

    def train_step_propagate(self, x, m, y):
        # Set target and auxilliary frames indexes
        t = x.size(2) // 2
        r_list = list(range(x.size(2)))
        r_list.pop(t)

        # Return result from the model
        return self.model(x, m, y, t, r_list), t, r_list

    def test(self, epoch, device):
        # Load state if epoch is set
        if epoch is not None:
            self.load_states(epoch, device)

        # Set model in evaluation mode
        self.model.eval()

        # Compute objective quality measures
        if self.experiment.configuration.get('model', 'mode') in ['full', 'encdec']:
            self._test_objective_measures(
                self.experiment.data.datasets['validation'], self.experiment.data.validation_objective_measures_indexes,
                device
            )

        # Inpaint individual frames given by self.experiment.data.test_frames_indexes
        self._test_frames(
            self.experiment.data.datasets['validation'], self.experiment.data.validation_frames_indexes, 'validation',
            device
        )
        self._test_frames(
            self.experiment.data.datasets['test'], self.experiment.data.test_frames_indexes, 'test', device
        )

        # Inpaint entire sequences given by self.experiment.data.test_sequences_indexes
        if self.experiment.configuration.get('model', 'mode') in ['full'] and \
                self.experiment.configuration.get('model', 'trained_aligner') != 'glu-net':
            self._test_sequences(device)

    def _test_objective_measures(self, samples_dataset, samples_indexes, device):
        subset_dataset = torch.utils.data.Subset(samples_dataset, samples_indexes)
        loader = torch.utils.data.DataLoader(
            subset_dataset, self.experiment.configuration.get('training', 'batch_size')
        )

        # Initialize LPSIS model to proper device
        self.utils_measures.init_lpips(device)

        # Create variables to store PSNR, SSIM and LPIPS
        psnr, ssim, lpips = [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, _ = it_data
            x = x.to(device)
            m = m.to(device)
            y = y.to(device)
            with torch.no_grad():
                (_, y_hat_comp, _, _, (_, _)), t, _ = self.train_step_propagate(x, m, y)
                psnr += self.utils_measures.psnr(y[:, :, t].cpu(), y_hat_comp.cpu())
                ssim += self.utils_measures.ssim(y[:, :, t].cpu(), y_hat_comp.cpu())
                lpips += self.utils_measures.lpips(y[:, :, t], y_hat_comp)

        # Remove LPSIS model from memory
        self.utils_measures.destroy_lpips()

        # Log measures in TensorBoard
        self.experiment.tbx.add_scalar('test_measures/psrn', np.mean(psnr), global_step=self.counters['epoch'])
        self.experiment.tbx.add_scalar('test_measures/ssim', np.mean(ssim), global_step=self.counters['epoch'])
        self.experiment.tbx.add_scalar('test_measures/lpips', np.mean(lpips), global_step=self.counters['epoch'])

    def _test_frames(self, samples_dataset, samples_indexes, label, device):
        # Set training parameters to the original test dataset
        samples_dataset.frames_n = self.experiment.configuration.get('data', 'frames_n')

        # Create a Subset using self.experiment.data.test_frames_indexes defined frames
        subset_dataset = torch.utils.data.Subset(samples_dataset, samples_indexes)
        loader = torch.utils.data.DataLoader(
            subset_dataset, self.experiment.configuration.get('training', 'batch_size')
        )

        # Create variables with the images to log inside TensorBoard -> (b,c,h,w)
        x_tbx, y_tbx, y_hat_tbx, y_hat_comp_tbx, x_aligned_tbx, x_aligned_match, x_aligned_spacing = [], [], [], [], \
                                                                                                     [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, info = it_data
            x = x.to(device)
            m = m.to(device)
            y = y.to(device)
            with torch.no_grad():
                (y_hat, y_hat_comp, _, ref_importance, (x_aligned, _)), t, _ = self.train_step_propagate(x, m, y)
            x_tbx.append(x.cpu().numpy())
            if self.experiment.configuration.get('model', 'mode') in ['full', 'encdec']:
                y_tbx.append(y.cpu().numpy())
                y_hat_tbx.append(y_hat.cpu().numpy())
                y_hat_comp_tbx.append(y_hat_comp.cpu().numpy())
            if self.experiment.configuration.get('model', 'mode') in ['full', 'aligner']:
                x_aligned_tbx.append(x_aligned.cpu().numpy())
            if self.experiment.configuration.get('model', 'mode') in ['full']:
                x_aligned_match.append(ref_importance.cpu().numpy())
                x_aligned_spacing.extend(list(info[1]))

        # Concatenate the results along dim=0
        x_tbx = np.concatenate(x_tbx) if len(x_tbx) > 0 else x_tbx
        y_tbx = np.concatenate(y_tbx) if len(y_tbx) > 0 else y_tbx
        y_hat_tbx = np.concatenate(y_hat_tbx) if len(y_hat_tbx) > 0 else y_hat_tbx
        y_hat_comp_tbx = np.concatenate(y_hat_comp_tbx) if len(y_hat_comp_tbx) > 0 else y_hat_comp_tbx
        x_aligned_match = np.concatenate(x_aligned_match) if len(x_aligned_match) > 0 else x_aligned_match
        x_aligned_tbx = np.concatenate(x_aligned_tbx) if len(x_aligned_tbx) > 0 else y_tbx

        # Save group image for each sample
        for b in range(len(self.experiment.data.test_frames_indexes)):
            if self.experiment.configuration.get('model', 'mode') in ['full', 'encdec']:
                test_frames = [x_tbx[b, :, t], y_hat_tbx[b], y_hat_comp_tbx[b], y_tbx[b, :, t]]
                self.experiment.tbx.add_images(
                    '{}_frames/{}'.format(label, b + 1), test_frames, global_step=self.counters['epoch'],
                    dataformats='CHW'
                )
            if self.experiment.configuration.get('model', 'mode') in ['full', 'aligner']:
                test_alignment_x = x_tbx[b].transpose(1, 0, 2, 3)
                test_alignment_x_aligned = np.insert(
                    x_aligned_tbx[b].transpose(1, 0, 2, 3), x_tbx[b].shape[1] // 2, x_tbx[b, :, t], 0
                )
                text_alignment_x_aligned_match = x_aligned_match[b].transpose(1, 0, 2, 3)
                test_alignment_spacing = utils.draws.text_to_image(
                    ['t = {}'.format(spa) for spa in x_aligned_spacing[b].split(',')], test_alignment_x.shape[3]
                )
                test_alignment = np.concatenate(
                    (test_alignment_x, test_alignment_x_aligned, text_alignment_x_aligned_match, test_alignment_spacing
                     ), axis=2
                )
                self.experiment.tbx.add_images(
                    '{}_alignment/{}'.format(label, b), test_alignment, global_step=self.counters['epoch']
                )

    def _test_sequences(self, device):
        # Set training parameters to the original test dataset
        self.experiment.data.datasets['test'].frames_n = -1

        # Iterate over the set of sequences
        for sequence_index in self.experiment.data.test_sequences_indexes:
            (x, m), y, info = self.experiment.data.datasets['test'][sequence_index]
            c, f, h, w = x.size()

            # Unsqueeze batch dimension
            x = x.unsqueeze(0)
            m = m.unsqueeze(0)
            y = y.unsqueeze(0)

            # Create a matrix to store inpainted frames. Size (B, 2, C, F, H, W), where the 2 is due to double direction
            y = y.to(device)
            y_hat_comp = np.zeros((2, c, f, h, w), dtype=np.float32)

            # Use the model twice: forward (0) and backward (1)
            for d in range(2):
                x_copy = x.clone().to(device)
                m_copy = m.clone().to(device)

                # Iterate over all the frames of the video
                for t in (list(range(f)) if d == 0 else reversed(list(range(f)))):
                    r_list = ThesisCPNRunner.get_reference_frame_indexes(t, f)

                    # Replace input_frames and input_masks with previous predictions to improve quality
                    with torch.no_grad():
                        _, x_copy[:, :, t], _, _, _ = self.model(x_copy, m_copy, y, t, r_list)
                        m_copy[:, :, t] = 0
                        y_hat_comp[d, :, t] = x_copy[:, :, t].squeeze(0).detach().cpu().numpy()

            # Combine forward and backward predictions
            forward_factor = np.arange(start=0, stop=y_hat_comp.shape[2]) / f
            backward_factor = (f - np.arange(start=0, stop=y_hat_comp.shape[2])) / f
            y_hat_comp = (
                    y_hat_comp[0].transpose(0, 2, 3, 1) * forward_factor +
                    y_hat_comp[1].transpose(0, 2, 3, 1) * backward_factor
            ).transpose(0, 3, 1, 2)

            # Log 4 middle frames inside TensorBoard
            y_hat_comp_tbx = [
                y_hat_comp[:, f] for f in range(y_hat_comp.shape[1] // 2 - 2, y_hat_comp.shape[1] // 2 + 2)
            ]
            self.experiment.tbx.add_images(
                'test_sequences/{}'.format(info[0]), y_hat_comp_tbx, global_step=self.counters['epoch'],
                dataformats='CHW'
            )

            # Save the sequence in disk
            self._save_sample((y_hat_comp * 255).astype(np.uint8), 'test', info[0], True)

    def _compute_loss_full(self, y_t, y_hat, y_hat_comp, x_t, x_aligned, v_map, m, c_mask):
        reduction = 'mean' if self.experiment.configuration.get('model', 'loss_constant_normalization') else 'sum'
        loss_weights = self.experiment.configuration.get('model', 'loss_lambdas')
        x_extended = x_t.unsqueeze(2).repeat(1, 1, x_aligned.size(2), 1, 1)
        loss_alignment = self.utils_losses.masked_l1(x_extended, x_aligned, v_map, 'sum', loss_weights[0])
        loss_vh = self.utils_losses.masked_l1(y_t, y_hat, m * (1 - c_mask), reduction, loss_weights[1])
        loss_nvh = self.utils_losses.masked_l1(y_t, y_hat, m * c_mask, reduction, loss_weights[2])
        loss_nh = self.utils_losses.masked_l1(y_t, y_hat, 1 - m, reduction, loss_weights[3])
        loss_perceptual, vgg_y, vgg_y_hat_comp = self.utils_losses.perceptual(y_t, y_hat_comp, loss_weights[4])
        loss_style = self.utils_losses.style(vgg_y, vgg_y_hat_comp, loss_weights[5])
        loss_tv = self.utils_losses.tv(y_hat, loss_weights[6])
        loss_grad = self.utils_losses.grad(y_t, y_hat, 1, reduction, loss_weights[7])
        loss = loss_vh + loss_nvh + loss_nh + loss_perceptual + loss_style + loss_tv + loss_grad
        return loss, [loss_alignment, loss_vh, loss_nvh, loss_nh, loss_perceptual, loss_style, loss_tv, loss_grad]

    def _compute_loss_aligner(self, y_t, y_hat, y_hat_comp, x_t, x_aligned, v_map, m, c_mask):
        x_extended = x_t.unsqueeze(2).repeat(1, 1, x_aligned.size(2), 1, 1)
        loss_alignment = self.utils_losses.masked_l1(x_extended, x_aligned, v_map, 'sum')
        return loss_alignment, [loss_alignment]

    def _compute_loss_encdec(self, y_t, y_hat, y_hat_comp, x_t, x_aligned, v_map, m, c_mask):
        reduction = 'mean' if self.experiment.configuration.get('model', 'loss_constant_normalization') else 'sum'
        loss_weights = self.experiment.configuration.get('model', 'loss_lambdas')
        loss_vh = self.utils_losses.masked_l1(y_t, y_hat, m * (1 - c_mask), reduction, loss_weights[0])
        loss_nvh = self.utils_losses.masked_l1(y_t, y_hat, m * c_mask, reduction, loss_weights[1])
        loss_nh = self.utils_losses.masked_l1(y_t, y_hat, 1 - m, reduction, loss_weights[2])
        loss_perceptual, vgg_y, vgg_y_hat_comp = self.utils_losses.perceptual(y_t, y_hat_comp, loss_weights[3])
        loss_style = self.utils_losses.style(vgg_y, vgg_y_hat_comp, loss_weights[4])
        loss_tv = self.utils_losses.tv(y_hat, loss_weights[5])
        loss = loss_vh + loss_nvh + loss_nh + loss_perceptual + loss_style + loss_tv
        return loss, [loss_vh, loss_nvh, loss_nh, loss_perceptual, loss_style, loss_tv]

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
