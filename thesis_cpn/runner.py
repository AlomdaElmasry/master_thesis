from .model import CPNet
import torch.optim
import numpy as np
import torch.nn.functional as F
import thesis.runner
from thesis.model_vgg import get_pretrained_model
from thesis.model_lpips import PerceptualLoss
import torch.utils.data
import copy
import utils.measures


class CopyPasteRunner(thesis.runner.ThesisRunner):
    model_vgg = None
    model_lpips = None
    losses_items_ids = ['alignment', 'vh', 'nvh', 'nh', 'perceptual', 'style', 'tv']
    scheduler = None

    def init_model(self, device):
        self.model = CPNet(use_aligner=self.experiment.configuration.get('model', 'use_aligner')).to(device)
        self.model_vgg = get_pretrained_model(device)

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

    def train_step(self, it_data, device):
        # Decompose iteration data
        (x, m), y, info = it_data

        # Move Tensors to correct device
        x = x.to(device)
        m = m.to(device)
        y = y.to(device)

        # Propagate through the model
        (y_hat, y_hat_comp, c_mask, (x_aligned, v_aligned)), t, r_list = self.train_step_propagate(x, m, y)

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
        e_losses_items['tv'].append(loss_items[6].item())

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
        self._test_objective_measures(device)

        # Inpaint individual frames given by self.experiment.data.test_frames_indexes
        self._test_frames(device)

        # Inpaint entire sequences given by self.experiment.data.test_sequences_indexes
        self._test_sequences(device)

    def _test_objective_measures(self, device):
        # Set training parameters to the original test dataset
        samples_dataset = copy.deepcopy(self.experiment.data.datasets['test'])
        samples_dataset.frames_n = self.experiment.configuration.get('data', 'frames_n')

        # Create a Subset using self.experiment.data.test_objective_measures_indexes defined frames
        subset_dataset = torch.utils.data.Subset(samples_dataset, self.experiment.data.test_objective_measures_indexes)
        loader = torch.utils.data.DataLoader(subset_dataset, batch_size=4)

        # Initialize LPSIS model to proper device
        self.model_lpips = PerceptualLoss(model='net-lin', net='alex', use_gpu='cuda' in device)

        # Create variables to store PSNR, SSIM and LPIPS
        psnr, ssim, lpips = [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, _ = it_data
            x = x.to(device)
            m = m.to(device)
            y = y.to(device)
            with torch.no_grad():
                (_, y_hat_comp, _, (_, _)), t, _ = self.train_step_propagate(x, m, y)
                psnr += utils.measures.psnr(y[:, :, t].cpu(), y_hat_comp.cpu())
                ssim += utils.measures.ssim(y[:, :, t].cpu(), y_hat_comp.cpu())
                lpips += utils.measures.lpips(y[:, :, t], y_hat_comp, self.model_lpips)

        # Remove LPSIS model from memory
        self.model_lpips = None

        # Log measures in TensorBoard
        self.experiment.tbx.add_scalar('test_measures/psrn', np.mean(psnr), global_step=self.counters['epoch'])
        self.experiment.tbx.add_scalar('test_measures/ssim', np.mean(ssim), global_step=self.counters['epoch'])
        self.experiment.tbx.add_scalar('test_measures/lpips', np.mean(lpips), global_step=self.counters['epoch'])

    def _test_frames(self, device):
        # Set training parameters to the original test dataset
        samples_dataset = copy.deepcopy(self.experiment.data.datasets['test'])
        samples_dataset.frames_n = self.experiment.configuration.get('data', 'frames_n')

        # Create a Subset using self.experiment.data.test_frames_indexes defined frames
        subset_dataset = torch.utils.data.Subset(samples_dataset, indices=self.experiment.data.test_frames_indexes)
        loader = torch.utils.data.DataLoader(subset_dataset, batch_size=4)

        # Create variables with the images to log inside TensorBoard -> (b,c,h,w)
        x_tbx, y_hat_tbx, y_hat_comp_tbx, y_tbx = [], [], [], []

        # Iterate over the samples
        for it_data in loader:
            (x, m), y, _ = it_data
            x = x.to(device)
            m = m.to(device)
            y = y.to(device)
            with torch.no_grad():
                (y_hat, y_hat_comp, _, (_, _)), t, _ = self.train_step_propagate(x, m, y)
            x_tbx.append(x[:, :, t].cpu().numpy())
            y_hat_tbx.append(y_hat.cpu().numpy())
            y_hat_comp_tbx.append(y_hat_comp.cpu().numpy())
            y_tbx.append(y[:, :, t].cpu().numpy())

        # Concatenate the results along dim=0
        x_tbx = np.concatenate(x_tbx)
        y_hat_tbx = np.concatenate(y_hat_tbx)
        y_hat_comp_tbx = np.concatenate(y_hat_comp_tbx)
        y_tbx = np.concatenate(y_tbx)

        # Save group image for each sample
        for b in range(len(self.experiment.data.test_sequences_indexes)):
            tensor_list = [x_tbx[b], y_hat_tbx[b], y_hat_comp_tbx[b], y_tbx[b]]
            self.experiment.tbx.add_images(
                'test_frames/{}'.format(b), tensor_list, global_step=self.counters['epoch'], dataformats='CHW'
            )

    def _test_sequences(self, device):
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
                    r_list = CopyPasteRunner.get_reference_frame_indexes(t, f)

                    # Replace input_frames and input_masks with previous predictions to improve quality
                    with torch.no_grad():
                        _, x_copy[:, :, t], _, _ = self.model(x_copy, m_copy, y, t, r_list)
                        m_copy[:, :, t] = 0
                        y_hat_comp[d, :, t] = x_copy[:, :, t].squeeze(0).detach().cpu().numpy()

            # Combine forward and backward predictions
            forward_factor = np.arange(start=0, stop=y_hat_comp.shape[2]) / f
            backward_factor = (f - np.arange(start=0, stop=y_hat_comp.shape[2])) / f
            y_hat_comp = (
                    y_hat_comp[0].transpose(0, 2, 3, 1) * forward_factor * 255 +
                    y_hat_comp[1].transpose(0, 2, 3, 1) * backward_factor * 255
            ).transpose(0, 3, 1, 2).astype(np.uint8)

            # Log 4 middle frames inside TensorBoard
            y_hat_comp_tbx = [
                y_hat_comp[:, f] for f in range(y_hat_comp.shape[1] // 2 - 2, y_hat_comp.shape[1] // 2 + 2)
            ]
            self.experiment.tbx.add_images(
                'test_sequences/{}'.format(info[0]), y_hat_comp_tbx, global_step=self.counters['epoch'],
                dataformats='CHW'
            )

            # Save the sequence in disk
            self._save_sample(y_hat_comp, 'test', info[0], True)

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
            self._save_sample(y_aligned, 'test_alignment', info[0], save_as_video)

    def _compute_loss(self, y_t, y_hat, y_hat_comp, x_t, x_aligned, v_map, m, c_mask):
        # Retrieve configuration parameters
        loss_constant_normalization = self.experiment.configuration.get('model', 'loss_constant_normalization')
        loss_weights = self.experiment.configuration.get('model', 'loss_lambdas')

        # Loss 1: Alignment Loss
        if self.experiment.configuration.get('model', 'use_aligner'):
            alignment_input = x_aligned * v_map
            alignment_target = x_t.unsqueeze(2).repeat(1, 1, x_aligned.size(2), 1, 1) * v_map
            if loss_constant_normalization:
                loss_alignment = F.l1_loss(alignment_input, alignment_target)
            else:
                loss_alignment = F.l1_loss(alignment_input, alignment_target, reduction='sum') / torch.sum(v_map)
            loss_alignment *= loss_weights[0]
        else:
            loss_alignment = torch.zeros(1).to(y_hat.device)

        # Loss 2: Visible Hole
        vh_input = m * (1 - c_mask) * y_hat
        vh_target = m * (1 - c_mask) * y_t
        if loss_constant_normalization:
            loss_vh = F.l1_loss(vh_input, vh_target)
        else:
            loss_vh = F.l1_loss(vh_input, vh_target, reduction='sum') / torch.sum(m * (1 - c_mask))
        loss_vh *= loss_weights[1]

        # Loss 3: Non-Visible Hole
        nvh_input = m * c_mask * y_hat
        nvh_target = m * c_mask * y_t
        if loss_constant_normalization:
            loss_nvh = F.l1_loss(nvh_input, nvh_target)
        else:
            loss_nvh = F.l1_loss(nvh_input, nvh_target, reduction='sum') / torch.sum(m * c_mask)
        loss_nvh *= loss_weights[2]

        # Loss 4: Non-Hole
        nh_input = (1 - m) * y_hat
        nh_target = (1 - m) * y_t
        if loss_constant_normalization:
            loss_nh = F.l1_loss(nh_input, nh_target)
        else:
            loss_nh = F.l1_loss(nh_input, nh_target, reduction='sum') / torch.sum(1 - m)
        loss_nh *= loss_weights[3]

        # User VGG-16 to compute features of both the estimation and the target
        with torch.no_grad():
            vgg_y = self.model_vgg(y_t.contiguous())
            vgg_y_hat_comp = self.model_vgg(y_hat_comp.contiguous())

        # Loss 5: Perceptual
        loss_perceptual = 0
        for p in range(len(vgg_y)):
            loss_perceptual += F.l1_loss(vgg_y_hat_comp[p], vgg_y[p])
        loss_perceptual /= len(vgg_y)
        loss_perceptual *= loss_weights[4]

        # Loss 6: Style
        loss_style = 0
        for p in range(len(vgg_y)):
            b, c, h, w = vgg_y[p].size()
            g_y = torch.mm(vgg_y[p].view(b * c, h * w), vgg_y[p].view(b * c, h * w).t())
            g_y_comp = torch.mm(vgg_y_hat_comp[p].view(b * c, h * w), vgg_y_hat_comp[p].view(b * c, h * w).t())
            loss_style += F.l1_loss(g_y_comp / (b * c * h * w), g_y / (b * c * h * w))
        loss_style /= len(vgg_y)
        loss_style *= loss_weights[5]

        # Loss 7: Smoothing Checkerboard Effect
        loss_tv_h = (y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]).pow(2).sum()
        loss_tv_w = (y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]).pow(2).sum()
        loss_tv = (loss_tv_h + loss_tv_w) / (y_hat.size(0) * y_hat.size(1) * y_hat.size(2) * y_hat.size(3))
        loss_tv *= loss_weights[6]

        # Compute combined loss
        loss = loss_alignment + loss_vh + loss_nvh + loss_nh + loss_perceptual + loss_style + loss_tv

        # Return combination of the losses
        return loss, [loss_alignment, loss_vh, loss_nvh, loss_nh, loss_perceptual, loss_style, loss_tv]

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
