import skeltorch
from .model import CPNet
import torch.optim
import numpy as np
from PIL import Image
import os.path
import utils
import torch.nn.functional as F
from vgg_16.model import get_pretrained_model
import torch.utils.data


class CopyPasteRunner(skeltorch.Runner):
    model_vgg = None

    def init_model(self, device):
        self.model = CPNet().to(device)
        self.model_vgg = get_pretrained_model().to(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train_step(self, it_data, device):
        # Decompose iteration data
        (x, m), y, _ = it_data

        # Move data to the correct device
        x = x.to(device)
        m = m.to(device)
        y = y.to(device)

        # Get target and reference frames
        t = x.size(2) // 2
        r_list = list(range(x.size(2)))
        r_list.pop(t)

        # Propagate through the model
        y_hat, y_hat_comp, c_mask, (x_rt, m_rt) = self.model(x, m, y, t, r_list)

        # Get visibility map of aligned frames and target frame
        visibility_maps = (1 - m[:, :, t].unsqueeze(2)) * m_rt

        # Compute loss and return
        return self.compute_loss(y[:, :, t], y_hat, y_hat_comp, x[:, :, t], x_rt, visibility_maps, m[:, :, t], c_mask)

    def train_after_epoch_tasks(self, device):
        # Create provisional DataLoader with the randomly selected samples and select 5 items
        loader = torch.utils.data.DataLoader(self.experiment.data.datasets['train'], shuffle=True, batch_size=5)
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

        # Log each image of the batch in TensorBoard
        self.experiment.tbx.add_images('x-t', x[:, :, t], global_step=self.counters['epoch'])
        self.experiment.tbx.add_images('m-t', m[:, :, t], global_step=self.counters['epoch'])
        self.experiment.tbx.add_images('y', y[:, :, t], global_step=self.counters['epoch'])
        self.experiment.tbx.add_images('y-hat', y_hat, global_step=self.counters['epoch'])
        self.experiment.tbx.add_images('y-hat-comp', y_hat_comp, global_step=self.counters['epoch'])
        self.experiment.tbx.flush()

    def compute_loss(self, y, y_hat, y_hat_comp, x_t, x_rt, v, m, c_mask):
        # Loss 1: Alignment Loss
        alignment_input = x_rt * v
        alignment_target = x_t.unsqueeze(2).repeat(1, 1, x_rt.size(2), 1, 1) * v
        loss_alignment = F.l1_loss(alignment_input, alignment_target, reduction='sum') / torch.sum(v)

        # Loss 2: Visible Hole
        vh_input = m * c_mask * y_hat
        vh_target = m * c_mask * y
        loss_vh = F.l1_loss(vh_input, vh_target)

        # Loss 3: Non-Visible Hole
        nvh_input = m * (1 - c_mask) * y_hat
        nvh_target = m * (1 - c_mask) * y
        loss_nvh = F.l1_loss(nvh_input, nvh_target)

        # Loss 4: Non-Hole
        nh_input = (1 - m) * c_mask * y_hat
        nh_target = (1 - m) * c_mask * y
        loss_nh = F.l1_loss(nh_input, nh_target)

        # Loss 5: Perceptual
        loss_perceptual = 0
        with torch.no_grad():
            _, vgg_y = self.model_vgg(y)
            _, vgg_y_hat_comp = self.model_vgg(y_hat_comp)
            for p in range(len(vgg_y)):
                loss_perceptual += F.l1_loss(vgg_y_hat_comp[p], vgg_y[p])
            loss_perceptual /= len(vgg_y)

        # Loss 6: Style
        loss_style = 0
        for p in range(len(vgg_y)):
            B, C, H, W = vgg_y[p].size()
            G_y = torch.mm(vgg_y[p].view(B * C, H * W), vgg_y[p].view(B * C, H * W).t())
            G_y_comp = torch.mm(vgg_y_hat_comp[p].view(B * C, H * W), vgg_y_hat_comp[p].view(B * C, H * W).t())
            loss_style += F.l1_loss(G_y_comp / (B * C * H * W), G_y / (B * C * H * W))
        loss_style /= len(vgg_y)

        # Loss 7: Smoothing Checkerboard Effect
        loss_smoothing = 0

        # Return combination of the losses
        return 2 * loss_alignment + 10 * loss_vh + 20 * loss_nvh + 6 * loss_nh + 0.01 * loss_perceptual + \
               24 * loss_style + 0.1 * loss_smoothing

    def test(self):
        # Set model to evaluation mode
        self.model.eval()

        # Iterate through the data of the loader
        #    it_data[0] is (B, 3, F, H, W) and contains masked frames
        #    it_data[1] is (B, 1, F, H, W) and contains masks
        #    it_target contains ground truth frames
        #    it_info contains the index of the video
        for it_data, it_target, it_info in self.experiment.data.loaders['test']:

            # Get relevant sizes of the iteration
            B, C, F, H, W = it_data[0].size()

            # Create a matrix to store inpainted frames. Size (B, 2, C, F, H, W), where the 2 is due to double direction
            frames_inpainted = np.zeros((B, 2, C, F, H, W), dtype=np.float32)

            # Create a list containing the indexes of the frames
            index = [f for f in range(it_data[0].size(2))]

            # Move data to cpu for memory purposes
            it_data[0] = it_data[0].cpu()
            it_data[1] = it_data[1].cpu()

            # Use the model twice: forward (0) and backward (1)
            for t in range(2):
                # Create two aux variables to store input frames and masks in current direction
                # Move to current device for memory purposes
                input_frames = it_data[0].clone().to(self.execution.device)
                input_masks = it_data[1].clone().to(self.execution.device)

                # Reverse the indexes in backward pass
                if t == 1:
                    index.reverse()

                # Iterate over all the frames of the video
                for f in index:
                    # Obtain a list containing the references frames of the current target frame
                    reference_frames = CopyPasteRunner.get_reference_frame_indexes(f, it_data[0].size(2))

                    # Replace input_frames and input_masks with previous predictions to improve quality
                    with torch.no_grad():
                        input_frames[:, :, f], _, _, _ = self.model(
                            input_frames, input_masks, it_target, f, reference_frames
                        )
                        input_masks[:, :, f] = 0

                    # Obtain an estimation of the inpainted frame f
                    frames_inpainted[:, t, :, f] = input_frames[:, :, f].detach().cpu().numpy()

            # Create forward and backward factors and combine them.
            # Transpose frames_inpainted to be (B,F,H,W,C)
            # Change scale from float64 [0,1] to uint8 [0,255]
            forward_factor = np.arange(start=0, stop=frames_inpainted.shape[3]) / len(index)
            backward_factor = (len(index) - np.arange(start=0, stop=frames_inpainted.shape[3])) / len(index)
            frames_inpainted = (frames_inpainted[:, 0].transpose(0, 1, 3, 4, 2) * forward_factor +
                                frames_inpainted[:, 1].transpose(0, 1, 3, 4, 2) * backward_factor
                                ).transpose(0, 4, 2, 3, 1)
            frames_inpainted = (frames_inpainted * 255.).astype(np.uint8)

            # Iterate over batch items and create the result for each one
            for b in range(frames_inpainted.shape[0]):

                # Save the result as frames or videos depending on the request
                if self.execution.args['save_as_video']:
                    frames_to_video = utils.FramesToVideo(0, 10, None)
                    frames_to_video.add_sequence(frames_inpainted[b])
                    frames_to_video.save(self.execution.args['data_output'], it_info[b])
                else:
                    frames_path = os.path.join(self.execution.args['data_output'], it_info[b])
                    if not os.path.exists(frames_path):
                        os.makedirs(os.path.join(frames_path))
                    for f in range(frames_inpainted.shape[1]):
                        pil_img = Image.fromarray(frames_inpainted[b, f])
                        pil_img.save(os.path.join(frames_path, '{}.jpg'.format(f)))

                # Log progress
                self.logger.info('Test generated as {} for video {}'.format(
                    'video' if self.execution.args['save_as_video'] else 'frames', it_info[b]
                ))

    def test_alignment(self):
        # Set model to evaluation mode
        self.model.eval()

        # Iterate through the data of the loader
        #    it_data is a tuple containing masked frames (pos=0) and masks (pos=1)
        #    it_target contains the ground truth frames
        #    it_info contains the index of the video
        for it_data, it_target, it_info in self.experiment.data.loaders['test']:

            # Set a frame index and obtain its reference frames
            target_index = 0
            reference_indexes = CopyPasteRunner.get_reference_frame_indexes(target_index, it_data[0].size(2))

            # Obtained aligned frames (provisional)
            _, _, aligned_gts = self.model.align(it_data[0], it_data[1], it_target, target_index, reference_indexes)

            # Create a numpy array of size (B,F,H,W,C)
            aligned_gts = (aligned_gts.detach().cpu().permute(0, 2, 3, 4, 1).numpy() * 255).astype(np.uint8)

            # Iterate over batch items and create the result for each one
            for b in range(aligned_gts.shape[0]):

                # Save the result as overlay frames or videos depending on the request
                if self.execution.args['save_as_video']:
                    frames_to_video = utils.FramesToVideo(0, 10, None)
                    frames_to_video.add_sequence(aligned_gts[b])
                    frames_to_video.save(self.execution.args['data_output'], it_info[b])
                else:
                    overlap_frames = utils.OverlapFrames(target_index, 50, 10)
                    overlap_frames.add_sequence(aligned_gts[b])
                    overlap_frames.save(self.execution.args['data_output'], it_info[b])

                # Log progress
                self.logger.info('Alignment test generated as {} for video {}'.format(
                    'video' if self.execution.args['save_as_video'] else 'image overlay', it_info[b]
                ))

    @staticmethod
    def get_reference_frame_indexes(target_frame, num_frames, num_length=120):
        # Set start and end frames
        start = target_frame - num_length
        end = target_frame + num_length

        # Adjust frames in case they are not in the limit
        if target_frame - num_length < 0:
            end = (target_frame + num_length) - (target_frame - num_length)
            end = num_frames - 1 if end > num_frames else end
            start = 0
        elif target_frame + num_length > num_frames:
            start = (target_frame - num_length) - (target_frame + num_length - num_frames)
            start = 0 if start < 0 else start
            end = num_frames - 1

        # Return list of reference_frames every n=2 frames
        return [i for i in range(start, end, 2) if i != target_frame]
