import skeltorch
from .model import CPNet
import torch.nn as nn
import torch.optim
import numpy as np
from PIL import Image
import os.path


class CopyPasteRunner(skeltorch.Runner):

    def init_model(self):
        self.model = nn.DataParallel(CPNet()).to(self.execution.device)
        self.model = CPNet()
        b = 1

    def load_checkpoint(self):
        pass

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def step_train(self, it_data: any, it_target: any, it_info: any):
        pass

    def step_validation(self, it_data: any, it_target: any, it_info: any):
        pass

    def train(self):
        raise NotImplementedError

    def test(self):

        # Set model to evaluation mode
        self.model.eval()

        # Iterate through the data of the loader
        #    it_data is a tuple containing masked frames (pos=0) and masks (pos=1)
        #    it_target contains the ground truth frames
        #    it_info contains the index of the video
        for it_data, it_target, it_info in self.experiment.data.loaders['train']:

            frames = it_data[0].to(self.execution.device)
            masks = it_data[1].to(self.execution.device)
            GTs = it_target.to(self.execution.device)

            # Get alignment features
            with torch.no_grad():
                rfeats = self.model(frames, masks)

            # Clone the masked frames and the masks
            frames_ = frames.clone()
            masks_ = masks.clone()

            # Create a list containing the indexes of the frames
            index = [f for f in reversed(range(frames.size(2)))]

            # Use the model twice: forward (0) and backward (1)
            for t in range(2):  # forward : 0, backward : 1

                # TO BE EXPLAINED
                if t == 1:
                    comp0 = frames.clone()
                    frames = frames_
                    masks = masks_
                    index.reverse()

                # Iterate over all the frames of the video
                for f in index:
                    # Obtain a list containing the references frames of the current target frame
                    ridx = CopyPasteRunner.get_reference_frame_indexes(f, it_data[0].size(2))

                    # Inpainting Part
                    with torch.no_grad():

                        # Obtain an estimation of the inpainted frame
                        comp = self.model(
                            rfeats[:, :, ridx],
                            frames[:, :, ridx],
                            masks[:, :, ridx],
                            frames[:, :, f],
                            masks[:, :, f],
                            GTs[:, :, f]
                        )

                        # Shape: batch x 3 x 240 x 424 (estimation of the image)
                        c_s = comp.shape

                        # Fs shape: batch x 3 x 1 (frames) x 240 x 424
                        Fs = torch.empty((c_s[0], c_s[1], 1, c_s[2], c_s[3])).float().to(self.execution.device)

                        # Hs shape: batch x 1 x 1 x 240 x 424 (mask) -> no mask
                        Hs = torch.zeros((c_s[0], 1, 1, c_s[2], c_s[3])).float().to(self.execution.device)

                        # Set FS to the output prediction
                        Fs[:, :, 0] = comp.detach()

                        # Update frame and mask with the predictions -> mask is all zeros
                        frames[:, :, f] = Fs[:, :, 0]
                        masks[:, :, f] = Hs[:, :, 0]

                        # Update rfeats of the frame
                        rfeats[:, :, f] = self.model(Fs, Hs)[:, :, 0]

                        print('feat done')

                    # TO BE EXPLAINED
                    save_path = '/Users/DavidAlvarezDLT/Desktop/test_mine'
                    if t == 1:
                        est = comp0[:, :, f].cpu() * (len(index) - f) / len(index) + comp.detach().cpu() * f / len(index)
                        canvas = (est[0].permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
                        if canvas.shape[1] % 2 != 0:
                            canvas = np.pad(canvas, [[0, 0], [0, 1], [0, 0]], mode='constant')
                        canvas = Image.fromarray(canvas)
                        canvas.save(os.path.join(save_path, 'f{}.jpg'.format(f)))
                        print('frame done')

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
