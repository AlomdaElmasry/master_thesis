import skeltorch
from .model import CPNet
import torch.optim
import numpy as np
from PIL import Image
import os.path
import matplotlib.pyplot as plt


class CopyPasteRunner(skeltorch.Runner):

    def init_model(self):
        self.model = CPNet().to(self.execution.device)

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

            # Clone the masked frames and the masks
            frames_ = it_data[0].clone()
            masks_ = it_data[1].clone()

            # Create a list containing the indexes of the frames
            index = [f for f in reversed(range(it_data[0].size(2)))]

            # Use the model twice: forward (0) and backward (1)
            for t in range(2):  # forward : 0, backward : 1

                # TO BE EXPLAINED
                if t == 1:
                    comp0 = it_data[0].clone()
                    it_data[0] = frames_
                    it_data[1] = masks_
                    index.reverse()

                # Iterate over all the frames of the video
                for f in index:
                    # Obtain a list containing the references frames of the current target frame
                    ridx = CopyPasteRunner.get_reference_frame_indexes(f, it_data[0].size(2))

                    # Inpainting Part
                    with torch.no_grad():

                        # Obtain an estimation of the inpainted frame
                        comp = self.model(it_data[0], it_data[1], it_target, f, ridx)

                        # Shape: batch x 3 x 240 x 424 (estimation of the image)
                        c_s = comp.shape

                        # Update frame and mask with the predictions -> mask is all zeros
                        it_data[0][:, :, f] = comp.detach()
                        it_data[1][:, :, f] = torch.zeros((c_s[0], 1, 1, c_s[2], c_s[3])).float().to(self.execution.device)

                        print('feat done')

                    # TO BE EXPLAINED
                    save_path = self.execution.args['data_output']
                    if t == 1:
                        est = comp0[:, :, f].cpu() * (len(index) - f) / len(index) + comp.detach().cpu() * f / len(index)
                        canvas = (est[0].permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
                        if canvas.shape[1] % 2 != 0:
                            canvas = np.pad(canvas, [[0, 0], [0, 1], [0, 0]], mode='constant')
                        canvas = Image.fromarray(canvas)
                        canvas.save(os.path.join(save_path, 'f{}.jpg'.format(f)))
                        print('frame done')

    def test_alignment(self):
        """Lalala"""

        # Set model to evaluation mode
        self.model.eval()

        # Iterate through the data of the loader
        #    it_data is a tuple containing masked frames (pos=0) and masks (pos=1)
        #    it_target contains the ground truth frames
        #    it_info contains the index of the video
        for it_data, it_target, it_info in self.experiment.data.loaders['train']:

            # Get alignment features
            with torch.no_grad():

                # Set a frame index and obtain its reference frames
                target_index = 0
                reference_indexes = CopyPasteRunner.get_reference_frame_indexes(target_index, it_data[0].size(2))

                # Obtained aligned frames (provisional)
                aligned_frames = self.model.align_frames(
                    it_data[0], it_data[1], it_target, target_index, reference_indexes
                )

                for i in range(aligned_frames.size(3)):
                    np_image = aligned_frames[0, :, i, :].permute(1, 2, 0).squeeze(2).numpy()
                    plt.imshow(np_image)
                    plt.show()
                exit()


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
