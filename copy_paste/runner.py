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
        #    it_data[0] is (B, 3, F, H, W) and contains masked frames
        #    it_data[1] is (B, 1, F, H, W) and contains masks
        #    it_target contains ground truth frames
        #    it_info contains the index of the video
        for it_data, it_target, it_info in self.experiment.data.loaders['train']:

            # Get relevant sizes of the iteration
            B, C, F, H, W = it_data[0].size()

            # Create a matrix to store inpainted frames. Size (B, 2, C, F, H, W), where the 2 is due to double direction
            frames_inpainted = torch.zeros((B, 2, C, F, H, W), device='cpu')

            # Create a list containing the indexes of the frames
            index = [f for f in range(it_data[0].size(2))]

            # Use the model twice: forward (0) and backward (1)
            for t in range(2):

                # Create two aux variables to store input frames and masks in current direction
                # Note: could be done better creating a copy of it_data[0] and replacing those values directly.
                input_frames = it_data[0].clone()
                input_masks = it_data[1].clone()

                # Reverse the indexes in backward pass
                if t == 1:
                    index.reverse()

                # Iterate over all the frames of the video
                for f in index:
                    # Obtain a list containing the references frames of the current target frame
                    ridx = CopyPasteRunner.get_reference_frame_indexes(f, it_data[0].size(2))

                    # Obtain an estimation of the inpainted frame f
                    frames_inpainted[:, t, :, f] = self.model(input_frames, input_masks, it_target, f, ridx)

                    # Replace input_frames and input_masks with previous predictions to improve quality
                    input_frames[:, :, f] = frames_inpainted[:, t, :, f]
                    input_masks[:, :, f] = 0

            # Store in disk each inpainted frame
            for f in range(frames_inpainted.size(3)):
                # Obtain both forward and backward estimates
                forward_prediction = frames_inpainted[:, 0, :, f].cpu().squeeze(0).permute(1, 2, 0).numpy()
                backward_prediction = frames_inpainted[:, 1, :, f].cpu().squeeze(0).permute(1, 2, 0).numpy()

                # Combine both estimates giving more importance depending on whether the estimate has been made using
                # more or less auxiliar frames with mask
                final_predition = forward_prediction * f / len(index) + backward_prediction * (len(index) - f) / \
                                  len(index)

                # Convert the image to range [0, 255] and save it in disk
                pil_img = Image.fromarray((final_predition * 255.).astype(np.uint8))
                pil_img.save(os.path.join(self.execution.args['data_output'], 'f{}.jpg'.format(f)))

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
