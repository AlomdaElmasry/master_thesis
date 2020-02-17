import os
import torch.utils.data
from PIL import Image
import glob
import numpy as np
import cv2


class DAVIS2017Dataset(torch.utils.data.Dataset):

    def __init__(self, split, dataset_folder, device='cpu', frame_transforms=None, mask_transforms=None):
        self.split = split
        self.dataset_folder = dataset_folder
        self.images_folder = os.path.join(self.dataset_folder, 'JPEGImages', '480p')
        self.annotations_folder = os.path.join(self.dataset_folder, 'Annotations', '480p')
        self.device = device
        self.frame_transforms = frame_transforms
        self.mask_transforms = mask_transforms
        self._validate_arguments()
        self._load_split()

    def _validate_arguments(self):
        assert self.split in ['train', 'val']
        assert os.path.exists(self.dataset_folder)

    def _load_split(self):
        split_filename = 'train.txt' if self.split == 'train' else 'val.txt'
        with open(os.path.join(self.dataset_folder, 'ImageSets', '2017', split_filename)) as items_file:
            self.items = items_file.read().splitlines()

    def __getitem__(self, item):
        # Get lists of both frames and masks files
        images_filenames = sorted(glob.glob(os.path.join(self.images_folder, self.items[item], '*.jpg')))
        masks_filenames = sorted(glob.glob(os.path.join(self.annotations_folder, self.items[item], '*.png')))

        # Create variables to return
        frames, masks = None, None

        # Iterate all the files
        for i in range(len(images_filenames)):
            # Load both the frame and the masks as Numpy arrays -> (H, W, C)
            frame = np.array(Image.open(images_filenames[i]).convert('RGB')) / 255
            mask = np.expand_dims(np.array(Image.open(masks_filenames[i]).convert('P')), 2)

            # Apply transforms to the frame
            if self.frame_transforms is not None:
                frame = self.frame_transforms(frame)

            # Apply transforms to the mask
            if self.mask_transforms is not None:
                mask = self.mask_transforms(mask)

            # Allocate space for both frames and masks
            if frames is None or masks is None:
                frames = torch.zeros((frame.shape[2], len(images_filenames), frame.shape[0], frame.shape[1]),
                                     device=self.device)
                masks = torch.zeros((mask.shape[2], len(masks_filenames), mask.shape[0], mask.shape[1]),
                                    device=self.device)

            # Store both frame and mask as Tensors
            frames[:, i, :, :] = torch.from_numpy(frame).permute(2, 0, 1)
            masks[:, i, :, :] = torch.from_numpy(mask).permute(2, 0, 1)

        return frames, masks, item

    def __len__(self):
        return len(self.items)
