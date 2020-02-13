import os
import torch.utils.data
from PIL import Image
import glob
import numpy as np
import cv2


class DAVIS2017Dataset(torch.utils.data.Dataset):

    def __init__(self, split, dataset_folder, image_size):
        self.split = split
        self.dataset_folder = dataset_folder
        self.image_size = image_size
        self.images_folder = os.path.join(self.dataset_folder, 'JPEGImages', '480p')
        self.annotations_folder = os.path.join(self.dataset_folder, 'Annotations', '480p')
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

        # Allocate space for both frames and masks
        images = torch.zeros((3, len(images_filenames), self.image_size[0], self.image_size[1]))
        masks = torch.zeros((1, len(masks_filenames), self.image_size[0], self.image_size[1]))

        # Iterate all the files
        for i in range(len(images_filenames)):
            # Load both the frame and the masks as Numpy arrays -> (H, W, C)
            image = np.array(Image.open(images_filenames[i]).convert('RGB')) / 255
            mask = np.array(Image.open(masks_filenames[i]).convert('P'))

            # Binarize mask so the values are (0,1)
            mask = (mask > 0.5).astype(np.uint8)

            # Resize both image and frames if the size is not correct
            if self.image_size != (image.shape[0], image.shape[1]):
                image = cv2.resize(image, dsize=(self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, dsize=(self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

            # Apply dilatation
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4).astype(np.float32)

            # Store both frame and mask as Tensors
            images[:, i, :, :] = torch.from_numpy(image).permute(2, 0, 1)
            masks[:, i, :, :] = torch.from_numpy(mask).unsqueeze(0)

        return images, masks, item

    def __len__(self):
        return len(self.items)
