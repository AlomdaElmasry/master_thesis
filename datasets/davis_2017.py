import os
import torch.utils.data
from PIL import Image
import glob
import numpy as np


class DAVIS2017Dataset(torch.utils.data.Dataset):

    def __init__(self, split, dataset_folder, image_size, transforms=None):
        self.split = split
        self.dataset_folder = dataset_folder
        self.image_size = image_size
        self.transforms = transforms
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
        images_filenames = sorted(glob.glob(os.path.join(self.images_folder, self.items[item], '*.jpg')))
        masks_filenames = sorted(glob.glob(os.path.join(self.annotations_folder, self.items[item], '*.png')))
        images = torch.zeros((3, len(images_filenames), self.image_size[0], self.image_size[1]))
        masks = torch.zeros((1, len(masks_filenames), self.image_size[0], self.image_size[1]))
        for i in range(len(images_filenames)):
            image_pil = Image.open(images_filenames[i]).convert('RGB')
            mask_pil = Image.open(masks_filenames[i]).convert('P')
            if self.transforms:
                image_pil = self.transforms(image_pil)
                mask_pil = self.transforms(mask_pil)
            images[:, i, :, :] = torch.from_numpy(np.array(image_pil) / 255).permute(2, 0, 1)
            masks[:, i, :, :] = torch.from_numpy(np.array(mask_pil)).unsqueeze(0)
        return images, masks, item

    def __len__(self):
        return len(self.items)
