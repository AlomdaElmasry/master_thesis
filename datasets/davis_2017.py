import os
import torch.utils.data
from PIL import Image
import glob
import numpy as np


class DAVIS2017Dataset(torch.utils.data.Dataset):

    def __init__(self, split, dataset_folder):
        self.split = split
        self.dataset_folder = dataset_folder
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
        annotations_filenames = sorted(glob.glob(os.path.join(self.annotations_folder, self.items[item], '*.png')))
        images = torch.zeros((3, len(images_filenames), 480, 854))
        annotations = torch.zeros((1, len(annotations_filenames), 480, 854))
        for i in range(len(images_filenames)):
            images[:, i, :, :] = torch.from_numpy(
                np.array(Image.open(images_filenames[i]).convert('RGB')) / 255
            ).permute(2, 0, 1)
            annotations[:, i, :, :] = torch.from_numpy(np.array(Image.open(annotations_filenames[i]))).unsqueeze(0)
        return images, annotations

    def __len__(self):
        return len(self.items)
