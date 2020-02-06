import os
import torch.utils.data
from PIL import Image
import glob
import numpy as np


class DAVIS2017Dataset(torch.utils.data.Dataset):

    def __init__(self, split, data_folder, dataset_folder_name='DAVIS-2017'):
        self.split = split
        self.dataset_folder = os.path.join(data_folder, dataset_folder_name)
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

    def __getitem__(self, indx):
        images_filenames = sorted(glob.glob(os.path.join(self.images_folder, self.items[indx], '*.jpg')))
        annotations_filenames = sorted(glob.glob(os.path.join(self.annotations_folder, self.items[indx], '*.png')))
        images = np.zeros((len(images_filenames), 480, 854, 3), dtype=float)
        annotations = np.zeros((len(annotations_filenames), 480, 854), dtype=int)
        for i in range(len(images_filenames)):
            images[i] = np.array(Image.open(images_filenames[i]).convert('RGB')) / 255
            annotations[i] = np.array(Image.open(annotations_filenames[i]))
        return images, annotations, self.items[indx]

    def __len__(self):
        return len(self.items)
