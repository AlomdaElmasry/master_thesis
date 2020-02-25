import torch.utils.data
import os
import pycocotools.coco
import torch
import torch.nn.functional as F
import math
import numpy as np
import utils
import glob
from PIL import Image
import random


class MasksDataset(torch.utils.data.Dataset):
    dataset_name = None
    dataset_folder = None
    split = None
    emulator = None
    device = None

    def __init__(self, dataset_name, dataset_folder, split, emulator=None, device=torch.device('cpu')):
        self.dataset_name = dataset_name
        self.dataset_folder = dataset_folder
        self.split = split
        self.emulator = emulator
        self.device = device
        self._load_paths()

    def _validate_arguments(self):
        if not os.path.exists(self.dataset_folder):
            raise ValueError('Dataset folder {} does not exist.'.format(self.dataset_folder))
        assert self.dataset_name in ['coco', 'youtube-vos']
        assert self.split in ['train', 'validation', 'test']
        assert not (self.dataset_name == 'coco' and self.emulator is None)

    def _load_paths(self):
        if self.dataset_name == 'coco':
            self._load_paths_coco()
        elif self.dataset_name == 'youtube-vos':
            self._load_paths_youtube_vos()

    def _load_paths_coco(self):
        self.coco = pycocotools.coco.COCO(os.path.join(self.dataset_folder, 'instances_train2017.json'))
        self.sequences_masks = self.coco.getAnnIds(iscrowd=False)

    def _load_paths_youtube_vos(self):
        split_folder = 'train' if self.split == 'train' else 'valid' if self.split == 'validation' else 'test'
        annotations_folder = os.path.join(self.dataset_folder, split_folder, 'Annotations')
        self.sequences_names = os.listdir(annotations_folder)
        self.sequences_masks = [
            sorted(glob.glob(os.path.join(annotations_folder, sequence_name, '*.png')))
            for sequence_name in self.sequences_names
        ]

    def __getitem__(self, item):
        raise NotImplementedError

    def get_random_item(self, n):
        if self.dataset_name == 'coco':
            return self._get_random_frames_coco(n)
        elif self.dataset_name == 'youtube-vos':
            return self._get_random_frames_youtube_vos(n)

    def _get_random_frames_coco(self, n):
        item = random.randint(0, len(self.sequences_masks))
        mask = torch.from_numpy((np.array(
            Image.fromarray(self.coco.annToMask(self.coco.loadAnns(self.sequences_masks[item])[0])).convert('RGB')
        ).astype(np.float32))).permute(2, 0, 1)
        return self.emulator.simulate_movement(mask, n).to(self.device)

    def _get_random_frames_youtube_vos(self, n):
        item = random.randint(0, len(self.sequences_masks))
        if len(self.sequences_masks[item]) < n:
            return self._get_random_frames_youtube_vos(n)
        else:
            random_start = random.randint(0, len(self.sequences_masks[item]) - n)
            frames = []
            for f in range(random_start, random_start + n):
                frames.append(torch.from_numpy(
                    (np.array(Image.open(self.sequences_masks[item][f]).convert('RGB')) / 255).astype(np.float32)
                ).permute(2, 0, 1))
            return torch.stack(frames, dim=1).to(self.device)

    def __len__(self):
        return len(self.sequences_masks)
