import os
import torch.utils.data
from PIL import Image
import glob
import numpy as np


class FramedDataset(torch.utils.data.Dataset):
    sequences_gts = None
    sequences_masks = None

    def __init__(self, dataset_folder, dataset_name, split, n_frames=-1, gt_transforms=None, mask_transforms=None,
                 device='cpu'):
        self.split = split
        self.dataset_folder = dataset_folder
        self.dataset_name = dataset_name
        self.n_frames = n_frames
        self.device = device
        self.gt_transforms = gt_transforms
        self.mask_transforms = mask_transforms
        self._validate_arguments()
        self._load_paths()
        self._create_index()

    def _validate_arguments(self):
        assert os.path.exists(self.dataset_folder)
        assert self.dataset_name in ['davis-2017', 'youtube-vos']
        assert self.split in ['train', 'validation', 'test']
        assert self.n_frames == -1 or self.n_frames % 2 == 1

    def _load_paths(self):
        if self.dataset_name == 'davis-2017':
            self._load_paths_davis2017()
        elif self.dataset_name == 'youtube-vos':
            self._load_paths_youtubevos()

    def _load_paths_davis2017(self):
        gts_folder = os.path.join(self.dataset_folder, 'JPEGImages', '480p')
        masks_folder = os.path.join(self.dataset_folder, 'Annotations', '480p')
        split_filename = 'train.txt' if self.split == 'train' else 'val.txt'
        with open(os.path.join(self.dataset_folder, 'ImageSets', '2017', split_filename)) as items_file:
            self.sequence_names = items_file.read().splitlines()
            self.sequences_gts = [sorted(glob.glob(os.path.join(gts_folder, item, '*.jpg'))) for item in
                                  self.sequence_names]
            self.sequences_masks = [sorted(glob.glob(os.path.join(masks_folder, item, '*.png'))) for item in
                                    self.sequence_names]

    def _load_paths_youtubevos(self):
        split_folder = 'train' if self.split == 'train' else 'valid' if self.split == 'validation' else 'test'
        gts_folder = os.path.join(self.dataset_folder, split_folder, 'JPEGImages')
        masks_folder = os.path.join(self.dataset_folder, split_folder, 'Annotations')
        self.sequences_names = os.listdir(gts_folder)
        self.sequences_gts = [sorted(glob.glob(os.path.join(gts_folder, item, '*.jpg'))) for item in
                              self.sequence_names]

        a = 1

    def _create_index(self):
        self.sequences_limits = np.cumsum([len(images) for images in self.sequences_gts])

    def __getitem__(self, item):

        # Case 1: return all the frames of the sequence. Each item is associated to one sequence_index.
        if self.n_frames == -1:
            sequence_index = item
            frames_indexes = list(range(len(self.sequences_gts[sequence_index])))

        # Case 2: return self.n_frames containing the target frame in the central position. Each item of the dataset is
        # associated with the frame at position frame_index of the sequence sequence_index. Return all the frames if
        # n_frames = -1. Otherwise, return self.n_frames // 2 in each direction. Clip this last case between 0 and the
        # length of the sequence. Interpretation: extreme cases mirror frames.
        else:
            sequence_index = next(x[0] for x in enumerate(self.sequences_limits) if x[1] > item)
            frame_index = item - (self.sequences_limits[sequence_index - 1] if sequence_index > 0 else 0)
            frames_indexes = list(range(frame_index - self.n_frames // 2, frame_index + self.n_frames // 2 + 1))
            frames_indexes = np.clip(frames_indexes, 0, len(self.sequences_gts[sequence_index]) - 1)

        # Create variables to return
        gts, masks = [], []

        # Iterate all the files
        for f in frames_indexes:

            # Load both the frame and the masks as Numpy arrays -> (H, W, C)
            frame_gt = np.array(Image.open(self.sequences_gts[sequence_index][f])) / 255
            frame_mask = np.expand_dims(np.array(Image.open(self.sequences_masks[sequence_index][f])), 2)

            # Apply transforms to the frame
            if self.gt_transforms is not None:
                frame_gt = self.gt_transforms(frame_gt)

            # Apply transforms to the mask
            if self.mask_transforms is not None:
                frame_mask = self.mask_transforms(frame_mask)

            # Store both frame and mask as Tensors
            gts.append(torch.from_numpy(frame_gt).permute(2, 0, 1).to(self.device))
            masks.append(torch.from_numpy(frame_mask).permute(2, 0, 1).to(self.device))

        # Return framed sequence as (C,F,H,W)
        return torch.stack(gts, dim=1), torch.stack(masks, dim=1), self.sequence_names[sequence_index]

    def __len__(self):
        return len(self.sequences_gts) if self.n_frames == -1 else self.sequences_limits[-1]