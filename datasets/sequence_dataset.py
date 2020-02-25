import os
import torch.utils.data
from PIL import Image
import glob
import numpy as np
from utils.transforms import ImageTransforms
import matplotlib.pyplot as plt


class SequencesDataset(torch.utils.data.Dataset):
    dataset_name = None
    dataset_folder = None
    split = None
    image_size = None
    frames_n = None
    frames_spacing = None
    dilatation_filter_size = None
    dilatation_iterations = None
    logger = None
    masks_dataset = None
    device = None

    sequences_gts = None
    sequences_masks = None
    channels_mean = None

    def __init__(self, dataset_name, dataset_folder, split, image_size, frames_n, frames_spacing,
                 dilatation_filter_size, dilatation_iterations, logger, masks_dataset=None):
        self.dataset_name = dataset_name
        self.dataset_folder = dataset_folder
        self.split = split
        self.image_size = image_size
        self.frames_n = frames_n
        self.frames_spacing = frames_spacing
        self.dilatation_filter_size = dilatation_filter_size
        self.dilatation_iterations = dilatation_iterations
        self.logger = logger
        self.masks_dataset = masks_dataset
        self._validate_arguments()
        self._load_paths()
        self._create_index()
        self.channels_mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32)

    def _validate_arguments(self):
        if not os.path.exists(self.dataset_folder):
            raise ValueError('Dataset folder {} does not exist.'.format(self.dataset_folder))
        assert self.dataset_name in ['davis-2017', 'got-10k']
        assert self.split in ['train', 'validation', 'test']
        assert self.frames_n == -1 or self.frames_n % 2 == 1
        assert not (self.dataset_name == 'got-10k' and self.masks_dataset is None)

    def _load_paths(self):
        if self.dataset_name == 'davis-2017':
            self._load_paths_davis2017()
        elif self.dataset_name == 'got-10k':
            self._load_paths_got10k()

    def _load_paths_davis2017(self):
        gts_folder = os.path.join(self.dataset_folder, 'JPEGImages', '480p')
        masks_folder = os.path.join(self.dataset_folder, 'Annotations', '480p')
        split_filename = 'train.txt' if self.split == 'train' else 'val.txt'
        with open(os.path.join(self.dataset_folder, 'ImageSets', '2017', split_filename)) as items_file:
            self.sequences_names = items_file.read().splitlines()
        self.sequences_gts = [sorted(glob.glob(os.path.join(gts_folder, sequence_name, '*.jpg')))
                              for sequence_name in self.sequences_names]
        self.sequences_masks = [sorted(glob.glob(os.path.join(masks_folder, sequence_name, '*.png')))
                                for sequence_name in self.sequences_names]

    def _load_paths_got10k(self):
        split_folder = 'train' if self.split == 'train' else 'val' if self.split == 'validation' else 'test'
        gts_folder = os.path.join(self.dataset_folder, split_folder)
        with open(os.path.join(gts_folder, 'list.txt')) as items_file:
            self.sequences_names = items_file.read().splitlines()
        self.sequences_gts = [sorted(glob.glob(os.path.join(gts_folder, sequence_name, '*.jpg')))
                              for sequence_name in self.sequences_names]

    def _create_index(self):
        self.sequences_limits = np.cumsum([len(images) for images in self.sequences_gts])

    def __getitem__(self, item):
        # Case 1: return all the frames of the sequence. Each item is associated to one sequence_index.
        if self.frames_n == -1:
            sequence_index = item
            frames_indexes = list(range(len(self.sequences_gts[sequence_index])))

        # Case 2: return self.n_frames containing the target frame in the central position. Each item of the dataset is
        # associated with the frame at position frame_index of the sequence sequence_index. Return all the frames if
        # n_frames = -1. Otherwise, return self.n_frames // 2 in each direction. Clip this last case between 0 and the
        # length of the sequence. Interpretation: extreme cases mirror frames.
        else:
            sequence_index = next(x[0] for x in enumerate(self.sequences_limits) if x[1] > item)
            frame_index = item - (self.sequences_limits[sequence_index - 1] if sequence_index > 0 else 0)
            frames_indexes = list(range(
                frame_index - (self.frames_n // 2) * self.frames_spacing,
                frame_index + (self.frames_n // 2) * self.frames_spacing + 1,
                self.frames_spacing
            ))
            frames_indexes = np.clip(frames_indexes, 0, len(self.sequences_gts[sequence_index]) - 1)

        # Create torch.Tensor for the GTs
        gts = torch.zeros((3, len(frames_indexes), self.image_size[0], self.image_size[1]), dtype=torch.float32)

        # Get the mask from the mask generator or create torch.Tensor to be filled
        if self.masks_dataset is None:
            masks = torch.zeros((1, len(frames_indexes), self.image_size[0], self.image_size[1]), dtype=torch.float32)
        else:
            masks = self.masks_dataset.get_random_item(len(frames_indexes))
            masks = torch.stack([self._transform_mask(masks[:, m]) for m in range(masks.size(1))], dim=0).unsqueeze(0)

        # Store previous crop position to create coherent cuts
        crop_position = None

        # Iterate all the files
        for i, f in enumerate(frames_indexes):
            # Load the frame and perform a random resize / crop of the given size
            frame_gt = torch.from_numpy((np.array(Image.open(self.sequences_gts[sequence_index][f])
                                                  .convert('RGB')) / 255).astype(np.float32)).permute(2, 0, 1)
            gts[:, i], crop_position = self._transform_gt(frame_gt, crop_position)

            # Check whether to use the mask of the instance or a random one
            if self.masks_dataset is None:
                frame_mask = torch.from_numpy((np.array(Image.open(self.sequences_masks[sequence_index][f])
                                                        .convert('RGB')) / 255).astype(np.float32)).permute(2, 0, 1)
                masks[:, i] = self._transform_mask(frame_mask)

        # Compute masked data
        masked_sequences = (1 - masks) * gts + (masks.permute(3, 2, 1, 0) * self.channels_mean).permute(3, 2, 1, 0)

        # Return framed sequence as (C,F,H,W)
        return (masked_sequences, masks), gts, self.sequences_names[sequence_index]

    def __len__(self):
        return len(self.sequences_gts) if self.frames_n == -1 else self.sequences_limits[-1]

    def _transform_gt(self, frame_gt, crop_position):
        return ImageTransforms.crop(frame_gt, self.image_size, crop_position) \
            if self.split in ['train', 'validation'] else (ImageTransforms.resize(frame_gt, self.image_size), None)

    def _transform_mask(self, frame_mask):
        frame_mask = ImageTransforms.resize(frame_mask, self.image_size)
        frame_mask = ImageTransforms.dilatate(frame_mask, self.dilatation_filter_size, self.dilatation_iterations)
        return ImageTransforms.binarize(frame_mask)
