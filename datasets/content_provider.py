import torch.utils.data
import numpy as np
from utils.paths import DatasetPaths
import random
import jpeg4py as jpeg
import cv2
import matplotlib.pyplot as plt


class ContentProvider(torch.utils.data.Dataset):
    dataset_name = None
    dataset_folder = None
    split = None
    movement_simulator = None
    logger = None
    return_gt = None
    return_mask = None
    items_names = None
    items_gts_paths = None
    items_masks_paths = None
    items_limits = None

    def __init__(self, dataset_name, data_folder, split, movement_simulator, logger, return_gt=True, return_mask=True):
        self.dataset_name = dataset_name
        self.data_folder = data_folder
        self.split = split
        self.movement_simulator = movement_simulator
        self.logger = logger
        self.return_gt = return_gt
        self.return_mask = return_mask
        self.items_names, self.items_gts_paths, self.items_masks_paths = \
            DatasetPaths.get_items(dataset_name, data_folder, split, return_gt, return_mask)
        self.items_limits = np.cumsum([
            len(item_paths) for item_paths in (
                self.items_gts_paths if self.items_gts_paths is not None else self.items_masks_paths
            )]
        )
        self._validate()

    def _validate(self):
        assert self.dataset_name in ['davis-2017', 'youtube-vos', 'got-10k']
        assert self.split in ['train', 'validation', 'test']
        assert not (self.dataset_name == 'youtube-vos' and self.return_mask and self.split != 'train')

    def __getitem__(self, frame_index):
        """Returns the frame with index ``frame_item``.

        Args:
            frame_index (int): frame index between 0 and ``self.__len__()``.

        Returns:
            torch.FloatTensor: frame quantized between [0,1] with shape (C,H,W).
            str: name of the sequence.
        """
        sequence_index = next(x[0] for x in enumerate(self.items_limits) if x[1] > frame_index)
        frame_index_bis = frame_index - (self.items_limits[sequence_index - 1] if sequence_index > 0 else 0)
        y = self._get_item_background(sequence_index, frame_index_bis) if self.return_gt else None
        m = self._get_item_mask(sequence_index, frame_index_bis) if self.return_mask else None
        return y, m, self.items_names[sequence_index]

    def _get_item_background(self, sequence_index, frame_index_bis):
        try:
            bg_np = jpeg.JPEG(self.items_gts_paths[sequence_index][frame_index_bis]).decode() / 255
            return torch.from_numpy(bg_np).permute(2, 0, 1)
        except Exception as e:
            self.logger.warning(
                'Exception with file {} - {}'.format(self.items_gts_paths[sequence_index][frame_index_bis], str(e))
            )
            random_frame_index_bix = random.randint(0, len(self.items_gts_paths[sequence_index]) - 1)
            return self._get_item_background(sequence_index, random_frame_index_bix)

    def _get_item_mask(self, sequence_index, frame_index_bis):
        try:
            mask_np = cv2.imread(self.items_masks_paths[sequence_index][frame_index_bis], cv2.IMREAD_COLOR) / 255
            return torch.from_numpy(mask_np).permute(2, 0, 1)
        except Exception as e:
            print('Exception with file {} - {}'.format(self.items_masks_paths[sequence_index][frame_index_bis], str(e)))
            random_frame_index_bix = random.randint(0, len(self.items_masks_paths[sequence_index]) - 1)
            return self._get_item_mask(sequence_index, random_frame_index_bix)

    def get_items(self, frames_indexes):
        y, m = None, None
        for i in range(len(frames_indexes)):
            gt, mask, _ = self.__getitem__(frames_indexes[i])
            # Check GT
            if self.return_gt and y is None:
                y = torch.zeros((gt.shape[0], len(frames_indexes), gt.shape[1], gt.shape[2]), dtype=torch.float32)
                y[:, i] = gt
            elif self.return_gt:
                y[:, i] = gt
            # Check mask
            if self.return_mask and m is None:
                m = torch.zeros((mask.shape[0], len(frames_indexes), mask.shape[1], mask.shape[2]), dtype=torch.float32)
                m[:, i] = mask
            elif self.return_mask:
                m[:, i] = mask
        return y, m

    def __len__(self):
        """Returns the sum of the frames of all sequences."""
        return self.items_limits[-1]

    def get_sequence(self, sequence_index):
        """Returns the sequence with index ``sequence_item``.

        Args:
            sequence_index (int): sequence index between 0 and ``self.len_sequences()``.

        Returns:
            torch.FloatTensor: sequence quantized between [0,1] with shape (C,F,H,W).
            str: name of the sequence.
        """
        sequence_first_frame_index = self.items_limits[sequence_index - 1] if sequence_index > 0 else 0
        sequence_last_frame_index = self.items_limits[sequence_index] - 1
        frames_indexes = list(range(sequence_first_frame_index, sequence_last_frame_index + 1))
        y, m = self.get_items(frames_indexes)
        return y, m, (self.items_names[sequence_index], 0)

    def len_sequences(self):
        """Return the number of different sequences."""
        return len(self.items_names)

    def get_patch(self, frame_index, frames_n, frames_spacing):
        if self.movement_simulator is None:
            return self._get_patch_contiguous(frame_index, frames_n, frames_spacing)
        else:
            return self._get_patch_simulated(frame_index, frames_n)

    def get_patch_random(self, frames_n, frames_spacing):
        return self.get_patch(random.randint(0, self.__len__() - 1), frames_n, frames_spacing)

    def _get_patch_contiguous(self, frame_index, frames_n, frames_spacing):
        sequence_item = next(x[0] for x in enumerate(self.items_limits) if x[1] > frame_index)
        sequence_first_frame_index = self.items_limits[sequence_item - 1] if sequence_item > 0 else 0
        sequence_last_frame_index = self.items_limits[sequence_item] - 1
        frames_indexes = list(range(
            frame_index - (frames_n // 2) * frames_spacing,
            frame_index + (frames_n // 2) * frames_spacing + 1,
            frames_spacing
        ))
        frames_indexes = np.clip(frames_indexes, sequence_first_frame_index, sequence_last_frame_index)
        y, m = self.get_items(frames_indexes)
        return y, m, (self.items_names[sequence_item], 0)

    def _get_patch_simulated(self, frame_index, frames_n):
        y, m, info = self.__getitem__(frame_index)
        transformation_matrices = None
        if y is not None:
            y, transformation_matrices = self.movement_simulator.simulate_movement(y, frames_n, transformation_matrices)
        if m is not None:
            m, transformation_matrices = self.movement_simulator.simulate_movement(m, frames_n, transformation_matrices)
        return y, m, (info, transformation_matrices)
