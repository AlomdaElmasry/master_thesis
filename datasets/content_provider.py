import torch.utils.data
import numpy as np
import random
import jpeg4py as jpeg
import cv2


class ContentProvider(torch.utils.data.Dataset):
    dataset_meta = None
    dataset_folder = None
    movement_simulator = None
    logger = None
    items_names = None
    items_limits = None

    def __init__(self, dataset_meta, data_folder, movement_simulator, logger):
        self.dataset_meta = dataset_meta
        self.data_folder = data_folder
        self.movement_simulator = movement_simulator
        self.logger = logger
        self.items_names = list(self.dataset_meta.keys())
        self.items_limits = np.cumsum([
            len(self.dataset_meta[item_name][0]) if self.dataset_meta[item_name][0] is not None
            else len(self.dataset_meta[item_name][1]) for item_name in self.items_names
        ])

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
        y = self._get_item_background(sequence_index, frame_index_bis)
        m = self._get_item_mask(sequence_index, frame_index_bis)
        return y, m, self.items_names[sequence_index]

    def _get_item_background(self, sequence_index, frame_index_bis):
        if self.dataset_meta[self.items_names[sequence_index]][0] is not None:
            bg_np = jpeg.JPEG(self.dataset_meta[self.items_names[sequence_index]][0][frame_index_bis]).decode() / 255
            return torch.from_numpy(bg_np).permute(2, 0, 1).float()
        else:
            return None

    def _get_item_mask(self, sequence_index, frame_index_bis):
        if self.dataset_meta[self.items_names[sequence_index]][1] is not None:
            mask_np = cv2.imread(
                self.dataset_meta[self.items_names[sequence_index]][1][frame_index_bis], cv2.IMREAD_GRAYSCALE
            ) / 255
            return torch.from_numpy(mask_np > 0).float()
        else:
            return None

    def get_items(self, frames_indexes):
        y, m = None, None
        y0, m0, _ = self.__getitem__(frames_indexes[0])
        if y0 is not None:
            y = torch.zeros((3, len(frames_indexes), y0.size(1), y0.size(2)), dtype=torch.float32)
            y[:, 0] = y0
        if m0 is not None:
            m = torch.zeros((1, len(frames_indexes), m0.size(0), m0.size(1)), dtype=torch.float32)
            m[:, 0] = m0.unsqueeze(0)
        for i in range(1, len(frames_indexes)):
            yi, mi, _ = self.__getitem__(frames_indexes[i])
            if y is not None:
                y[:, i] = yi
            if m is not None:
                m[:, i] = mi
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
            y, transformation_matrices = self.movement_simulator.simulate_movement(
                y, frames_n, transformation_matrices
            )
        if m is not None:
            m, transformation_matrices = self.movement_simulator.simulate_movement(
                m, frames_n, transformation_matrices
            )
        return y, m, (info, transformation_matrices)
