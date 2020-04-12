import torch.utils.data
import numpy as np
import random
import jpeg4py as jpeg
import cv2
import os.path
import utils.transforms


class ContentProvider(torch.utils.data.Dataset):
    data_folder = None
    dataset_meta = None
    movement_simulator = None
    logger = None
    items_names = None
    items_limits = None
    _ram_data = None

    def __init__(self, data_folder, dataset_meta, movement_simulator, logger, load_in_ram=False):
        self.data_folder = data_folder
        self.dataset_meta = dataset_meta
        self.movement_simulator = movement_simulator
        self.logger = logger
        self.items_names = list(self.dataset_meta.keys())
        self.items_limits = np.cumsum([
            len(self.dataset_meta[item_name][0]) if self.dataset_meta[item_name][0] is not None
            else len(self.dataset_meta[item_name][1]) for item_name in self.items_names
        ])
        if load_in_ram:
            self._load_data_in_ram()

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
        item_name = self.items_names[sequence_index]
        if self.dataset_meta[item_name][0] is None:
            return None
        if self._ram_data is not None and self._ram_data[item_name][0][frame_index_bis] is not None:
            return self._ram_data[item_name][0][frame_index_bis]
        item_path = os.path.join(self.data_folder, self.dataset_meta[item_name][0][frame_index_bis])
        return torch.from_numpy(jpeg.JPEG(item_path).decode() / 255).permute(2, 0, 1).float()

    def _get_item_mask(self, sequence_index, frame_index_bis):
        item_name = self.items_names[sequence_index]
        if self.dataset_meta[item_name][1] is None:
            return None
        if self._ram_data is not None and self._ram_data[item_name][1][frame_index_bis] is not None:
            return self._ram_data[item_name][1][frame_index_bis]
        item_path = os.path.join(self.data_folder, self.dataset_meta[item_name][1][frame_index_bis])
        return torch.from_numpy(cv2.imread(item_path, cv2.IMREAD_GRAYSCALE) / 255 > 0).float()

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

    def _load_data_in_ram(self):
        self._ram_data = {}
        for dataset_item_key in self.dataset_meta.keys():
            self._ram_data[dataset_item_key] = [None, None]
            if self.dataset_meta[dataset_item_key][0] is not None:
                self._ram_data[dataset_item_key][0] = [None] * len(self.dataset_meta[dataset_item_key][0])
                self._load_data_in_ram_background(dataset_item_key)
            if self.dataset_meta[dataset_item_key][1] is not None:
                self._ram_data[dataset_item_key][1] = [None] * len(self.dataset_meta[dataset_item_key][1])
                self._load_data_in_ram_masks(dataset_item_key)

    def _load_data_in_ram_background(self, dataset_item_key):
        for i, item_path in enumerate(self.dataset_meta[dataset_item_key][1]):
            self._ram_data[dataset_item_key][0][i] = self._get_item_background(
                self.items_names.index(dataset_item_key), i
            )

    def _load_data_in_ram_masks(self, dataset_item_key):
        for i, item_path in enumerate(self.dataset_meta[dataset_item_key][1]):
            self._ram_data[dataset_item_key][1][i] = self._get_item_mask(self.items_names.index(dataset_item_key), i)


class MaskedSequenceDataset(torch.utils.data.Dataset):
    gts_dataset = None
    masks_dataset = None
    image_size = None
    frames_n = None
    frames_spacing = None
    dilatation_filter_size = None
    dilatation_iterations = None
    force_resize = None
    keep_ratio = None
    fill_color = None

    def __init__(self, gts_dataset, masks_dataset, image_size, frames_n, frames_spacing, dilatation_filter_size,
                 dilatation_iterations, force_resize, keep_ratio):
        self.gts_dataset = gts_dataset
        self.masks_dataset = masks_dataset
        self.image_size = image_size
        self.frames_n = frames_n
        self.frames_spacing = frames_spacing
        self.force_resize = force_resize
        self.keep_ratio = keep_ratio
        self.dilatation_filter_size = dilatation_filter_size
        self.dilatation_iterations = dilatation_iterations
        self.fill_color = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32)

    def __getitem__(self, item):
        # Get the data associated to the GT
        y, m, info = self.gts_dataset.get_sequence(item) if self.frames_n == -1 \
            else self.gts_dataset.get_patch(item, self.frames_n, self.frames_spacing)

        # If self.gts_dataset and self.masks_dataset are not the same, obtain new mask
        if self.masks_dataset is not None:
            masks_n = self.frames_n if self.frames_n != -1 else y.size(1)
            _, m, _ = self.masks_dataset.get_patch_random(masks_n, self.frames_spacing)

        # Apply GT transformations
        if self.force_resize:
            y = utils.transforms.ImageTransforms.resize(y, self.image_size, keep_ratio=self.keep_ratio)
        else:
            y, _ = utils.transforms.ImageTransforms.crop(y, self.image_size)

        # Apply Mask transformations
        if self.image_size != (m.size(2), m.size(3)):
            m = utils.transforms.ImageTransforms.resize(m, self.image_size, mode='nearest', keep_ratio=self.keep_ratio)
        m = utils.transforms.ImageTransforms.dilatate(m, self.dilatation_filter_size, self.dilatation_iterations)

        # Compute x
        x = (1 - m) * y + (m.permute(3, 2, 1, 0) * self.fill_color).permute(3, 2, 1, 0)
        return (x, m), y, info

    def __len__(self):
        return self.gts_dataset.len_sequences() if self.frames_n == -1 else len(self.gts_dataset)