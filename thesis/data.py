import skeltorch
from .dataset import ContentProvider
from .dataset import MaskedSequenceDataset
import random
import torch.utils.data
import utils.movement
import utils.paths
import cv2
import os.path
import numpy as np


class ThesisData(skeltorch.Data):
    train_gts_meta = None
    train_masks_meta = None
    validation_gts_meta = None
    validation_masks_meta = None
    test_meta = None

    def create(self, data_path):
        self.train_gts_meta = utils.paths.DatasetPaths.get_items(
            dataset_name=self.experiment.configuration.get('data', 'train_gts_dataset'),
            data_folder=data_path,
            split='train',
            return_masks=False
        )
        self.validation_gts_meta = utils.paths.DatasetPaths.get_items(
            dataset_name=self.experiment.configuration.get('data', 'validation_gts_dataset'),
            data_folder=data_path,
            split='validation',
            return_masks=False
        )
        self.train_masks_meta = utils.paths.DatasetPaths.get_items(
            dataset_name=self.experiment.configuration.get('data', 'train_masks_dataset'),
            data_folder=data_path,
            split='train',
            return_gts=False
        )
        self.validation_masks_meta = utils.paths.DatasetPaths.get_items(
            dataset_name=self.experiment.configuration.get('data', 'validation_masks_dataset'),
            data_folder=data_path,
            split='train',
            return_gts=False
        )
        self.test_meta = utils.paths.DatasetPaths.get_items(
            dataset_name=self.experiment.configuration.get('data', 'test_dataset'),
            data_folder=data_path,
            split='train'
        )

        # Clean masks that are too big
        if self.experiment.configuration.get('data', 'max_mask_size') is not None:
            self._clean_masks(data_path, self.train_masks_meta)
            self._clean_masks(data_path, self.validation_masks_meta)

    def _clean_masks(self, data_path, masks_meta):
        train_masks_items = list(masks_meta.keys())
        for train_masks_item in train_masks_items:
            item_path = os.path.join(data_path, masks_meta[train_masks_item][1][0])
            image = cv2.imread(item_path, cv2.IMREAD_GRAYSCALE) > 0
            image_mask_size = np.count_nonzero(image) / (image.shape[0] * image.shape[1])
            if image_mask_size > self.experiment.configuration.get('data', 'max_mask_size'):
                masks_meta.pop(train_masks_item)

    def load_datasets(self, data_path):
        gts_datasets = self._load_datasets_gts(data_path)
        masks_datasets = self._load_datasets_masks(data_path)
        self.datasets['train'] = MaskedSequenceDataset(
            gts_dataset=gts_datasets[0],
            masks_dataset=masks_datasets[0],
            image_size=tuple(self.experiment.configuration.get('data', 'train_size')),
            frames_n=self.experiment.configuration.get('data', 'frames_n'),
            frames_spacing=self.experiment.configuration.get('data', 'frames_spacing'),
            dilatation_filter_size=tuple(self.experiment.configuration.get('data', 'dilatation_filter_size')),
            dilatation_iterations=self.experiment.configuration.get('data', 'dilatation_iterations'),
            force_resize=self.experiment.configuration.get('data', 'train_resize'),
            keep_ratio=True
        )
        self.datasets['validation'] = MaskedSequenceDataset(
            gts_dataset=gts_datasets[1],
            masks_dataset=masks_datasets[1],
            image_size=tuple(self.experiment.configuration.get('data', 'train_size')),
            frames_n=self.experiment.configuration.get('data', 'frames_n'),
            frames_spacing=self.experiment.configuration.get('data', 'frames_spacing'),
            dilatation_filter_size=tuple(self.experiment.configuration.get('data', 'dilatation_filter_size')),
            dilatation_iterations=self.experiment.configuration.get('data', 'dilatation_iterations'),
            force_resize=self.experiment.configuration.get('data', 'train_resize'),
            keep_ratio=True
        )
        self.datasets['test'] = MaskedSequenceDataset(
            gts_dataset=gts_datasets[2],
            masks_dataset=masks_datasets[2],
            image_size=tuple(self.experiment.configuration.get('data', 'test_size')),
            frames_n=-1,
            frames_spacing=None,
            dilatation_filter_size=tuple(self.experiment.configuration.get('data', 'dilatation_filter_size')),
            dilatation_iterations=self.experiment.configuration.get('data', 'dilatation_iterations'),
            force_resize=True,
            keep_ratio=False
        )

    def _load_datasets_gts(self, data_path):
        train_gts_dataset = ContentProvider(
            data_folder=data_path,
            dataset_meta=self.train_gts_meta,
            movement_simulator=None,
            logger=self.logger
        )
        validation_gts_dataset = ContentProvider(
            data_folder=data_path,
            dataset_meta=self.validation_gts_meta,
            movement_simulator=None,
            logger=self.logger,
        )
        test_gts_dataset = ContentProvider(
            data_folder=data_path,
            dataset_meta=self.test_meta,
            movement_simulator=None,
            logger=self.logger,
        )
        return train_gts_dataset, validation_gts_dataset, test_gts_dataset

    def _load_datasets_masks(self, data_path):
        train_masks_dataset = ContentProvider(
            dataset_meta=self.train_masks_meta,
            data_folder=data_path,
            movement_simulator=None,
            logger=self.logger,
            load_in_ram=False
        )
        validation_masks_dataset = ContentProvider(
            dataset_meta=self.validation_masks_meta,
            data_folder=data_path,
            movement_simulator=None,
            logger=self.logger,
        )
        return train_masks_dataset, validation_masks_dataset, None

    def load_loaders(self, data_path, num_workers):
        batch_size = self.experiment.configuration.get('training', 'batch_size')

        # Check that the number of items in the GT do not surpass it
        train_max_items = batch_size * self.experiment.configuration.get('training', 'train_max_iterations')
        if len(self.datasets['train']) > train_max_items:
            train_gts_indexes = random.sample(list(range(len(self.datasets['train']))), train_max_items)
        else:
            train_gts_indexes = list(range(len(self.datasets['train'])))

        validation_max_items = batch_size * self.experiment.configuration.get('training', 'validation_max_iterations')
        if len(self.datasets['validation']) > validation_max_items:
            val_gts_indexes = random.sample(list(range(len(self.datasets['validation']))), validation_max_items)
        else:
            val_gts_indexes = list(range(len(self.datasets['validation'])))

        self.loaders['train'] = torch.utils.data.DataLoader(
            dataset=self.datasets['train'],
            sampler=torch.utils.data.SubsetRandomSampler(indices=train_gts_indexes),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
        self.loaders['validation'] = torch.utils.data.DataLoader(
            dataset=self.datasets['validation'],
            sampler=torch.utils.data.SubsetRandomSampler(indices=val_gts_indexes),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
        self.loaders['test'] = torch.utils.data.DataLoader(
            dataset=self.datasets['test'],
            batch_size=1,
            num_workers=num_workers,
            pin_memory=True
        )