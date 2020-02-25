import skeltorch
import utils
from datasets.sequence_dataset import SequencesDataset
from datasets.masks_dataset import MasksDataset
import os.path
import random
import torch.utils.data


class CopyPasteData(skeltorch.Data):
    dataset_paths = {'davis-2017': 'DAVIS-2017', 'got-10k': 'GOT10k', 'youtube-vos': 'YouTubeVOS', 'coco': 'CoCo'}

    def create(self, data_path):
        pass

    def load_datasets(self, data_path):
        masks_dataset = MasksDataset(
            dataset_name=self.configuration.get('data', 'masks_dataset'),
            dataset_folder=os.path.join(data_path, self.dataset_paths[self.configuration.get('data', 'masks_dataset')]),
            split='train',
            emulator=utils.MovementSimulator()
        )
        self.datasets['train'] = SequencesDataset(
            dataset_name=self.configuration.get('data', 'train_dataset'),
            dataset_folder=os.path.join(data_path, self.dataset_paths[self.configuration.get('data', 'train_dataset')]),
            split='train',
            image_size=tuple(self.configuration.get('data', 'train_size')),
            frames_n=self.configuration.get('data', 'frames_n'),
            frames_spacing=self.configuration.get('data', 'frames_spacing'),
            dilatation_filter_size=tuple(self.configuration.get('data', 'dilatation_filter_size')),
            dilatation_iterations=self.configuration.get('data', 'dilatation_iterations'),
            logger=self.logger,
            masks_dataset=masks_dataset
        )
        self.datasets['validation'] = SequencesDataset(
            dataset_name=self.configuration.get('data', 'train_dataset'),
            dataset_folder=os.path.join(data_path, self.dataset_paths[self.configuration.get('data', 'train_dataset')]),
            split='validation',
            image_size=tuple(self.configuration.get('data', 'train_size')),
            frames_n=self.configuration.get('data', 'frames_n'),
            frames_spacing=self.configuration.get('data', 'frames_spacing'),
            dilatation_filter_size=tuple(self.configuration.get('data', 'dilatation_filter_size')),
            dilatation_iterations=self.configuration.get('data', 'dilatation_iterations'),
            logger=self.logger,
            masks_dataset=masks_dataset
        )
        self.datasets['test'] = SequencesDataset(
            dataset_name=self.configuration.get('data', 'test_dataset'),
            dataset_folder=os.path.join(data_path, self.dataset_paths[self.configuration.get('data', 'test_dataset')]),
            split='test',
            image_size=tuple(self.configuration.get('data', 'test_size')),
            frames_n=-1,
            frames_spacing=None,
            dilatation_filter_size=tuple(self.configuration.get('data', 'dilatation_filter_size')),
            dilatation_iterations=self.configuration.get('data', 'dilatation_iterations'),
            logger=self.logger,
            masks_dataset=None
        )

    def load_loaders(self, data_path, num_workers):
        self.regenerate_loaders(num_workers)

    def regenerate_loaders(self, num_workers):
        train_max_items = self.configuration.get('training', 'train_max_iterations') * \
                          self.configuration.get('training', 'batch_size')
        validation_max_items = self.configuration.get('training', 'validation_max_iterations') * \
                               self.configuration.get('training', 'batch_size')
        train_indexes = random.sample(list(range(len(self.datasets['train']))), train_max_items)
        validation_indexes = random.sample(list(range(len(self.datasets['validation']))), validation_max_items)
        self.loaders['train'] = torch.utils.data.DataLoader(
            dataset=self.datasets['train'],
            sampler=torch.utils.data.SubsetRandomSampler(indices=train_indexes),
            batch_size=self.configuration.get('training', 'batch_size'),
            num_workers=num_workers
        )
        self.loaders['validation'] = torch.utils.data.DataLoader(
            dataset=self.datasets['validation'],
            sampler=torch.utils.data.SubsetRandomSampler(indices=validation_indexes),
            batch_size=self.configuration.get('training', 'batch_size'),
            num_workers=num_workers
        )
        self.loaders['test'] = torch.utils.data.DataLoader(
            dataset=self.datasets['test'],
            batch_size=1,
            num_workers=num_workers
        )
