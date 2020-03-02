import skeltorch
from datasets.content_provider import ContentProvider
from datasets.masked_sequence_dataset import MaskedSequenceDataset
import random
import torch.utils.data


class CopyPasteData(skeltorch.Data):
    dataset_paths = {'davis-2017': 'DAVIS-2017', 'got-10k': 'GOT10k', 'youtube-vos': 'YouTubeVOS', 'coco': 'CoCo'}

    def create(self, data_path):
        pass

    def load_datasets(self, data_path):
        train_gts_dataset, validation_gts_dataset, test_gts_dataset = self._load_datasets_gts(data_path)
        train_masks_dataset, validation_masks_dataset, test_masks_dataset = self._load_datasets_masks(data_path)
        self.datasets['train'] = MaskedSequenceDataset(
            gts_dataset=train_gts_dataset,
            masks_dataset=train_masks_dataset,
            image_size=tuple(self.experiment.configuration.get('data', 'train_size')),
            frames_n=self.experiment.configuration.get('data', 'frames_n'),
            frames_spacing=self.experiment.configuration.get('data', 'frames_spacing')
        )
        self.datasets['validation'] = MaskedSequenceDataset(
            gts_dataset=validation_gts_dataset,
            masks_dataset=validation_masks_dataset,
            image_size=tuple(self.experiment.configuration.get('data', 'train_size')),
            frames_n=self.experiment.configuration.get('data', 'frames_n'),
            frames_spacing=self.experiment.configuration.get('data', 'frames_spacing')
        )
        self.datasets['test'] = MaskedSequenceDataset(
            gts_dataset=test_gts_dataset,
            masks_dataset=test_gts_dataset,
            image_size=tuple(self.experiment.configuration.get('data', 'test_size')),
            frames_n=-1,
            frames_spacing=None,
            force_resize=True
        )

    def _load_datasets_gts(self, data_path):
        train_gts_dataset = ContentProvider(
            dataset_name=self.experiment.configuration.get('data', 'train_dataset'),
            data_folder=data_path,
            split='train',
            movement_simulator=None,
            return_mask=False
        )
        validation_gts_dataset = ContentProvider(
            dataset_name=self.experiment.configuration.get('data', 'train_dataset'),
            data_folder=data_path,
            split='validation',
            movement_simulator=None,
            return_mask=False
        )
        test_gts_dataset = ContentProvider(
            dataset_name=self.experiment.configuration.get('data', 'test_dataset'),
            data_folder=data_path,
            split='validation',
            movement_simulator=None
        )
        return train_gts_dataset, validation_gts_dataset, test_gts_dataset

    def _load_datasets_masks(self, data_path):
        train_masks_dataset = ContentProvider(
            dataset_name=self.experiment.configuration.get('data', 'masks_dataset'),
            data_folder=data_path,
            split='train',
            movement_simulator=None,
            return_gt=False
        )
        validation_masks_dataset = ContentProvider(
            dataset_name=self.experiment.configuration.get('data', 'masks_dataset'),
            data_folder=data_path,
            split='validation',
            movement_simulator=None,
            return_gt=False
        )
        test_masks_dataset = None
        return train_masks_dataset, validation_masks_dataset, test_masks_dataset

    def load_loaders(self, data_path, num_workers):
        self.regenerate_loaders(num_workers)

    def regenerate_loaders(self, num_workers):
        train_max_items = self.experiment.configuration.get('training', 'train_max_iterations') * \
                          self.experiment.configuration.get('training', 'batch_size')

        validation_max_items = self.experiment.configuration.get('training', 'validation_max_iterations') * \
                               self.experiment.configuration.get('training', 'batch_size')

        train_indexes = random.sample(list(range(len(self.datasets['train']))), train_max_items)
        validation_indexes = random.sample(list(range(len(self.datasets['validation']))), validation_max_items)

        # Load in RAM indexes
        self.datasets['train'].load_items_in_ram(train_indexes)
        # self.datasets['validation'].load_items_in_ram(validation_indexes)

        self.loaders['train'] = torch.utils.data.DataLoader(
            dataset=self.datasets['train'],
            sampler=torch.utils.data.SubsetRandomSampler(indices=train_indexes),
            batch_size=self.experiment.configuration.get('training', 'batch_size'),
            num_workers=num_workers
        )
        self.loaders['validation'] = torch.utils.data.DataLoader(
            dataset=self.datasets['validation'],
            sampler=torch.utils.data.SubsetRandomSampler(indices=validation_indexes),
            batch_size=self.experiment.configuration.get('training', 'batch_size'),
            num_workers=num_workers
        )
        self.loaders['test'] = torch.utils.data.DataLoader(
            dataset=self.datasets['test'],
            batch_size=1,
            num_workers=num_workers
        )
