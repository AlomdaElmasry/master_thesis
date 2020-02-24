import skeltorch
from datasets.sequence_dataset import SequencesDataset
from datasets.masks_dataset import MasksDataset
import os.path
import torch.utils.data


class CopyPasteData(skeltorch.Data):
    dataset_paths = {'davis-2017': 'DAVIS-2017', 'youtube-vos': 'YouTubeVOS', 'coco': 'CoCo'}

    def create(self, data_path):
        pass

    def load_datasets(self, data_path, device):
        # masks_dataset = MasksDataset(
        #     dataset_folder=os.path.join(data_path, self.dataset_paths[self.configuration.get('data', 'masks_dataset')]),
        #     device=device
        # )
        masks_dataset = None
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
            masks_dataset=masks_dataset,
            device=device
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
            masks_dataset=masks_dataset,
            device=device
        )
        self.datasets['test'] = SequencesDataset(
            dataset_name=self.configuration.get('data', 'test_dataset'),
            dataset_folder=os.path.join(data_path, self.dataset_paths[self.configuration.get('data', 'test_dataset')]),
            split='train',
            image_size=tuple(self.configuration.get('data', 'test_size')),
            frames_n=-1,
            frames_spacing=None,
            dilatation_filter_size=tuple(self.configuration.get('data', 'dilatation_filter_size')),
            dilatation_iterations=self.configuration.get('data', 'dilatation_iterations'),
            logger=self.logger,
            masks_dataset=None,
            device=device
        )

    def load_loaders(self, data_path, device):
        self.loaders['train'] = torch.utils.data.DataLoader(
            self.datasets['train'], shuffle=True, batch_size=self.configuration.get('training', 'batch_size')
        )
        self.loaders['validation'] = torch.utils.data.DataLoader(
            self.datasets['validation'], shuffle=True, batch_size=self.configuration.get('training', 'batch_size')
        )
        self.loaders['test'] = torch.utils.data.DataLoader(self.datasets['test'], batch_size=1)
