import skeltorch
from datasets.sequence_dataset import SequencesDataset
from datasets.masks_dataset import MasksDataset
from datasets.transforms import Resize, Dilatate, Binarize
import os.path
import torch.utils.data
import torchvision
import cv2


class CopyPasteData(skeltorch.Data):
    datasets: dict
    loaders: dict
    frame_transforms: torchvision.transforms.Compose
    mask_transforms: torchvision.transforms.Compose

    def create(self, data_path):
        pass

    def load_transforms(self):
        self.frame_transforms = torchvision.transforms.Compose([Resize(size=(240, 480), method=cv2.INTER_LINEAR)])
        self.mask_transforms = torchvision.transforms.Compose([
            Binarize(), Resize(size=(240, 480), method=cv2.INTER_NEAREST), Dilatate(filter_size=(3, 3), iterations=4)
        ])

    def _get_dataset_path(self, dataset_name):
        if dataset_name == 'davis-2017':
            return os.path.join(self.execution.args['data_path'], 'DAVIS-2017')
        elif dataset_name == 'youtube-vos':
            return os.path.join(self.execution.args['data_path'], 'YouTubeVOS')
        elif dataset_name == 'coco':
            return os.path.join(self.execution.args['data_path'], 'CoCo')
        else:
            return None

    def load_datasets(self, data_path, device):
        self.load_transforms()
        # masks_dataset = MasksDataset(data_folder=self._get_dataset_path('coco'))
        train_dataset = SequencesDataset(
            dataset_name=self.configuration.get('data', 'test_dataset'),
            dataset_folder=self._get_dataset_path(self.configuration.get('data', 'test_dataset')),
            split='train',
            n_frames=5,
            masks_dataset=None,
            gt_transforms=self.frame_transforms,
            mask_transforms=self.mask_transforms,
            device=device
        )
        validation_dataset = SequencesDataset(
            dataset_name=self.configuration.get('data', 'test_dataset'),
            dataset_folder=self._get_dataset_path(self.configuration.get('data', 'test_dataset')),
            split='train',
            n_frames=5,
            masks_dataset=None,
            gt_transforms=self.frame_transforms,
            mask_transforms=self.mask_transforms,
            device=device
        )
        test_dataset = SequencesDataset(
            dataset_name=self.configuration.get('data', 'test_dataset'),
            dataset_folder=self._get_dataset_path(self.configuration.get('data', 'test_dataset')),
            split='train',
            n_frames=-1,
            masks_dataset=None,
            gt_transforms=self.frame_transforms,
            mask_transforms=self.mask_transforms,
            device=device
        )
        self.datasets = {'train': train_dataset, 'validation': validation_dataset, 'test': test_dataset}

    def load_loaders(self, data_path, device):
        batch_size = 1 if self.execution.command in ['test', 'test_alignment'] else \
            self.configuration.get('training', 'batch_size')
        self.loaders = {
            'train': torch.utils.data.DataLoader(self.datasets['train'], batch_size=batch_size),
            'validation': torch.utils.data.DataLoader(self.datasets['validation'], batch_size=batch_size),
            'test': torch.utils.data.DataLoader(self.datasets['test'], batch_size=batch_size)
        }