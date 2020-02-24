import skeltorch
from datasets.sequence_dataset import SequencesDataset
from datasets.masks_dataset import MasksDataset
from datasets.transforms import Resize, Dilatate, Binarize
import os.path
import torch.utils.data
import torchvision
import cv2


class CopyPasteData(skeltorch.Data):
    frame_transforms: torchvision.transforms.Compose
    mask_transforms: torchvision.transforms.Compose

    def create(self, data_path):
        pass

    def load_datasets(self, data_path, device):
        self._load_transforms()
        masks_dataset = MasksDataset(
            dataset_folder=self._get_dataset_path(data_path, self.configuration.get('data', 'train_dataset_masks'))
        )
        self.datasets['train'] = SequencesDataset(
            dataset_name=self.configuration.get('data', 'train_dataset'),
            dataset_folder=self._get_dataset_path(data_path, self.configuration.get('data', 'train_dataset')),
            split='train',
            logger=self.logger,
            n_frames=5,
            masks_dataset=masks_dataset,
            gt_transforms=self.frame_transforms,
            mask_transforms=self.mask_transforms,
            device=device
        )
        self.datasets['validation'] = SequencesDataset(
            dataset_name=self.configuration.get('data', 'train_dataset'),
            dataset_folder=self._get_dataset_path(data_path, self.configuration.get('data', 'train_dataset')),
            split='train',
            logger=self.logger,
            n_frames=5,
            masks_dataset=masks_dataset,
            gt_transforms=self.frame_transforms,
            mask_transforms=self.mask_transforms,
            device=device
        )
        self.datasets['test'] = SequencesDataset(
            dataset_name=self.configuration.get('data', 'test_dataset'),
            dataset_folder=self._get_dataset_path(data_path, self.configuration.get('data', 'test_dataset')),
            split='train',
            logger=self.logger,
            n_frames=-1,
            masks_dataset=None,
            gt_transforms=self.frame_transforms,
            mask_transforms=self.mask_transforms,
            device=device
        )

    def load_loaders(self, data_path, device):
        self.loaders['train'] = torch.utils.data.DataLoader(
            self.datasets['train'], shuffle=True, batch_size=self.configuration.get('training', 'batch_size')
        )
        self.loaders['validation'] = torch.utils.data.DataLoader(
            self.datasets['validation'], shuffle=True, batch_size=self.configuration.get('training', 'batch_size')
        )
        self.loaders['test'] = torch.utils.data.DataLoader(
            self.datasets['test'], batch_size=1
        )

    def _load_transforms(self):
        self.frame_transforms = torchvision.transforms.Compose([Resize(size=(240, 480), method=cv2.INTER_LINEAR)])
        self.mask_transforms = torchvision.transforms.Compose([
            Binarize(), Resize(size=(240, 480), method=cv2.INTER_NEAREST), Dilatate(filter_size=(3, 3), iterations=4)
        ])

    def _get_dataset_path(self, data_path, dataset_name):
        if dataset_name == 'davis-2017':
            return os.path.join(data_path, 'DAVIS-2017')
        elif dataset_name == 'youtube-vos':
            return os.path.join(data_path, 'YouTubeVOS')
        elif dataset_name == 'coco':
            return os.path.join(data_path, 'CoCo')
        else:
            return None
