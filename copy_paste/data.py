import skeltorch
from datasets.framed_dataset import FramedDataset
from datasets.masked_dataset import MaskedDataset
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

    def create(self):
        pass

    def load(self, data_file_path: str):
        super().load(data_file_path)
        self.load_transforms()
        self.load_datasets()
        self.load_loaders()

    def load_transforms(self):
        self.frame_transforms = torchvision.transforms.Compose([Resize(size=(240, 480), method=cv2.INTER_LINEAR)])
        self.mask_transforms = torchvision.transforms.Compose([Binarize(),
                                                               Resize(size=(240, 480), method=cv2.INTER_NEAREST),
                                                               Dilatate(filter_size=(3, 3), iterations=4)])

    def _get_dataset_path(self, dataset_name):
        if dataset_name == 'davis-2017':
            return os.path.join(self.execution.args['data_path'], 'DAVIS-2017')
        elif dataset_name == 'youtube-vos':
            return os.path.join(self.execution.args['data_path'], 'YouTubeVOS')
        else:
            return None

    def load_datasets(self):
        dataset_name = self.configuration.get('data', 'dataset')
        dataset_folder = self._get_dataset_path(dataset_name)
        n_frames = -1 if self.execution.command in ['test', 'test_alignment'] else 5
        train_dataset = FramedDataset(
            dataset_name=dataset_name,
            dataset_folder=dataset_folder,
            split='train',
            n_frames=n_frames,
            gt_transforms=self.frame_transforms,
            mask_transforms=self.mask_transforms,
            device=self.execution.device
        )
        validation_dataset = FramedDataset(
            dataset_name=dataset_name,
            dataset_folder=dataset_folder,
            split='train',
            n_frames=n_frames,
            gt_transforms=self.frame_transforms,
            mask_transforms=self.mask_transforms,
            device=self.execution.device
        )
        test_dataset = FramedDataset(
            dataset_name=dataset_name,
            dataset_folder=dataset_folder,
            split='train',
            n_frames=n_frames,
            gt_transforms=self.frame_transforms,
            mask_transforms=self.mask_transforms,
            device=self.execution.device
        )
        self.datasets = {
            'train': MaskedDataset(train_dataset, None),
            'validation': MaskedDataset(validation_dataset, None),
            'test': MaskedDataset(test_dataset, None)
        }

    def load_loaders(self):
        batch_size = 1 if self.execution.command in ['test', 'test_alignment'] else 8
        self.loaders = {
            'train': torch.utils.data.DataLoader(self.datasets['train'], batch_size=batch_size),
            'validation': torch.utils.data.DataLoader(self.datasets['validation'], batch_size=batch_size),
            'test': torch.utils.data.DataLoader(self.datasets['test'], batch_size=batch_size)
        }
