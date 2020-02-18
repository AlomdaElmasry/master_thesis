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

    def create(self):
        pass

    def load(self, data_file_path: str):
        super().load(data_file_path)
        self.load_datasets()
        self.load_loaders()

    def _get_transforms(self):
        frame_transforms = torchvision.transforms.Compose([
            Resize(size=(240, 480), method=cv2.INTER_LINEAR)
        ])
        mask_transforms = torchvision.transforms.Compose([
            Binarize(),
            Resize(size=(240, 480), method=cv2.INTER_NEAREST),
            Dilatate(filter_size=(3, 3), iterations=4)
        ])
        return frame_transforms, mask_transforms

    def load_datasets(self):
        frame_transforms, mask_transforms = self._get_transforms()
        if self.configuration.get('data', 'dataset') == 'davis-2017':
            train_dataset = FramedDataset(
                dataset_folder=os.path.join(self.execution.args['data_path'], 'DAVIS-2017'),
                split='train',
                n_frames=5,
                gt_transforms=frame_transforms,
                mask_transforms=mask_transforms,
                device=self.execution.device
            )
            self.datasets = {'train': MaskedDataset(train_dataset, None)}

    def load_loaders(self):
        self.loaders = {'train': torch.utils.data.DataLoader(self.datasets['train'], batch_size=8)}
