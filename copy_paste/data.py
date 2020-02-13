import skeltorch
from skeltorch import Configuration
from datasets.davis_2017 import DAVIS2017Dataset
from datasets.masked_dataset import MaskedDataset
import os.path
import torch.utils.data
import torchvision
from datasets.transforms import Resize, Dilatate, Binarize
import cv2


class CopyPasteData(skeltorch.Data):

    def __init__(self):
        super().__init__()
        self.frame_transforms = torchvision.transforms.Compose([
            Resize(size=(240, 480), method=cv2.INTER_LINEAR)
        ])
        self.mask_transforms = torchvision.transforms.Compose([
            Binarize(),
            Resize(size=(240, 480), method=cv2.INTER_NEAREST),
            Dilatate(filter_size=(3,3), iterations=4)
        ])

    def create(self, data_folder_path: str, conf: skeltorch.Configuration):
        pass

    def load_datasets(self, data_path: str, conf: Configuration):
        if conf.get('data', 'dataset') == 'davis-2017':
            train_dataset = DAVIS2017Dataset(
                'train', os.path.join(data_path, 'DAVIS-2017'), self.frame_transforms, self.mask_transforms
            )
            self.datasets['train'] = MaskedDataset(train_dataset, None)

    def load_loaders(self, data_path: str, conf: Configuration):
        self.loaders['train'] = torch.utils.data.DataLoader(self.datasets['train'], batch_size=1, num_workers=2)
