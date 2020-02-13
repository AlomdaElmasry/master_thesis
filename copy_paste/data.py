import skeltorch
from skeltorch import Configuration
from datasets.davis_2017 import DAVIS2017Dataset
from datasets.masked_dataset import MaskedDataset
import os.path
import torch.utils.data
import torchvision


class CopyPasteData(skeltorch.Data):
    image_size = (240, 424)

    def __init__(self):
        super().__init__()
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_size)
        ])

    def create(self, data_folder_path: str, conf: skeltorch.Configuration):
        pass

    def load_datasets(self, data_path: str, conf: Configuration):
        if conf.get('data', 'dataset') == 'davis-2017':
            train_dataset = DAVIS2017Dataset(
                'train', os.path.join(data_path, 'DAVIS-2017'), self.image_size
            )
            self.datasets['train'] = MaskedDataset(train_dataset, None)

    def load_loaders(self, data_path: str, conf: Configuration):
        self.loaders['train'] = torch.utils.data.DataLoader(self.datasets['train'], batch_size=1, num_workers=2)
