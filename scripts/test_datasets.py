from datasets.content_provider import ContentProvider
from utils.movement import MovementSimulator
from datasets.masked_sequence_dataset import MaskedSequenceDataset
from utils.paths import DatasetPaths
import random
import matplotlib.pyplot as plt
import torch.utils.data
import time
import torchvision.datasets

dataset_name = 'got-10k'
data_folder = '/Users/DavidAlvarezDLT/Data'
split = 'train'
movemenet_simulator = MovementSimulator(max_rotation=0, max_scaling=0.1, max_displacement=20)

background_dataset = ContentProvider('got-10k', data_folder, split, None, return_mask=False)
mask_dataset = ContentProvider('youtube-vos', data_folder, split, None, return_gt=False)
mix_dataset = MaskedSequenceDataset(background_dataset, mask_dataset, (256, 256), 5, 2, force_resize=False)


transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_dataset = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transforms)
loader = torch.utils.data.DataLoader(
    dataset=mix_dataset,
    batch_size=1
)

batch_time = None
for i, f in enumerate(loader):
    if batch_time is not None:
        print('batch time: {}'.format(time.time() - batch_time))
    batch_time = time.time()
    if i > 0:
        break