from datasets.content_provider import ContentProvider
from utils.movement import MovementSimulator
from datasets.synthetic_dataset import MaskedSequenceDataset
from utils.paths import DatasetPaths
import random
import matplotlib.pyplot as plt

dataset_name = 'got-10k'
data_folder = '/Users/DavidAlvarezDLT/Data'
split = 'train'
movemenet_simulator = MovementSimulator()

background_dataset = ContentProvider('got-10k', data_folder, split, 'objects', movemenet_simulator, return_mask=False)
mask_dataset = ContentProvider('davis-2017', data_folder, split, 'objects', movemenet_simulator, return_gt=False)

rand_item = random.randint(0, background_dataset.len_sequences())


mix_dataset = MaskedSequenceDataset(background_dataset, mask_dataset, (256, 256), 5, 2, force_resize=True)
len_dataset = len(mix_dataset)

print('{}/{}'.format(rand_item, len_dataset))

r = random.randint(0, len(mix_dataset) - 1)
(x, m), y, info = mix_dataset.__getitem__(r)

for f in range(x.size(1)):
    f_image = x[:, f].permute(1, 2, 0)
    plt.imshow(f_image.numpy())
    plt.show()

a = 1
