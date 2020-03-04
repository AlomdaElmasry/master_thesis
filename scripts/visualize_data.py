import argparse
from datasets.content_provider import ContentProvider
from datasets.masked_sequence_dataset import MaskedSequenceDataset
import torch.utils.data
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Creates a video from a set of static frames')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--gts-dataset', default='got-10k', help='Name of GTs dataset')
parser.add_argument('--gts-split', default='train', help='Split of GTs dataset')
parser.add_argument('--masks-dataset', default='youtube-vos', help='Name of masks dataset')
parser.add_argument('--masks-split', default='train', help='Split of masks dataset')
parser.add_argument('--image-size', default=256, type=int, help='Size of the image')
parser.add_argument('--frames-n', default=5, type=int, help='Frame number')
parser.add_argument('--frames-spacing', default=2, type=int, help='Frame spacing')
args = parser.parse_args()

# Load data sets

gts_dataset = ContentProvider(args.gts_dataset, args.data_path, args.gts_split, None, return_mask=False)
masks_dataset = ContentProvider(args.masks_dataset, args.data_path, args.masks_split, None, return_gt=False)
dataset = MaskedSequenceDataset(gts_dataset, masks_dataset, (args.image_size, args.image_size), args.frames_n,
                                args.frames_spacing)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Iterate over the data sets
for i, data in enumerate(loader):
    (x, m), y, info = data
    for i in range(x.size(2)):
        plt.imshow(x[0, :, i].permute(1, 2, 0))
        plt.show()
    exit()
