import argparse
from thesis.data import ContentProvider, MaskedSequenceDataset
import torch.utils.data
import matplotlib.pyplot as plt
import utils.movement
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Visualize samples from the dataset')
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
movement = utils.movement.MovementSimulator()
gts_dataset = ContentProvider(args.gts_dataset, args.data_path, args.gts_split, movement, None)
masks_dataset = ContentProvider(args.masks_dataset, args.data_path, args.masks_split, None, None)
dataset = MaskedSequenceDataset(gts_dataset, masks_dataset, (args.image_size, args.image_size), args.frames_n,
                                args.frames_spacing, force_resize=False)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Iterate over the data sets
for i, data in enumerate(loader):
    (x, m), y, (info, random_affines_stacked, (h, w)) = data

    random_affines_stacked = [torch.inverse(random_affines_stacked[0, i]) for i in range(5)]
    random_thetas_stacked = torch.stack([
        utils.MovementSimulator.affine2theta(ra, 256, 256) for ra in random_affines_stacked
    ])
    affine_grid = F.affine_grid(random_thetas_stacked, [5, 3, 256, 256])
    data_out = F.grid_sample(x[0].transpose(0, 1), affine_grid).squeeze(0)

    for i in range(x.size(2)):
        plt.imshow(x[0, :, i].permute(1, 2, 0))
        plt.imshow(data_out[i].permute(1, 2, 0))
        plt.show()

    exit()
