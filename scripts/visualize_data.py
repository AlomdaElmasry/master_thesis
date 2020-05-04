import argparse
from thesis.data import ContentProvider, MaskedSequenceDataset
import torch.utils.data
import matplotlib.pyplot as plt
import utils.movement
import torch.nn.functional as F
import utils.paths

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

# Get Meta
gts_meta = utils.paths.DatasetPaths.get_items(
    dataset_name='got-10k',
    data_folder=args.data_path,
    split='train',
    return_masks=False
)
masks_meta = utils.paths.DatasetPaths.get_items(
    dataset_name='youtube-vos',
    data_folder=args.data_path,
    split='train',
    return_gts=False
)

# Create ContentProviders
movement_null = utils.movement.MovementSimulator(0, 0, 0)
movement_simulator = utils.movement.MovementSimulator()
gts_dataset = ContentProvider(args.data_path, gts_meta, None)
masks_dataset = ContentProvider(args.data_path, masks_meta, None)

# Create ContentProviders
dataset = MaskedSequenceDataset(
    gts_dataset=gts_dataset,
    masks_dataset=masks_dataset,
    gts_simulator=movement_null,
    masks_simulator=movement_simulator,
    image_size=[256, 256],
    frames_n=5,
    frames_spacing=2,
    frames_randomize=False,
    dilatation_filter_size=(3, 3),
    dilatation_iterations=4,
    force_resize=False,
    keep_ratio=True,
    p_simulator=0.5
)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

# Iterate over the data sets
for i, data in enumerate(loader):
    (x, m), y, info = data

    # Check if it's simulated
    a = 1
    for i in range(x.size(2)):
        plt.imshow(x[1, :, i].permute(1, 2, 0))
        plt.show()
        a = 1

    # random_affines_stacked = [torch.inverse(random_affines_stacked[0, i]) for i in range(5)]
    # random_thetas_stacked = torch.stack([
    #     utils.MovementSimulator.affine2theta(ra, 256, 256) for ra in random_affines_stacked
    # ])
    # affine_grid = F.affine_grid(random_thetas_stacked, [5, 3, 256, 256])
    # data_out = F.grid_sample(x[0].transpose(0, 1), affine_grid).squeeze(0)


    exit()
