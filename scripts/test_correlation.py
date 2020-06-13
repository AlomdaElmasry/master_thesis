import argparse
import utils.paths
import torch.utils.data
from thesis.data import ContentProvider, MaskedSequenceDataset
import matplotlib.pyplot as plt
import torch
import models.vgg_16
import models.corr
import numpy as np
import random
import matplotlib.patches as patches
import utils.flow

parser = argparse.ArgumentParser(description='Visualize samples from the dataset')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
args = parser.parse_args()


# Create a function to plot an image with grids
def plot_with_grid(x, h_pos, w_pos, size=256):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, size, size // 16))
    ax.set_yticks(np.arange(0, size, size // 16))
    plt.imshow(x)
    rect = patches.Rectangle((w_pos * 16, h_pos * 16), 16, 16, linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.grid()
    plt.show()


def plot_horiz(plots_list):
    for i, plots_item in enumerate(plots_list):
        x1, x2 = plots_item
        x2 = torch.nn.functional.interpolate(x2.view(1, 1, x2.size(0), x2.size(1)), size=(256, 256))
        plt.subplot(len(plots_list), 2, 2*i + 1)
        plt.imshow(x1)
        plt.subplot(len(plots_list), 2, 2*i + 2)
        plt.imshow(x2.view(256, 256))
    plt.show()


# Seed seeds
torch.manual_seed(9)
np.random.seed(15)
random.seed(15)

# Get meta
gts_meta = utils.paths.DatasetPaths.get_items('got-10k', args.data_path, 'train', return_masks=False)
masks_meta = utils.paths.DatasetPaths.get_items('youtube-vos', args.data_path, 'train', return_gts=False)

# Create ContentProvider objects
gts_dataset = ContentProvider(args.data_path, gts_meta, None)
masks_dataset = ContentProvider(args.data_path, masks_meta, None)

# Create MaskedSequenceDataset object
dataset = MaskedSequenceDataset(
    gts_dataset=gts_dataset,
    masks_dataset=masks_dataset,
    gts_simulator=None,
    masks_simulator=None,
    image_size=[256, 256],
    frames_n=5,
    frames_spacing=2,
    frames_randomize=False,
    dilatation_filter_size=(3, 3),
    dilatation_iterations=4,
    force_resize=False,
    keep_ratio=True
)

# Created Loader object
loader = iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True))

# Get first sample
(x, m), y, info = next(loader)

# Get correlation volume
corr = models.corr.CorrelationVGG('cpu')
t, r_list = 2, [0, 1, 3, 4]
with torch.no_grad():
    x_corr_vol = corr(y, m, t, r_list).detach()

# Transform corr to dense flow
corr_flow = utils.flow.corr_to_flow(x_corr_vol)
corr_flow_relative = utils.flow.flow_abs_to_relative(corr_flow)

# Ask for position to plot. x_corr_vol is (b, t, h, w, h, w)
h_pos, w_pos = 11, 0

# Plot the target frame with a square in the pos
plot_with_grid(x[0, :, 2].permute(1, 2, 0), h_pos, w_pos)

# Plot map for each t
plots_list = []
for t, ref_index in enumerate(r_list):
    plots_list.append(
        (x[0, :, ref_index].permute(1, 2, 0), x_corr_vol[0, t, h_pos, w_pos])
    )
plot_horiz(plots_list)