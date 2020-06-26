import utils.paths
import argparse
import thesis.data
import matplotlib.pyplot as plt
import models.thesis_alignment
import models.vgg_16
import torch
import thesis_alignment.runner

parser = argparse.ArgumentParser(description='Visualize the inpainting process')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--device', default='cpu', help='Device to use')
args = parser.parse_args()

# Prepare the dataset with the data
test_meta = utils.paths.DatasetPaths.get_items('davis-2017', args.data_path, split='train')
test_gts_dataset = thesis.data.ContentProvider(args.data_path, test_meta, None)
test_dataset = thesis.data.MaskedSequenceDataset(
    gts_dataset=test_gts_dataset,
    masks_dataset=None,
    gts_simulator=None,
    masks_simulator=None,
    image_size=(240, 480),
    frames_n=-1,
    frames_spacing=1,
    frames_randomize=False,
    dilatation_filter_size=(3, 3),
    dilatation_iterations=4,
    force_resize=True,
    keep_ratio=False
)

# Load models
model_vgg = models.vgg_16.get_pretrained_model(args.device)
model_alignment = models.thesis_alignment.ThesisAlignmentModel(model_vgg).to(args.device)

# Load model states
model_alignment_path = '/Users/DavidAlvarezDLT/Documents/PyCharm/master_thesis/experiments/test/checkpoints/45.checkpoint.pkl'
with open(model_alignment_path, 'rb') as checkpoint_file:
    model_alignment.load_state_dict(torch.load(checkpoint_file, map_location=args.device)['model'])

# Select a random sample
(x, m), y, info = test_dataset[3]
frame_to_fill = x.size(1) // 2

# Iterate until the frame is filled
for d in range(2, 8, 2):
    t, r_list = frame_to_fill, [frame_to_fill + d]
    x_aligned, v_aligned = thesis_alignment.runner.ThesisAlignmentRunner.infer_step_propagate(
        model_alignment, x.unsqueeze(0), m.unsqueeze(0), t, r_list
    )

    plt.imshow(x[:, frame_to_fill].permute(1, 2, 0))
    plt.show()
    plt.imshow(x[:, frame_to_fill + d].permute(1, 2, 0))
    plt.show()
    a = 1
