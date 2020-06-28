import utils.paths
import argparse
import thesis.data
import matplotlib.pyplot as plt
import models.thesis_alignment
import models.thesis_inpainting
import models.vgg_16
import torch
import thesis_alignment.runner
import thesis_inpainting.runner
import utils.framing

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
    image_size=(256, 512),
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
model_inpainting = models.thesis_inpainting.ThesisInpaintingVisible().to(args.device)

# Load model states
model_alignment_path = '/Users/DavidAlvarezDLT/Documents/PyCharm/master_thesis/experiments/test/checkpoints/45.checkpoint.pkl'
model_inpainting_path = '/Users/DavidAlvarezDLT/Documents/PyCharm/master_thesis/experiments/test_inp/checkpoints/54.checkpoint.pkl'
with open(model_alignment_path, 'rb') as checkpoint_file:
    model_alignment.load_state_dict(torch.load(checkpoint_file, map_location=args.device)['model'])
with open(model_inpainting_path, 'rb') as checkpoint_file:
    model_inpainting.load_state_dict(torch.load(checkpoint_file, map_location=args.device)['model'])

# Select a random sample
(x, m), y, info = test_dataset[53]
fill_color = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)

# Initialize variables to fill in the database
t = x.size(1) // 2
x_t, m_t, y_t = x[:, t].unsqueeze(0), m[:, t].unsqueeze(0), y[:, t].unsqueeze(0)

# Iterate until the frame is filled
inpainted_frame = [x_t[0].permute(1, 2, 0) * 255]
for d in range(2, 20, 2):
    # Define the target and the r_list of the current iteration
    r_list = [t + d]

    # Align the reference frame with respect to the target frame
    x_aligned, v_aligned = thesis_alignment.runner.ThesisAlignmentRunner.infer_step_propagate(
        model_alignment, x_t, m_t, x[:, r_list].unsqueeze(0), m[:, r_list].unsqueeze(0)
    )

    # Inpaint the target frame using the reference frame
    y_hat, y_hat_comp, v_map = thesis_inpainting.runner.ThesisInpaintingRunner.infer_step_propagate(
        model_inpainting, x_t, (1 - m_t), y_t, x_aligned, v_aligned
    )

    # Update values for the next iteration
    m_t = m_t - v_map[:, :, 0]
    x_t = (1 - m_t) * y_hat[:, :, 0] + (m_t.repeat(1, 3, 1, 1) * fill_color)

    # Include the frame in the list
    inpainted_frame.append(x_t[0].permute(1, 2, 0) * 255)

    # Plot current result
    plt.imshow(y_hat[0, :, 0].permute(1, 2, 0))
    plt.show()

# Save the video
framing = utils.framing.FramesToVideo(0, 1, None)
framing.add_sequence(torch.stack(inpainted_frame, dim=0).numpy())
framing.save('.', 'test')
