from PIL import Image
from datasets.davis_2017 import DAVIS2017Dataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
import utils

parser = argparse.ArgumentParser(description='Creates a video from a set of static frames')
parser.add_argument('--sequence', required=True, help='Path to the folder containing the frames')
parser.add_argument('--reference-frame', type=int, default=0, help='Index of the reference frame')
parser.add_argument('--alpha', type=int, default=50, help='Transparency level')
parser.add_argument('--dest_folder', type=str, default='.', help='Path where the resulting video should be saved')
parser.add_argument('--filename', type=str, help='Force a name for the output file')
args = parser.parse_args()

# Create a FramesToVideo object
overlap_frames = utils.OverlapFrames(args.reference_frame, args.alpha)

# Add the sequence to the object
overlap_frames.add_sequence_from_path(args.sequence)

# Save the video
overlap_frames.save(args.dest_folder, args.filename)


# train_dataset = DAVIS2017Dataset(dataset_folder='/Users/DavidAlvarezDLT/Data/DAVIS-2017', split='train')
# next_video = next(iter(train_dataset))
#
# img = Image.fromarray((next_video[0][:, 0, :].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).convert('RGBA')
# for i in range(1, next_video[0].size(1), 10):
#     aux_frame = Image.fromarray((next_video[0][:, i, :].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).convert('RGBA')
#     aux_frame.putalpha(50)
#     img = Image.alpha_composite(img, aux_frame)
#
# plt.imshow(img)
# plt.show()
# a = 1