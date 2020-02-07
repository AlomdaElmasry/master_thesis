import argparse
import cv2
import numpy as np
import os
import time

parser = argparse.ArgumentParser(description='Creates a video from a set of static frames')
parser.add_argument('--frames', required=True, nargs='+', help='Path to one or more folders containing the frames')
parser.add_argument('--dest_folder', type=str, default='.', help='Path where the resulting video should be saved')
parser.add_argument('--margin', type=int, default=10, help='Margin added to each of the items')
parser.add_argument('--rate', type=int, default=25, help='Frame rate of the resulting video')
args = parser.parse_args()

# Accept a maximum grid of 2x2 videos
if len(args.frames) > 4:
    exit('The maximum number of frame folders that you can place in one video is 4.')

# Load the path of all the images for every element of the grid
frames = []
for video_path in args.frames:
    video_frames = []
    for frame_path in sorted(os.listdir(video_path)):
        video_frames.append(cv2.imread(os.path.join(video_path, frame_path)))
    frames.append(video_frames)

# Check that the size of all the videos is the same
prev_size = None
for video in frames:
    for frame in video:
        if prev_size != frame.shape and prev_size is not None:
            exit('All the videos must be of the same size')
        else:
            prev_size = frame.shape

# If the number of frames is different in the different frame paths, cut the video
num_frames = min([len(f) for f in frames])
item_size = (prev_size[0] + 2 * args.margin, prev_size[1] + 2 * args.margin)
video_size = (item_size[0] * (2 if len(frames) > 2 else 1), item_size[1] * (2 if len(frames) > 1 else 1), 3)

# Create the the destination folder if it does not exists. Set the filename using the current time
if not os.path.exists(args.dest_folder):
    os.makedirs(args.dest_folder)
while True:
    video_name = os.path.join(args.dest_folder, time.strftime("%H%M%S") + '.avi')
    if not os.path.exists(video_name):
        break

# Create the video from the frames
final_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), args.rate, (video_size[1], video_size[0]))
for i in range(num_frames):
    it_frame = np.zeros(video_size, dtype=np.uint8)
    for j in range(len(frames)):
        video_frame = np.pad(frames[j][i], ((args.margin, args.margin), (args.margin, args.margin), (0, 0)))
        it_frame[(j // 2) * item_size[0]:(j // 2 + 1) * item_size[0], (j % 2) * item_size[1]:(j % 2 + 1) * item_size[1],
        :] = video_frame
    final_video.write(it_frame)
final_video.release()
