import argparse
import cv2
import numpy as np
import os
import re
import time

parser = argparse.ArgumentParser(description='Creates a video from a set of static frames')
parser.add_argument('--frames', required=True, nargs='+', help='Path to one or more folders containing the frames')
parser.add_argument('--dest_folder', type=str, default='.', help='Path where the resulting video should be saved')
parser.add_argument('--margin', type=int, default=10, help='Margin added to each of the items')
parser.add_argument('--rate', type=int, default=25, help='Frame rate of the resulting video')
parser.add_argument('--resize', default=None, choices=['upscale', 'downscale'], help='Resizing strategy')
parser.add_argument('--filename', type=str, help='Force a name for the output file')
args = parser.parse_args()

# Accept a maximum grid of 2x2 videos
if len(args.frames) > 4:
    exit('The maximum number of frame folders that you can place in one video is 4.')

# Load the path of all the images for every element of the grid
frames = []
for video_path in args.frames:
    video_frames = []
    for frame_path in sorted(os.listdir(video_path), key=lambda x: int(re.search(r'\d+', x).group())):
        video_frames.append(cv2.imread(os.path.join(video_path, frame_path)))
    frames.append(video_frames)

# Check that the size of all the videos is the same
frame_size = frames[0][0].shape
for video in frames:
    for frame in video:
        if frame_size != frame.shape:
            if args.resize is None:
                exit('All the videos must be of the same size')
            elif args.resize == 'upscale' and frame.shape[0] > frame_size[0] or args.resize == 'downscale' and \
                    frame.shape[0] < frame_size[0]:
                frame_size = frame.shape

# If the number of frames is different in the different frame paths, cut the video
num_frames = min([len(f) for f in frames])
item_size = (frame_size[0] + 2 * args.margin, frame_size[1] + 2 * args.margin)
video_size = (item_size[0] * (2 if len(frames) > 2 else 1), item_size[1] * (2 if len(frames) > 1 else 1), 3)

# Create the the destination folder if it does not exists. Set the filename using the current time
if not os.path.exists(args.dest_folder):
    os.makedirs(args.dest_folder)
if args.filename:
    video_name = os.path.join(args.dest_folder, args.filename + '.avi')
else:
    while True:
        video_name = os.path.join(args.dest_folder, time.strftime("%H%M%S") + '.avi')
        if not os.path.exists(video_name):
            break

# Create the video from the frames
final_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), args.rate, (video_size[1], video_size[0]))
for i in range(num_frames):
    it_frame = np.zeros(video_size, dtype=np.uint8)
    for j in range(len(frames)):
        video_frame = frames[j][i]
        if video_frame.shape != frame_size:
            video_frame = cv2.resize(video_frame, dsize=(frame_size[1], frame_size[0]), interpolation=cv2.INTER_LINEAR)
        video_frame = np.pad(video_frame, ((args.margin, args.margin), (args.margin, args.margin), (0, 0)))
        it_frame[(j // 2) * item_size[0]:(j // 2 + 1) * item_size[0], (j % 2) * item_size[1]:(j % 2 + 1) * item_size[1],
        :] = video_frame
    final_video.write(it_frame)
final_video.release()
