import os.path
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, clips_array

# Define folder where the sequences are
sequences_folders = {
    'sota': '/Users/DavidAlvarezDLT/Desktop/CPN',
    'ff': '/Users/DavidAlvarezDLT/Desktop/CHN_FF',
    'ip': '/Users/DavidAlvarezDLT/Desktop/CHN_IP',
    'cp': '/Users/DavidAlvarezDLT/Desktop/CHN_CP'
}

# Define the folder where the results are stored
results_folder = '/Users/DavidAlvarezDLT/Desktop/results_sequences_videos/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Get all the videos of the first sequences_folder
videos_list = glob.glob(os.path.join(sequences_folders['sota'], '*.avi'))

# Iterate over the set of videos_list
for video_item in videos_list:
    video_name, videos = video_item.replace(sequences_folders['sota'], '')[1:-4], []
    for _, sequence_item in sequences_folders.items():
        videos.append(VideoFileClip(os.path.join(sequence_item, video_name + '.avi')))
    final_clip = clips_array([[videos[0], videos[1]], [videos[2], videos[3]]])
    final_clip.write_videofile(os.path.join(results_folder, video_name + '.mp4'))
