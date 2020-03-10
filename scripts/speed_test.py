import argparse
from pathlib import Path
import random
import time
import jpeg4py
import cv2

parser = argparse.ArgumentParser(description='Cleans invalid images')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--max-items', type=int, default=1000, help='Number of images to analyze')
parser.add_argument('--ext', default='jpg', choices=['jpg', 'png'], help='Extension of the images')
parser.add_argument('--seed', type=int, default=0, help='Number of images to analyze')
args = parser.parse_args()

# Find 100 images inside data_path
folder_paths = list(Path(args.data_path).rglob("*.{}".format(args.ext)))

# Set the seed
random.seed(args.seed)

# Choose max_items random elements
analyzed_paths = random.sample(folder_paths, args.max_items)

# Start timing
if args.ext == 'jpg':
    time_start = time.time()
    for path in analyzed_paths:
        image = jpeg4py.JPEG(path).decode()
    time_end = time.time()
else:
    time_start = time.time()
    for path in analyzed_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    time_end = time.time()

# Print result
print('Analyzed {} images - Mean time: {}'.format(args.max_items, (time_end - time_start) / args.max_items))
