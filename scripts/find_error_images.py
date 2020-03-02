import torch
from PIL import Image
from pathlib import Path
import os
import numpy as np
import progressbar
import shutil
from utils.transforms import ImageTransforms
import argparse
import concurrent.futures
import glob

parser = argparse.ArgumentParser(description='Cleans invalid images')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--formats', nargs='+', default=['jpg', 'jpeg', 'png'], help='Image formats to search in the path')
parser.add_argument('--max-workers', type=int, default=10, help='Number of workers to use')
args = parser.parse_args()

# Generate the list of images to verify
# images_paths = []
# for ext in args.formats:
#     images_paths += list(Path(args.data_path).rglob('*.{}'.format(ext)))

# Generate a list of sequences
images_sequences = glob.glob(os.path.join(args.data_path, '*'))

# Create progress bar
bar = progressbar.ProgressBar(max_value=len(images_sequences))


def verify_sequence(sequence_path, bar, i):
    try:
        images_paths = []
        for ext in args.formats:
            images_paths += glob.glob(os.path.join(sequence_path, '*.{}'.format(ext)))
        image_size = None
        for image_path in images_paths:
            image = torch.from_numpy(
                (np.array(Image.open(image_path).convert('RGB')) / 255).astype(np.float32)
            ).permute(2, 0, 1)
            if image_size is not None and image.size() != image_size:
                print('Sequence {} not correct'.format(sequence_path))
                raise ValueError
            elif image_size is None:
                image_size = image.size()
        if bar.value < i:
            bar.update(i)
    except Exception:
        shutil.rmtree(sequence_path)


with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    for i in range(len(images_sequences)):
        executor.submit(verify_sequence, images_sequences[i], bar, i)