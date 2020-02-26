import torch
from PIL import Image
from pathlib import Path
import os
import numpy as np
import progressbar
from utils.transforms import ImageTransforms
import argparse
import concurrent.futures

parser = argparse.ArgumentParser(description='Cleans invalid images')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--formats', nargs='+', default=['jpg', 'jpeg', 'png'], help='Image formats to search in the path')
parser.add_argument('--max-workers', type=int, default=10, help='Number of workers to use')
args = parser.parse_args()

# Generate the list of images to verify
images_paths = []
for ext in args.formats:
    images_paths += list(Path(args.data_path).rglob('*.{}'.format(ext)))

# Create progress bar
bar = progressbar.ProgressBar(max_value=len(images_paths))


def verify_image(image_path, bar, i):
    try:
        image = torch.from_numpy(
            (np.array(Image.open(image_path).convert('RGB')) / 255).astype(np.float32)
        ).permute(2, 0, 1)
        image, _ = ImageTransforms.crop(image, (256, 256))
        bar.update(i)
    except Exception:
        os.remove(images_paths[i])
        print('Image {} not valid. Removed.'.format(image_path))


with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    for i in range(len(images_paths)):
        executor.submit(verify_image, images_paths[i], bar, i)
