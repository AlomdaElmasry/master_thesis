from datasets.sequence_dataset import SequencesDataset
from datasets.masks_dataset import MasksDataset
from utils.movement import MovementSimulator
import glob
import os.path
from PIL import Image
from pathlib import Path


dataset_path = '/Users/DavidAlvarezDLT/Data/GOT10k'
images_extensions = ['png', 'jpeg', 'jpg']

# Iterate over extensions
for ext in images_extensions:
    for image_path in Path(dataset_path).rglob('*.{}'.format(ext)):
        try:
            pil_image = Image.open(image_path)
        except Exception:
            os.remove(image_path)
            print('Image {} not valid. Removed.'.format(image_path))