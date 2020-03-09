import argparse
import concurrent.futures
import os
import progressbar
import cv2
import numpy as np
import shutil

parser = argparse.ArgumentParser(description='Crops mask to the visual')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--destination-path', required=True, help='Whether or not to remove entire sequence')
parser.add_argument('--formats', nargs='+', default=['jpg', 'jpeg', 'png'], help='Image formats to search in the path')
parser.add_argument('--max-workers', type=int, default=10, help='Number of workers to use')
args = parser.parse_args()


def crop_image(image, tol=0):
    return image[np.ix_(image.any(1), image.any(0))]


def crop_sequence(sequence_path, bar, i):
    new_folder_path = folder_path.replace(args.data_path, args.destination_path)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    folder_content = os.listdir(folder_path)
    for folder_item in folder_content:
        file_path = os.path.join(folder_path, folder_item)
        if os.path.isfile(file_path):
            new_file_path = file_path.replace(folder_path, new_folder_path)
            file_extension = os.path.splitext(folder_item)[-1].replace('.', '')
            if file_extension in ['png']:
                image = cv2.imread(os.path.join(folder_path, folder_item), cv2.IMREAD_GRAYSCALE)
                image_croped = crop_image(image)
                if image_croped.shape != (0, 0):
                    cv2.imwrite(new_file_path, image_croped)
                else:
                    print('Image {} has no mask in it'.format(file_path))
            else:
                pass # shutil.copy(file_path, new_file_path)
    # Update bar counter
    if bar.value < i:
        bar.update(i)


# Create ThreadPool for parallel execution
executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers)

# Generate a list of sequences
folder_paths = sorted([root for root, _, _ in os.walk(args.data_path)])

# Create progress bar
bar = progressbar.ProgressBar(max_value=len(folder_paths))

# Walk through the folders of args.data_path
for i, folder_path in enumerate(folder_paths):
    crop_sequence(folder_path, bar, i)
    #executor.submit(crop_sequence, folder_path, bar, i)
