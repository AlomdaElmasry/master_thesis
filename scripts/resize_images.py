import argparse
import os
import cv2
import shutil
import concurrent.futures
import progressbar

parser = argparse.ArgumentParser(description='Cleans invalid images')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--destination-path', required=True, help='Destination path where the will be stored')
parser.add_argument('--max-width', type=int, default=910, help='Number of workers to use')
parser.add_argument('--max-height', type=int, default=480, help='Number of workers to use')
parser.add_argument('--keep-ratio', type=bool, default=True, help='Number of workers to use')
parser.add_argument('--max-workers', type=int, default=10, help='Number of workers to use')

args = parser.parse_args()


# Create function to handle multi-threading
def handle_folder(folder_path, args, bar, i):
    new_folder_path = folder_path.replace(args.data_path, args.destination_path)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    folder_content = os.listdir(folder_path)
    for folder_item in folder_content:
        file_path = os.path.join(folder_path, folder_item)
        if os.path.isfile(file_path):
            new_file_path = file_path.replace(folder_path, new_folder_path)
            file_extension = os.path.splitext(folder_item)[-1].replace('.', '')
            saving_flags = [int(cv2.IMWRITE_JPEG_QUALITY), 50] if file_extension == 'jpg' else []
            if file_extension in ['jpg', 'png']:
                image = cv2.imread(os.path.join(folder_path, folder_item))
                if image.shape[0] > args.max_height or image.shape[1] > args.max_width:
                    new_height = args.max_height if image.shape[0] >= image.shape[1] else \
                        round(image.shape[0] * args.max_width / image.shape[1])
                    new_width = args.max_width if image.shape[1] > image.shape[0] else \
                        round(image.shape[1] * args.max_height / image.shape[0])
                    new_height = new_height if args.keep_ratio else args.max_height
                    new_width = new_width if args.keep_ratio else args.max_width
                    image = cv2.resize(image, (new_width, new_height))
                cv2.imwrite(new_file_path, image, saving_flags)
            else:
                shutil.copy(file_path, new_file_path)
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
    executor.submit(handle_folder, folder_path, args, bar, i)
