import os
import random
import glob
import pickle
import jpeg4py as jpeg
import time
import cv2
import cProfile
from datasets.content_provider import ContentProvider
from datasets.masked_sequence_dataset import MaskedSequenceDataset
import cProfile, pstats, io
from pstats import SortKey


got10k_folder = '/Users/DavidAlvarezDLT/Data/DAVIS-2017/JPEGImages'
items_in_folder = os.listdir(os.path.join(got10k_folder, '480p'))
random_sel = random.randint(0, len(items_in_folder) - 1)
images_paths = glob.glob(os.path.join(got10k_folder, '480p', items_in_folder[random_sel], '*.jpg'))

data_folder = '/Users/DavidAlvarezDLT/Data'
split = 'train'

img_path = '/Users/DavidAlvarezDLT/Data/GOT10k/train/GOT-10k_Val_000050/00000048.jpg'
# background_dataset = ContentProvider('got-10k', data_folder, split, None, return_mask=False)
# mask_dataset = ContentProvider('youtube-vos', data_folder, split, None, return_gt=False)
# mix_dataset = MaskedSequenceDataset(background_dataset, mask_dataset, (256, 256), 5, 2, force_resize=False)
#
# pr = cProfile.Profile()
# pr.enable()
# mix_dataset.__getitem__(0)
# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())

start = time.time()
img_read = cv2.imread(img_path)
end = time.time()
img_enc, _ = cv2.imencode('.jpg', img_read)
end2 = time.time()
print('{}'.format(end-start))

print('{}'.format(end2-end))
a = 1

# path = '/Users/DavidAlvarezDLT/Desktop/00000.jpg'
# start_time = time.time()
# a = jpeg.JPEG(path).decode() / 255
# end_time = time.time()
# print('Time per image: {}'.format(end_time - start_time))
#
# exit()
# print('Loading {} images from {}'.format(len(images_paths), items_in_folder[random_sel]))
#
# # JPEG4Y
# print('JPEG4PY TEST')
# start_time = time.time()
# data = []
# for i in images_paths:
#     data.append(jpeg.JPEG(i).decode() / 255)
# end_time = time.time()
# time_per_image = (end_time - start_time) / len(images_paths)
# print('Time per image: {}'.format(time_per_image))
# print('With 80 images: {}'.format(time_per_image * 80))
#
# # OpenCV2
# print('CV2 TEST')
# start_time = time.time()
# data = []
# for i in images_paths:
#     data.append(cv2.imread(i, cv2.IMREAD_COLOR) / 255)
# end_time = time.time()
# time_per_image = (end_time - start_time) / len(images_paths)
# print('Time per image: {}'.format(time_per_image))
# print('With 80 images: {}'.format(time_per_image * 80))
