import os
import random
import glob
import pickle
import jpeg4py as jpeg
import time
import cv2

got10k_folder = '/Users/DavidAlvarezDLT/Data/DAVIS-2017/JPEGImages'
items_in_folder = os.listdir(os.path.join(got10k_folder, '480p'))
random_sel = random.randint(0, len(items_in_folder) - 1)
images_paths = glob.glob(os.path.join(got10k_folder, '480p', items_in_folder[random_sel], '*.jpg'))

print('Loading {} images from {}'.format(len(images_paths), items_in_folder[random_sel]))

# JPEG4Y
print('JPEG4PY TEST')
start_time = time.time()
data = []
for i in images_paths:
    data.append(jpeg.JPEG(i).decode() / 255)
end_time = time.time()
time_per_image = (end_time - start_time) / len(images_paths)
print('Time per image: {}'.format(time_per_image))
print('With 80 images: {}'.format(time_per_image * 80))

# OpenCV2
print('CV2 TEST')
start_time = time.time()
data = []
for i in images_paths:
    data.append(cv2.imread(i, cv2.IMREAD_COLOR) / 255)
end_time = time.time()
time_per_image = (end_time - start_time) / len(images_paths)
print('Time per image: {}'.format(time_per_image))
print('With 80 images: {}'.format(time_per_image * 80))