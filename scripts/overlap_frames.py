from PIL import Image
from datasets.davis_2017 import DAVIS2017Dataset
import numpy as np
import matplotlib.pyplot as plt

train_dataset = DAVIS2017Dataset(dataset_folder='/Users/DavidAlvarezDLT/Data/DAVIS-2017', split='train')
next_video = next(iter(train_dataset))

img = Image.fromarray((next_video[0][:, 0, :].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).convert('RGBA')
for i in range(1, next_video[0].size(1), 10):
    aux_frame = Image.fromarray((next_video[0][:, i, :].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).convert('RGBA')
    aux_frame.putalpha(50)
    img = Image.alpha_composite(img, aux_frame)

plt.imshow(img)
plt.show()
a = 1