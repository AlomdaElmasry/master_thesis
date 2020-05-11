from PIL import Image, ImageDraw
import numpy as np
import torch
import matplotlib.pyplot as plt


def text_to_image(labels, width, height=50):
    images_tensor = torch.zeros((len(labels), 3, height, width))
    for i, label in enumerate(labels):
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(label)
        draw.text(((width - w) // 2, (height - h) // 2), label, fill=(0, 0, 0))
        images_tensor[i] = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    return images_tensor


a = text_to_image(['Test 1', 'Test 2'], width=256)

for b in range(a.size(0)):
    plt.imshow(a[b].permute(1, 2, 0).numpy())
    plt.show()

b = 1
