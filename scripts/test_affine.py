import numpy as np
import jpeg4py as jpeg
import matplotlib.pyplot as plt
from utils import MovementSimulator
import torch.nn.functional as F
import torch


mov = MovementSimulator()
#affine_matrix = np.array([[1, 0, 50], [0, 1, 50], [0, 0, 1]])
affine_matrix = mov.random_affine()
affine_inverse = np.linalg.inv(affine_matrix)
img = jpeg.JPEG('/Users/DavidAlvarezDLT/Data/GOT10k/train/GOT-10k_Val_000001/00000001.jpg').decode() / 255
img = torch.from_numpy(img).float().permute(2, 0, 1)
c, h, w = img.shape
n = 1
plt.imshow(img.permute(1, 2, 0))
plt.show()

# Transform entire image
theta_matrix = MovementSimulator.affine2theta(affine_matrix, h, w)
affine_grid = F.affine_grid(theta_matrix.unsqueeze(0), [n, c, h, w])
trasf_image = F.grid_sample(img.unsqueeze(0), affine_grid).squeeze(0)
plt.imshow(trasf_image.permute(1, 2, 0))
plt.show()

# Reverse transformation
theta_matrix = MovementSimulator.affine2theta(affine_inverse, h, w)
affine_grid = F.affine_grid(theta_matrix.unsqueeze(0), [n, c, h, w])
trasf_image = F.grid_sample(trasf_image.unsqueeze(0), affine_grid).squeeze(0)
plt.imshow(trasf_image.permute(1, 2, 0))
plt.show()

# Crop original image
crop_position = (200, 400)
img_cropped = img[:, crop_position[0]:crop_position[0] + 256, crop_position[1]:crop_position[1] + 256]
plt.imshow(img_cropped.permute(1, 2, 0))
plt.show()

# Lala
theta_matrix = MovementSimulator.affine2theta(affine_matrix, 256, 256)
affine_grid = F.affine_grid(theta_matrix.unsqueeze(0), [n, c, 256, 256])
trasf_image = F.grid_sample(img_cropped.unsqueeze(0), affine_grid).squeeze(0)
plt.imshow(trasf_image.permute(1, 2, 0))
plt.show()

exit()


# Crop transformed image
img_transf_cropped = trasf_image[:, crop_position[0]:crop_position[0] + 256, crop_position[1]:crop_position[1] + 256]

# Transform crop of the image
theta_inv = MovementSimulator.affine2theta(affine_inverse, 256, 256)
affine_inv = F.affine_grid(theta_inv.unsqueeze(0), [n, c, h, w])
img_transf_cropped_inv = F.grid_sample(img_transf_cropped.unsqueeze(0), affine_inv).squeeze(0)

# Plot original image

# Plot transformed image

# Plot cropped original image

# Plot cropped transformed image
plt.imshow(img_transf_cropped.permute(1, 2, 0))
plt.show()

# Plot TARGET
plt.imshow(img_transf_cropped_inv.permute(1, 2, 0))
plt.show()
