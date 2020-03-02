import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F


class ImageTransforms:

    @staticmethod
    def resize(image, size, mode='bilinear'):
        """Resize an image using the the algorithm given in ``mode``.

        Args:
            image (torch.FloatTensor): tensor of size (C,F,H,W) containing the image quantized from [0, 1].
            size (tuple): tuple containing the desired size in the form (H, W).
            mode (str): mode used to resize the image. Same format as in ``torch.nn.functional.interpolate()``.

        Returns:
            torch.FloatTensor: resized image.
        """
        return F.interpolate(image.transpose(0, 1), size, mode=mode, align_corners=False).transpose(0, 1)

    @staticmethod
    def crop(image, size, crop_position=None):
        """Crop a patch from the image.

        Args:
            image (torch.FloatTensor): tensor of size (C, F, H, W) containing the image.
            size (tuple): tuple containing the desired size in the form (H, W).
            crop_position (tuple): coordinates of the top-left pixel from where to cut the patch. If not set, it is
            generated randomly.

        Returns:
            torch.FloatTensor: patch of the image.
        """
        if crop_position is None:
            crop_position = (random.randint(0, image.size(2) - size[0]), random.randint(0, image.size(3) - size[1]))
        return image[:, :, crop_position[0]:crop_position[0] + size[0], crop_position[1]:crop_position[1] + size[1]], \
               crop_position

    @staticmethod
    def binarize(image, threshold=0.5):
        """Binarizes an image using ``threshold``.

        Args:
            image (torch.FloatTensor): tensor of size (C, H, W) containing the image.
            threshold (float): value used to binarize the image.

        Returns:
            torch.FloatTensor: binary image containing only 0's and 1s.
        """
        return (torch.sum(image, dim=0) > threshold).type(torch.float32)

    @staticmethod
    def dilatate(image, filter_size, iterations):
        """Dilatates an image with a filter of size ``filter_size``.

        Args:
            image (torch.FloatTensor): tensor of size (C,F,H,W) containing the image.
            filter_size (tuple): size of the filter in the form (H,W).
            iterations (integer): number of times to apply the filter.

        Returns:
            torch.FloatTensor: dilatated image.
        """
        image = cv2.dilate(
            image.permute(1, 2, 0).numpy(),
            cv2.getStructuringElement(cv2.MORPH_CROSS, filter_size),
            iterations=iterations
        ).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)
