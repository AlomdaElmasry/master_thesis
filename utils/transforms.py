import cv2
import numpy as np
import random


class ImageTransforms:

    @staticmethod
    def resize(image, size, method=cv2.INTER_LINEAR):
        image = cv2.resize(image, dsize=(size[1], size[0]), interpolation=method)
        return np.expand_dims(image, 2) if len(image.shape) == 2 else image

    @staticmethod
    def crop(image, size, crop_position=None):
        if crop_position is None:
            crop_position = (random.randint(0, image.shape[0] - size[0]), random.randint(0, image.shape[1] - size[1]))
        return image[crop_position[0]:crop_position[0] + size[0], crop_position[1]:crop_position[1] + size[1], :], \
               crop_position

    @staticmethod
    def binarize(image, threshold=0.5):
        return (image > threshold).astype(np.uint8)

    @staticmethod
    def dilatate(image, filter_size, iterations):
        image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_CROSS, filter_size), iterations=iterations) \
            .astype(np.float32)
        return np.expand_dims(image, 2) if len(image.shape) == 2 else image
