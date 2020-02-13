import cv2
import numpy as np


class Resize:

    def __init__(self, size, method):
        self.size = size
        self.method = method

    def __call__(self, image):
        image = cv2.resize(image, dsize=(self.size[1], self.size[0]), interpolation=self.method)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        return image


class Binarize:

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, image):
        return (image > self.threshold).astype(np.uint8)


class Dilatate:

    def __init__(self, filter_size, iterations):
        self.filter_size = filter_size
        self.iterations = iterations

    def __call__(self, image):
        image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_CROSS, self.filter_size),
                           iterations=self.iterations).astype(np.float32)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        return image
