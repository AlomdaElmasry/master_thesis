from skimage.transform import AffineTransform, warp
import numpy as np


class MovementEmulator:

    def __init__(self, max_displacement=5, max_scaling=0.1):
        self.max_displacement = max_displacement
        self.max_scaling = max_scaling
        self.max_rotation = np.pi / 16

    def _create_random_transformation(self):
        tx, ty = np.random.randint(low=-self.max_displacement, high=self.max_displacement, size=2)
        sx, sy = np.random.uniform(low=1 - self.max_scaling, high=1 + self.max_scaling, size=2)
        rot = np.random.uniform(low=-self.max_rotation, high=self.max_rotation)
        return AffineTransform(translation=(tx, ty), scale=(sx, sy), rotation=rot).inverse

    def simulate_movement(self, frame, n):
        frames = [frame]
        for i in range(n):
            frames.append(warp(frames[i], self._create_random_transformation()))
        return frames