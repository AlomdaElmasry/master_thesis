from skimage.transform import AffineTransform, warp
import numpy as np
import torch


class MovementSimulator:

    def __init__(self, max_displacement=1, max_scaling=0.01, max_rotation=np.pi / 16):
        self.max_displacement = max_displacement
        self.max_scaling = max_scaling
        self.max_rotation = max_rotation

    def _create_random_transformation(self):
        tx, ty = np.random.randint(low=-self.max_displacement, high=self.max_displacement, size=2)
        sx, sy = np.random.uniform(low=1 - self.max_scaling, high=1 + self.max_scaling, size=2)
        rot = np.random.uniform(low=-self.max_rotation, high=self.max_rotation)
        return AffineTransform(translation=(tx, ty), scale=(sx, sy), rotation=rot).inverse

    def simulate_movement(self, frame, n):
        frames = torch.zeros((1, n, frame.size(1), frame.size(2)), dtype=torch.float32)
        frames[:, 0] = frame
        for i in range(1, n):
            frames[:, i] = torch.from_numpy(
                warp(frames[:, i - 1].numpy().squeeze(0), self._create_random_transformation())
            )
        return frames
