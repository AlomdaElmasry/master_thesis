from skimage.transform import AffineTransform, warp
import numpy as np
import torch
import matplotlib.pyplot as plt


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

    def _create_empty_transformation(self):
        return AffineTransform(translation=(0, 0), scale=(1, 1), rotation=0).inverse

    def simulate_movement(self, data_in, n, transformation_matrices=None):
        """Simulates a moving sequence of ``n` frames using ``frame`` as starting point.

        Args:
            data (torch.FloatTensor): tensor of size (C,H,W) containing the first frame.
            n (int): number of frames of the sequence.

        Returns:
            torch.FloatTensor: tensor of size (C,F,H,W) containing the moving sequence.
        """
        if transformation_matrices is None:
            transformation_matrices = [self._create_empty_transformation()]
            transformation_matrices += [self._create_random_transformation() for i in range(n - 1)]
        data_moved = torch.zeros((data_in.size(0), n, data_in.size(1), data_in.size(2)), dtype=torch.float32)
        data_moved[:, 0] = data_in
        for i in range(1, n):
            data_moved[:, i] = torch.from_numpy(
                warp(data_moved[:, i - 1].permute(1, 2, 0).numpy(), transformation_matrices[i])
            ).permute(2, 0, 1)
        return data_moved, transformation_matrices
