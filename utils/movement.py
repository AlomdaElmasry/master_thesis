from skimage.transform import AffineTransform, warp
import torch
import torch.nn.functional as F
import numpy as np


class MovementSimulator:

    def __init__(self, max_displacement=1, max_scaling=0.01, max_rotation=np.pi / 16):
        self.max_displacement = max_displacement
        self.max_scaling = max_scaling
        self.max_rotation = max_rotation

    def random_affine(self):
        tx, ty = np.random.randint(low=-self.max_displacement, high=self.max_displacement, size=2)
        sx, sy = np.random.uniform(low=1 - self.max_scaling, high=1 + self.max_scaling, size=2)
        rot = np.random.uniform(low=-self.max_rotation, high=self.max_rotation)
        affine_matrix = np.linalg.inv(AffineTransform(translation=(tx, ty), scale=(sx, sy), rotation=rot).params)
        return torch.from_numpy(affine_matrix).float()

    def empty_affine(self):
        affine_matrix = np.linalg.inv(AffineTransform(translation=(0, 0), scale=(1, 1), rotation=0).params)
        return torch.from_numpy(affine_matrix).float()

    def simulate_movement(self, data_in, n, random_affines=None):
        """Simulates a moving sequence of ``n` frames using ``frame`` as starting point.

        Args:
            data (torch.FloatTensor): tensor of size (C,H,W) containing the first frame.
            n (int): number of frames of the sequence.

        Returns:
            torch.FloatTensor: tensor of size (C,F,H,W) containing the moving sequence.
        """
        c, h, w = data_in.size()

        # Create a Tensor with n affine transformations
        if random_affines is None:
            random_affines = [self.empty_affine()]
            random_affines += [self.random_affine() for _ in range(n - 1)]
            random_affines = torch.stack(random_affines, dim=0)

        # Stack affine transformations with respect to the first frame
        random_affines_stacked = MovementSimulator.stack_transformations(random_affines, h, w, t=0)

        # Create theta matrices from affine matrices
        random_thetas_stacked = torch.stack([MovementSimulator.affine2theta(ra, h, w) for ra in random_affines_stacked])
        affine_grid = F.affine_grid(random_thetas_stacked, [n, c, h, w])
        data_out = F.grid_sample(data_in.unsqueeze(0).repeat(n, 1, 1, 1), affine_grid)

        # Return both data_out and random_thetas_stacked
        return data_out.permute(1, 0, 2, 3), random_thetas_stacked

    @staticmethod
    def stack_transformations(affine_matrices, h, w, t=0):
        affine_matrices_stacked = torch.zeros(affine_matrices.size(), dtype=torch.float32)
        affine_matrices_stacked[0] = affine_matrices[0]
        for i in range(1, affine_matrices.size(0)):
            affine_matrices_stacked[i] = torch.matmul(affine_matrices_stacked[i - 1], affine_matrices[i])
        return affine_matrices_stacked

    @staticmethod
    def affine2theta(param, h, w):
        theta = np.zeros([2, 3])
        theta[0, 0] = param[0, 0]
        theta[0, 1] = param[0, 1] * w / h
        theta[0, 2] = param[0, 2] * 2 / h + theta[0, 0] + theta[0, 1] - 1
        theta[1, 0] = param[1, 0] * h / w
        theta[1, 1] = param[1, 1]
        theta[1, 2] = param[1, 2] * 2 / w + theta[1, 0] + theta[1, 1] - 1
        return torch.from_numpy(theta).float()

    @staticmethod
    def theta2affine(theta, h, w):
        param = np.zeros([3, 3])
        param[0, 0] = theta[0, 0]
        param[0, 1] = theta[0, 1] * h / w
        param[0, 2] = (theta[0, 2] - theta[0, 0] - theta[0, 1] + 1) * h / 2
        param[1, 0] = theta[1, 0] * w / h
        param[1, 1] = theta[1, 1]
        param[1, 2] = (theta[1, 2] - theta[1, 0] - theta[1, 1] + 1) * w / 2
        param[2, 2] = 1
        return torch.from_numpy(param).float()
