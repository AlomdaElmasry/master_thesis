import torch.utils.data
import os
import pycocotools.coco
import torch
import torch.nn.functional as F
import math
import numpy as np
import utils
import random


class MasksDataset(torch.utils.data.Dataset):
    dataset_folder = None
    device = None

    def __init__(self, dataset_folder, device=torch.device('cpu')):
        self.dataset_folder = dataset_folder
        self.device = device
        self.emulator = utils.MovementSimulator()
        self.json_path = os.path.join(dataset_folder, 'instances_train2017.json')
        self.coco = pycocotools.coco.COCO(self.json_path)
        self.masks_ids = self.coco.getAnnIds(iscrowd=False)

    def _resize_frame(self, frame, size):
        """ Fits the image size of the dataset. Given an input of size (C,H,W):
            1. Resizes the larger dimension to match the desired size.
        """
        # Get dimensions and create dim=0 representing the batch
        _, H, W = frame.size()
        frame = frame.unsqueeze(0)

        # Step 1: resize the larger dimension
        new_H = size[0] if H > W else int(H * size[1] / W)
        new_W = size[1] if W > H else int(W * size[0] / H)
        frame = F.interpolate(frame, size=(new_H, new_W))

        # Step 2: pad with zeros if required
        H_pad = size[0] - frame.size(2) if size[0] > frame.size(2) else 0
        W_pad = size[1] - frame.size(3) if size[1] > frame.size(3) else 0
        frame = F.pad(frame, (math.ceil(W_pad / 2), math.ceil(W_pad / 2), math.ceil(H_pad / 2), math.ceil(H_pad / 2)))

        # Step 3: cut if required and return
        return frame[0, :, :size[0], :size[1]]

    def _resize_frame_simple(self, frame, size):
        return F.interpolate(frame.unsqueeze(0), size=size).squeeze(0)

    def __getitem__(self, item):
        raise NotImplementedError

    def get_random_item(self, size, n):
        if True:
            item = self._get_random_item()
            mask = torch.from_numpy(self.coco.annToMask(self.coco.loadAnns(self.masks_ids[item])[0]).astype(np.float32))
            frames = self.emulator.simulate_movement(self._resize_frame_simple(mask.unsqueeze(0), size), n)
            return torch.as_tensor((frames > 0.5), dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.masks_ids)

    def _get_random_item(self):
        return random.randint(0, len(self.masks_ids))
