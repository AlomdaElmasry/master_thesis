import torch.utils.data
import os
import pycocotools.coco
from . import MovementEmulator
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import numpy as np


class COCOMasks(torch.utils.data.Dataset):

    def __init__(self, data_folder, json_filename='instances_train2017.json'):
        self.emulator = MovementEmulator()
        self.json_path = os.path.join(data_folder, json_filename)
        self.coco = pycocotools.coco.COCO(self.json_path)
        self.masks_ids = self.coco.getAnnIds(iscrowd=False)
        self.size = (1000, 1000)

    def _resize_frame(self, frame):
        """ Fits the image size of the dataset. Given an input of size (C,H,W):
            1. Resizes the larger dimension to match the desired size.
        """
        # Get dimensions and create dim=0 representing the batch
        _, H, W = frame.size()
        frame = frame.unsqueeze(0)

        # Step 1: resize the larger dimension
        new_H = self.size[0] if H > W else int(H * self.size[1] / W)
        new_W = self.size[1] if W > H else int(W * self.size[0] / H)
        frame = F.interpolate(frame, size=(new_H, new_W))

        # Step 2: pad with zeros if required
        H_pad = frame.size(2) - self.size[0] if self.size[0] > frame.size(2) else 0
        W_pad = self.size[1] - frame.size(3) if self.size[1] > frame.size(3) else 0
        frame = F.pad(frame, (math.ceil(W_pad/2), math.ceil(W_pad/2), math.ceil(H_pad/2), math.ceil(H_pad/2)))

        # Step 3: cut if required and return
        return frame[0, :, :self.size[0], :self.size[1]]

    def __getitem__(self, index):
        mask = self.coco.annToMask(self.coco.loadAnns(self.masks_ids[index])[0]).astype(np.float)
        mask = torch.from_numpy(mask)
        frame = self._resize_frame(mask.unsqueeze(0))
        return self.emulator.simulate_movement(frame, 10)

    def __len__(self):
        return len(self.masks_ids)
