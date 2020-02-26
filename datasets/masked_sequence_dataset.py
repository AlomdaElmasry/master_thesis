import torch.utils.data
import utils.transforms


class MaskedSequenceDataset(torch.utils.data.Dataset):
    gts_dataset = None
    masks_dataset = None
    image_size = None
    frames_n = None
    frames_spacing = None
    force_resize = None
    fill_color = None

    def __init__(self, gts_dataset, masks_dataset, image_size, frames_n, frames_spacing, force_resize=False):
        self.gts_dataset = gts_dataset
        self.masks_dataset = masks_dataset
        self.image_size = image_size
        self.frames_n = frames_n
        self.frames_spacing = frames_spacing
        self.force_resize = force_resize
        self.fill_color = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32)

    def __getitem__(self, item):
        # Get the data associated to the GT
        y, m, info = self.gts_dataset.get_sequence(item) if self.frames_n == -1 \
            else self.gts_dataset.get_patch(item, self.frames_n, self.frames_spacing)

        # If self.gts_dataset and self.masks_dataset are not the same, obtain new mask
        if self.gts_dataset != self.masks_dataset:
            masks_n = self.frames_n if self.frames_n != -1 else y.size(1)
            _, m, _ = self.masks_dataset.get_patch_random(masks_n, self.frames_spacing)

        # Apply GT transformations
        if self.force_resize:
            y = utils.transforms.ImageTransforms.resize(y, self.image_size)
        else:
            y, _ = utils.transforms.ImageTransforms.crop(y, self.image_size)

        # Apply Mask transformations
        m = utils.transforms.ImageTransforms.resize(m, self.image_size)
        m = (torch.sum(m, dim=0, keepdim=True) > 0).float()
        # m = utils.transforms.ImageTransforms.dilatate(m, (3, 3), 4)

        # Compute x
        x = (1 - m) * y + (m.permute(3, 2, 1, 0) * self.fill_color).permute(3, 2, 1, 0)

        # Return (x,m), y, info
        return (x, m), y, info

    def __len__(self):
        return self.gts_dataset.len_sequences() if self.frames_n == -1 else len(self.gts_dataset)
