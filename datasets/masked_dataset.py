import torch.utils.data
import datasets.davis_2017
import datasets.coco_masks


class MaskedDataset(torch.utils.data.Dataset):

    def __init__(self, data, masks):
        self.data = data
        self.masks = masks
        self._validate_arguments()

    def _validate_arguments(self):
        assert isinstance(self.data, datasets.davis_2017.DAVIS2017Dataset)
        if self.masks is not None:
            assert isinstance(self.masks, datasets.coco_masks.COCOMasks)

    def __getitem__(self, item):
        gt, annotations, info = self.data[item]
        masks = self.masks.get_item(10, (gt.size(2), gt.size(3)), gt.size(1)) if self.masks else annotations
        masked_frames = (1 - masks) * gt + masks*torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        return (masked_frames, masks), gt, info

    def __len__(self):
        return len(self.data)
