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
        assert isinstance(self.masks, datasets.coco_masks.COCOMasks)

    def __getitem__(self, item):
        gt, annotations = self.data[item]
        masks = self.masks.get_item(10, (gt.size(2), gt.size(3)), gt.size(1))
        masked_frames = (1 - masks) * gt
        return masked_frames, masks, gt

    def __len__(self):
        return len(self.data)
