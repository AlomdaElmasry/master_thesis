from datasets.content_provider import ContentProvider
from datasets.masked_sequence_dataset import MaskedSequenceDataset
import torch.utils.data

data_path = '/Users/DavidAlvarezDLT/Data'

train_gts_dataset = ContentProvider(
    dataset_name='got-10k',
    data_folder=data_path,
    split='train',
    movement_simulator=None,
    return_mask=False
)

train_masks_dataset = ContentProvider(
    dataset_name='youtube-vos',
    data_folder=data_path,
    split='train',
    movement_simulator=None,
    return_gt=False
)

global_dataset = MaskedSequenceDataset(
    gts_dataset=train_gts_dataset,
    masks_dataset=train_masks_dataset,
    image_size=(256, 256),
    frames_n=5,
    frames_spacing=2
)

loader = torch.utils.data.DataLoader(
    dataset=global_dataset,
    batch_size=8
)

for i, d in enumerate(loader):
    print(i)
