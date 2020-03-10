import argparse
import cProfile
from datasets.content_provider import ContentProvider
from datasets.masked_sequence_dataset import MaskedSequenceDataset
import io
import pstats

parser = argparse.ArgumentParser(description='Creates a video from a set of static frames')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--gts-dataset', default='got-10k', help='Name of GTs dataset')
parser.add_argument('--gts-split', default='train', help='Split of GTs dataset')
parser.add_argument('--masks-dataset', default='youtube-vos', help='Name of masks dataset')
parser.add_argument('--masks-split', default='train', help='Split of masks dataset')
parser.add_argument('--image-size', default=256, type=int, help='Size of the image')
parser.add_argument('--frames-n', default=5, type=int, help='Frame number')
parser.add_argument('--frames-spacing', default=2, type=int, help='Frame spacing')
args = parser.parse_args()

# Load data sets
gts_dataset = ContentProvider(args.gts_dataset, args.data_path, args.gts_split, None, None, return_mask=False)
masks_dataset = ContentProvider(args.masks_dataset, args.data_path, args.masks_split, None, None, return_gt=False)
dataset = MaskedSequenceDataset(gts_dataset, masks_dataset, (args.image_size, args.image_size), args.frames_n,
                                args.frames_spacing)

# Enable profiling
pr = cProfile.Profile()
pr.enable()
item = dataset[10]
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
ps.print_stats()
print(s.getvalue())
