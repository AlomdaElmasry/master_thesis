import argparse
import utils.paths
from thesis.dataset import ContentProvider, MaskedSequenceDataset

parser = argparse.ArgumentParser(description='Cleans invalid images')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
args = parser.parse_args()

# Iterate over the set of displacements
for s in range(1, 11):
    # Create the dataset object
    dataset = MaskedSequenceDataset(
        gts_dataset=gts_datasets[0],
        masks_dataset=masks_datasets[0],
        gts_simulator=None,
        masks_simulator=None,
        image_size=(256, 256),
        frames_n=2,
        frames_spacing=self.experiment.configuration.get('data', 'frames_spacing') * 2,
        frames_randomize=self.experiment.configuration.get('data', 'frames_randomize'),
        dilatation_filter_size=tuple(self.experiment.configuration.get('data', 'dilatation_filter_size')),
        dilatation_iterations=self.experiment.configuration.get('data', 'dilatation_iterations'),
        force_resize=self.experiment.configuration.get('data', 'train_resize'),
        keep_ratio=True,
        p_simulator_gts=self.experiment.configuration.get('data', 'p_simulator_gts'),
        p_simulator_masks=self.experiment.configuration.get('data', 'p_simulator_masks'),
        p_repeat=self.experiment.configuration.get('data', 'p_repeat')
    )
