import argparse
import skeltorch
from copy_paste.data import CopyPasteData
from copy_paste.runner import CopyPasteRunner

parser = argparse.ArgumentParser(description='Creates a video from a set of static frames')
parser.add_argument('--experiment-name', required=True, help='Name of the experiment')
parser.add_argument('--experiments-path', required=True, help='Path to the folder containing the frames')
parser.add_argument('--data-path', required=True, help='Path to the folder containing the frames')
args = parser.parse_args()

# Load data sets

# Iterate over the data sets
