import argparse
from datasets.davis_2017 import DAVIS2017Dataset

# Create main parser and subparsers
parser = argparse.ArgumentParser(description='Copy-and-Paste Networks Implementation')
subparsers = parser.add_subparsers(dest='command')
subparsers_test = subparsers.add_parser(name='test')

# Add arguments to the parsers
parser.add_argument('--data-folder', required=True, help='Path to the folder where the data is stored')
parser.add_argument('--seed', type=int, default=0, help='Seed for the generation of random values')
parser.add_argument('--cuda', action='store_true', help='Whether you want to run in GPU')
parser.add_argument('--verbose', action='store_true', help='Whether to output the log using the standard output')

# Store the arguments inside args
args = parser.parse_args()

# Handle different commands
david_dataset = iter(DAVIS2017Dataset(split='train', data_folder=args.data_folder))

a = next(david_dataset)
b = 1