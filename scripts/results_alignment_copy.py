import argparse
import utils.paths
from thesis.dataset import ContentProvider, MaskedSequenceDataset
import models.thesis_alignment
import models.vgg_16
import random
import matplotlib.pyplot as plt
import utils.losses
import seaborn as sns
import torch
import pandas as pd
import numpy as np
import os.path
import thesis_alignment.runner
import progressbar

parser = argparse.ArgumentParser(description='Cleans invalid images')
parser.add_argument('--data-path', required=True, help='Path where the images are stored')
parser.add_argument('--experiments-path', required=True, help='Path where the experiments are stored')
parser.add_argument('--n-samples', default=1000, type=int, help='Number of samples')
parser.add_argument('--device', default='cpu', help='Device to use')
args = parser.parse_args()
