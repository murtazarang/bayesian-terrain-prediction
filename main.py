import torch
import numpy as np
import pandas as pd
import warnings

import matplotlib
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.figsize'] = (12.0, 12.0)

from config_args import parse_args
from data_utils.data_preprocess import load_data
from data_utils.seq2seq_processing import seq_data
from trainer_utils.trainer import TorchTrainer


def train():
    # Parse arguments and load data
    args = parse_args()

    # If new dataset is to be loaded and processed with scaling/norms etc, then
    # Create batches of input sequence and output sequence that needs to be predicted
    if args.load_data:
        load_data(args)
        seq_data(args)
    elif args.sequence_data:
        seq_data(args)

    sequence_data = pd.read_pickle('./data/sequence_data/' + args.dataset + '.pkl')

if __name__ == '__main__':
    train()

