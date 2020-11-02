import torch
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.figsize'] = (12.0, 12.0)

import multiprocessing as mp

from config_args import parse_args
from data_utils.data_preprocess import load_data
from data_utils.sequence_builder import seq_builder

from trainer_utils.trainer import TorchTrainer


def train():
    # Parse arguments and load data
    args = parse_args()

    # If new dataset is to be loaded and processed with scaling/norms etc, then
    # Create batches of input sequence and output sequence that needs to be predicted
    if args.load_data:
        with mp.Pool(1) as pool:
            result = pool.map(load_data, [args])[0]
        with mp.Pool(2) as pool:
            result = pool.map(seq_builder, [(args, False), (args, True)])[0]
        # with mp.Pool(1) as pool:
        #     result = pool.map(seq_builder, [(args, True)])[0]
    elif args.sequence_data:
        with mp.Pool(1) as pool:
            result = pool.map(seq_builder, [(args, False), (args, True)])[0]
        # with mp.Pool(1) as pool:
        #     result = pool.map(seq_builder, [(args, False)])[0]
        # with mp.Pool(1) as pool:
        #     result = pool.map(seq_builder, [(args, True)])[0]
    # sequence_data = pd.read_pickle('./data/sequence_data/' + args.model + 'log_seq_data' + '.pkl')


if __name__ == '__main__':
    train()
