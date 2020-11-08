import torch
import torch.optim as optim
from trainer_utils.trainer import TorchTrainer
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

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
from data_utils.data_loader import ItemDataset

from trainer_utils.trainer import TorchTrainer
from networks.encoder import LSTMEncoder, ConvLSTMEncoder
from networks.decoder import LSTMDecoder, ConvLSTMDecoder
from networks.encoderdecoder import EncoderDecoderWrapper


def train():
    # Parse arguments and load data
    args = parse_args()

    # If new dataset is to be loaded and processed with scaling/norms etc, then
    # Create batches of input sequence and output sequence that needs to be predicted
    if args.load_data:
        with mp.Pool(1) as pool:
            result = pool.map(load_data, [args])[0]
        with mp.Pool(1) as pool:
            result = pool.map(seq_builder, [args])[0]
    elif args.sequence_data:
        with mp.Pool(1) as pool:
            result = pool.map(seq_builder, [args])[0]

    # Let's get the X_enc, X_dec, and y_target values for input to the training scheme.
    # ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
    feature_list = ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']

    sequence_data = pd.read_pickle('./data/sequence_data/' + args.model + '_seq_data' + '.pkl')

    if args.training_mode == 'train':
        train_sequence_data = sequence_data[sequence_data['date'] <= args.validation_start_date]
        testing_sequence_data = sequence_data[(sequence_data['date'] > args.validation_start_date) & (sequence_data['date'] <
                                                                                            args.testing_start_date)]

    else:
        train_sequence_data = sequence_data[sequence_data['date'] <= args.testing_start_date]
        testing_sequence_data = sequence_data[(sequence_data['date'] > args.testing_start_date) & (sequence_data['date'] <
                                                                                            args.testing_end_date)]
    # print("Testing data")
    # print(testing_sequence_data.head())
    # print(testing_sequence_data.tail())
    #
    # print("Train data")
    # print(train_sequence_data.head())
    # print(train_sequence_data.tail())

    train_dataset = ItemDataset(args, feature_list)
    train_dataset.load_sequence_data(train_sequence_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    testing_dataset = ItemDataset(args, feature_list)
    testing_dataset.load_sequence_data(testing_sequence_data)
    testing_dataloader = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=True,
                                       drop_last=False)

    (X_enc, X_dec), y = next(iter(train_dataloader))
    # Load all features to initialize models
    # Encoder Features
    # ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
    if args.model in ['1d', '1D']:
        if args.use_log_h:
            x_enc_features_1d = np.concatenate([X_enc[1], X_enc[3], X_enc[4], X_enc[5]], axis=-1)
            x_dec_features_1d = np.concatenate([X_enc[3], X_enc[4], X_enc[5]], axis=-1)
        else:
            x_enc_features_1d = np.concatenate([X_enc[0], X_enc[2], X_enc[3], X_enc[4], X_enc[5]], axis=-1)
            x_dec_features_1d = np.concatenate([X_enc[2], X_enc[3], X_enc[4], X_enc[5]], axis=-1)

        enc_n_features = np.shape(x_enc_features_1d[0, 0, :])
        dec_n_features = np.shape(x_dec_features_1d[0, 0, :])

        encoder = LSTMEncoder(args, enc_n_features)
        decoder = LSTMDecoder(args, dec_n_features)

    elif args.model in ['3d', '3D']:
        if args.use_log_h:
            x_enc_features_3d = (1, 1)
            x_dec_features_3d = (1, 1)
        else:
            # Use the yearly correlation along with it, so two channels
            x_enc_features_3d = (2, 1)
            x_dec_features_3d = (2, 1)

        encoder = ConvLSTMEncoder(args, x_enc_features_3d)
        decoder = ConvLSTMDecoder(args, x_dec_features_3d)



    else:
        assert "Incorrect Model Chosen"
        return

    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)
    model = EncoderDecoderWrapper(args, encoder, decoder)
    model.to(args.device)
    print(repr(model))

    encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-2)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-2)

    encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=1e-3,
                                                      steps_per_epoch=len(train_dataloader), epochs=6)
    decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=1e-3,
                                                      steps_per_epoch=len(train_dataloader), epochs=6)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)

    # Training Loop
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    batch_size = args.batch_size
    n_epochs = args.epoch

    trainer = TorchTrainer(
        'encdec_ohe_std_mse_wd1e-2_do2e-1_test_hs100_tf0_adam',
        model,
        [encoder_optimizer, decoder_optimizer],
        loss_fn,
        [encoder_scheduler, decoder_scheduler],
        args.device,
        scheduler_batch_step=True,
        pass_y=True
    )

    trainer.lr_find(train_dataloader, model_optimizer, start_lr=1e-5, end_lr=1e-2, num_iter=500)




if __name__ == '__main__':
    train()
