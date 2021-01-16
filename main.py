import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils.fast_tensor_data_loader import FastTensorDataLoader
import gc

import scipy.io

import os
import sys
from pathlib import Path
import csv

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import pickle
import h5py

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
from data_utils.sequence_builder import seq_builder, np_to_csv
from data_utils.seq_loader import load_seq_as_np
from data_utils.data_loader import ItemDataset, ItrDataset
from data_utils.data_postprocess import plot_surface

from trainer_utils.trainer import TorchTrainer
from networks.encoder import LSTMEncoder, ConvLSTMEncoder
from networks.decoder import LSTMDecoder, ConvLSTMDecoder
from networks.encoderdecoder import EncoderDecoderWrapper
from networks.encoderdecoder2 import EncoderDecoderWrapper3d

torch.manual_seed(420)
np.random.seed(420)

def dir_checks(args):
    process_dir = Path('./data/processed_data')
    if not process_dir.exists():
        os.makedirs(process_dir)

    seq_dir = Path('./data/sequence_data/numpy/')
    if not seq_dir.exists():
        os.makedirs(seq_dir)

    pred_dir = Path('./data/prediction_data')
    if not pred_dir.exists():
        os.makedirs(pred_dir)

    train_file = Path('./data/sequence_data/numpy/' + args.model + '_train_seq_data_' + str(args.in_seq_len) + '_' +
                      str(args.out_seq_len) + '.h5')
    test_file = Path('./data/sequence_data/numpy/' + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
                     str(args.out_seq_len) + '.h5')

    if train_file.exists():
        print("Train File Seq Exists")
        sys.exit()

    if test_file.exists():
        print("Test File Seq Exists")
        sys.exit()

    return

def train():
    # Parse arguments and load data
    args = parse_args()
    load_train_data = False
    if args.lr_search or args.train_network:
        load_train_data = True
    if args.load_data or args.sequence_data or args.sequence_to_np:
        dir_checks(args)

    # If new dataset is to be loaded and processed with scaling/norms etc, then
    # Create batches of input sequence and output sequence that needs to be predicted
    # feature_list = ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
    feature_list = []
    if args.use_log_h:
        feature_list.append('log_h_in')
    else:
        feature_list.append('h_in')
        if args.use_yr_corr:
            feature_list.append('h_yearly_corr')

    if args.use_add_features:
        feature_list += ['day_of_year_cos', 'day_of_year_sin', 'year_mod']

    if args.load_data:
        with mp.Pool(12) as pool:
            result = pool.map(load_data, [args])[0]
        # with mp.Pool(12) as pool:
        #     result = pool.map(seq_builder, [(args, feature_list)])[0]
    if args.sequence_data:
        with mp.Pool(12) as pool:
            result = pool.map(seq_builder, [(args, feature_list, 'train')])[0]
        with mp.Pool(12) as pool:
            result = pool.map(seq_builder, [(args, feature_list, 'test')])[0]
    if args.sequence_to_np:
        # seq_np_data = []
        pool = mp.Pool(12)
        train_hf = h5py.File(
            './data/sequence_data/numpy/' + args.model + '_train_seq_data_' + str(args.in_seq_len) + '_' +
            str(args.out_seq_len) + '.h5', 'w')
        test_hf = h5py.File(
            './data/sequence_data/numpy/' + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
            str(args.out_seq_len) + '.h5', 'w')

        # train_seq, test_seq = [], []
        for f in range(2 * len(feature_list)):
            print(f"Working with {f}")
            result = pool.map(load_seq_as_np, [(args, feature_list, f, False)])[0]
            train_seq_f, test_seq_f = result
            train_hf.create_dataset('train' + str(f), data=np.asarray(train_seq_f))
            test_hf.create_dataset('test' + str(f), data=np.asarray(test_seq_f))
            # train_seq.append(train_seq_f)
            # test_seq.append(test_seq_f)

        result = pool.map(load_seq_as_np, [(args, feature_list, f, True)])[0]
        _, test_seq_date = result
        test_hf.create_dataset('testdate', data=test_seq_date.astype('S10'))

        train_hf.close()
        test_hf.close()

        # with open('./data/sequence_data/numpy/' + args.model + '_train_seq_data_' + str(args.in_seq_len) + '_' +
        #           str(args.out_seq_len) + '.pkl', 'wb+') as fp:
        #     pickle.dump(train_seq, fp)
        #
        # with open('./data/sequence_data/numpy/' + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
        #           str(args.out_seq_len) + '.pkl', 'wb+') as fp:
        #     pickle.dump(test_seq, fp)
        pool.close()
        del result
        del train_seq_f
        del test_seq_f
        gc.collect()

    # Let's get the X_enc, X_dec, and y_target values for input to the training scheme.
    # ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
    # train_sequence_data, testing_sequence_data = [], []
    # if load_train_data:
    #     with open('./data/sequence_data/numpy/' + args.model + '_train_seq_data_' + str(args.in_seq_len) + '_' +
    #                       str(args.out_seq_len) + '.pkl', 'rb') as fp:
    #         try:
    #             while True:
    #                 train_sequence_data.append(pickle.load(fp))
    #         except EOFError:
    #             pass

    # with open('./data/sequence_data/numpy/' + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
    #                   str(args.out_seq_len) + '.pkl', 'rb') as fp:
    #     try:
    #         while True:
    #             testing_sequence_data.append(pickle.load(fp))
    #     except EOFError:
    #         pass

    # testing_dataset = ItrDataset(args, feature_list)
    testing_dataset = ItemDataset(args, feature_list, 'test')
    testing_dataset.load_sequence_data('./data/sequence_data/numpy/' + args.model + '_test_seq_data_'
                                       + str(args.in_seq_len) + '_' + str(args.out_seq_len) + '.h5')
    # testing_dataset.load_sequence_data(test_np)
    testing_dataloader = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=10)

    if load_train_data:
        # Load into memory
        train_dataset = ItemDataset(args, feature_list, 'train')
        # train_dataset.load_sequence_data(train_sequence_data)
        # Use iteratable type
        # train_dataset = ItrDataset(args, feature_list)
        train_dataset.load_sequence_data('./data/sequence_data/numpy/' + args.model + '_train_seq_data_'
                                         + str(args.in_seq_len) + '_' + str(args.out_seq_len) + '.h5')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=10)

        (X_enc, X_dec), y = next(iter(train_dataloader))
    else:
        (X_enc, X_dec), y = next(iter(testing_dataloader))

    # Load all features to initialize models
    # Encoder Features
    # ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
    if args.model in ['3d', '3D']:
        c = 1
        t = 1

        if args.use_yr_corr:
            # Use the yearly correlation along with it, so two channels
            c = 2
            t = 1

        print(X_enc[0].shape)
        h, w = X_enc[0][0, 0, :, :].shape

        add_lin_dim = 0
        if args.use_add_features:
            # Add day of year and year features
            add_lin_dim = np.shape(np.concatenate([X_dec[1], X_dec[2], X_dec[3]], axis=-1)[0, 0, :])[0]
            # print(add_lin_dim)

        x_enc_features_3d = (c, t, h, w, add_lin_dim)
        x_dec_features_3d = (c, t, h, w, add_lin_dim)
        x_features = x_dec_features_3d
    else:
        if args.use_log_h:
            x_enc_features_1d = np.concatenate([X_enc[1], X_enc[3], X_enc[4], X_enc[5]], axis=-1)
            x_dec_features_1d = np.concatenate([X_dec[3], X_dec[4], X_dec[5]], axis=-1)
        else:
            x_enc_features_1d = np.concatenate([X_enc[0], X_enc[2], X_enc[3], X_enc[4], X_enc[5]], axis=-1)
            x_dec_features_1d = np.concatenate([X_dec[2], X_dec[3], X_dec[4], X_dec[5]], axis=-1)

        enc_n_features = np.shape(x_enc_features_1d[0, 0, :])
        dec_n_features = np.shape(x_dec_features_1d[0, 0, :])
        x_features = None

    if args.use3d_autoencoder and args.model in ['3d', '3D']:
        model = EncoderDecoderWrapper3d(args, None, None, feature_list, x_features)
    else:
        if args.model in ['3d', '3D']:
            encoder = ConvLSTMEncoder(args, x_enc_features_3d)
            decoder = ConvLSTMDecoder(args, x_dec_features_3d)
        else:
            encoder = LSTMEncoder(args, enc_n_features)
            decoder = LSTMDecoder(args, dec_n_features)

        encoder = encoder.to(args.device)
        decoder = decoder.to(args.device)
        model = EncoderDecoderWrapper(args, encoder, decoder, feature_list, x_features)

    model.to(args.device)
    print(repr(model))

    n_epochs = args.epoch
    if args.train_network:
        steps_epoch = len(train_dataloader)
    else:
        steps_epoch = 10

    # Training Loop
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()

    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # model_scheduler = optim.lr_scheduler.OneCycleLR(model_optimizer, max_lr=1e-3,
    #                                                       steps_per_epoch=steps_epoch, epochs=n_epochs)

    if not args.use3d_autoencoder:
        encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.lr_decay)
        decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.lr_decay)
        # if load_train_data:
        encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=1e-2,
                                                          steps_per_epoch=steps_epoch, epochs=n_epochs)
        decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=1e-2,
                                                          steps_per_epoch=steps_epoch, epochs=n_epochs)

        optimizers = [encoder_optimizer, decoder_optimizer]
        schedulers = [encoder_scheduler, decoder_scheduler]
    else:
        optimizers = [model_optimizer]
        schedulers = []

    trainer = TorchTrainer(
        args.exp_name,
        model,
        optimizers,
        loss_fn,
        schedulers,
        args.device,
        scheduler_batch_step=True,
        pass_y=False
    )

    if args.lr_search and load_train_data:
        trainer.lr_find(train_dataloader, model_optimizer, start_lr=1e-5, end_lr=1e-2, num_iter=500)

    if args.train_network and load_train_data:
        trainer.train(n_epochs, train_dataloader, testing_dataloader, resume_only_model=True, resume=True)

    print("Loading Prediction and Plot")
    trainer._load_checkpoint(only_model=True)

    test_predictions, target_values = trainer.predict(testing_dataloader, n_samples=args.n_samples, plot_phase=True)
    test_predictions_mean = np.mean(test_predictions, axis=1)
    test_predictions_std = np.std(test_predictions, axis=1)

    # seq_len = test_predictions.shape[1]
    # batch_idx = np.random.randint(low=0, high=test_predictions.shape[0], size=1)[0]
    # print(f'{seq_len}, {test_predictions.shape[0]},{batch_idx}')
    # print(f'{test_predictions.shape}, {test_predictions[0].shape}, {test_predictions[0][0].shape}')
    # print(f'{target_values.shape}, {target_values[0].shape}, {target_values[0][0].shape}')
    # save_target = []
    # for i in range(seq_len):
    #     # Take mean when using bayesian inference
    #     # print(np.shape(test_predictions[batch_idx]))
    #     # z_pred_mean = np.mean(test_predictions[batch_idx], axis=0)
    #     # z_pred_std = np.std(test_predictions[batch_idx], axis=0)
    #     # print(np.shape(z_pred))
    #     # print(z_pred.shape)
    #     plot_surface(test_predictions_mean[batch_idx][i], title=f"Predict Mean: t{i}")
    #     # plot_surface(z_pred_std[i], title=f"Predict Std: t{i}")
    #     plot_surface(target_values[batch_idx][i], title=f'{batch_idx}: Target: t{i}')
    #     # save_target.append(target_values[batch_idx][i])

    # with open('./data/sequence_data/numpy/' + args.model + '_y_label_seq_data_' + str(args.in_seq_len) + '_' +
    #           str(args.out_seq_len) + '.pkl', 'wb+') as fp:
    #     pickle.dump(np.stack(save_target), fp)
    #
    # label_3d = pickle.load(open('./data/sequence_data/numpy/' + args.model + '_y_label_seq_data_' + str(args.in_seq_len) + '_' +
    #           str(args.out_seq_len) + '.pkl', 'rb+'))
    #
    # scipy.io.savemat('./data/sequence_data/numpy/' + args.model + '_y_label_seq_data_' + str(args.in_seq_len) + '_' +
    #           str(args.out_seq_len) + '.mat', mdict={'Target100x100': label_3d})
    test_hf = h5py.File('./data/sequence_data/numpy/' + args.model + '_test_seq_data_'
                                       + str(args.in_seq_len) + '_' + str(args.out_seq_len) + '.h5', 'r')
    target_dates = test_hf.get('testdate')
    target_dates = np.asarray(target_dates).reshape(-1, 1)

    np_to_csv(test_predictions_mean, test_predictions_std, target_values, target_dates, args)
    test_hf.close()

if __name__ == '__main__':
    train()
