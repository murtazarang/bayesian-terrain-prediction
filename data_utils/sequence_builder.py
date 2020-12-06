import tqdm
import numpy as np
import pandas as pd

import multiprocessing as mp
from functools import partial
from collections import deque

import sys


# Load and Save Sequence to Pickle
def seq_builder(argv):
    args, feature_list = argv
    # feature_list = ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
    if args.model in ['1D', '1d']:
        scaled_data = pd.read_pickle('./data/processed_data/' + args.model + '_train_processed_data' + '.pkl')
    elif args.model in ['3D', '3d']:
        scaled_data = pd.read_pickle('./data/processed_data/' + args.model + '_train_processed_data' + '.pkl')
    else:
        assert "Incorrect model"
        sys.exit()

    # height_list = ["h" + str(i + 1) for i in range(args.num_features)]
    # height_yearly_corr_list = [h + '_yearly_corr' for h in height_list]
    # log_height_list = ["log_h" + str(i + 1) for i in range(args.num_features)]

    print("Starting Sequencing Inputs")
    if args.model in ['1d', '1D']:
        if args.use_log_h:
            scaled_data['log_h_in'] = scaled_data['log_h1']
        else:
            scaled_data['h_in'] = scaled_data['h1']
        if args.use_yr_corr:
            scaled_data['h_yearly_corr'] = scaled_data['h1_yearly_corr']

        drop_features_list = [h for h in list(scaled_data.columns)
                            if h not in feature_list + ['date']]

    elif args.model in ['3d', '3D']:
        if args.use_log_h:
            log_height_list = ["log_h" + str(i + 1) for i in range(args.num_features)]
            # Produces [num_features, data_length]
            h_log_aggr_list = np.array([np.array(scaled_data[h]) for h in log_height_list])
            # Change to (data_len, num_features) and then move to 3D
            h_log_aggr_list = np.swapaxes(h_log_aggr_list, 1, 0)
            h_log_aggr_list = np.reshape(h_log_aggr_list, (-1, args.xdim, args.ydim))
            h_log_aggr_list = h_log_aggr_list
            # print(np.shape(h_log_aggr_list))
            h_log_aggr_list = list(h_log_aggr_list)
            # print(np.shape(h_log_aggr_list[0]))
            scaled_data['log_h_in'] = h_log_aggr_list

            # drop_features_list = [h for h in list(scaled_data.columns)
            #                       if h not in ['h_in', 'log_h_in', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']]

            # Reshape to 3D Space for all features
            # Normalized Height
        else:
            height_list = ["h" + str(i + 1) for i in range(args.num_features)]  # This is already scaled
            h_aggr_list = np.array([np.array(scaled_data[h]) for h in height_list])
            # Change to (data_len, num_features) and then move to 3D
            h_aggr_list = np.swapaxes(h_aggr_list, 1, 0)
            h_aggr_list = np.reshape(h_aggr_list, (-1, args.xdim, args.ydim))
            h_aggr_list = list(h_aggr_list)
            scaled_data['h_in'] = h_aggr_list

            if args.use_yr_corr:
                # Yearly Correlation
                height_yearly_corr = [h + '_yearly_corr' for h in height_list]
                h_corr_aggr_list = np.array([np.array(scaled_data[h_corr]) for h_corr in height_yearly_corr])
                # Change to (data_len, num_features) and then move to 3D
                h_corr_aggr_list = np.swapaxes(h_corr_aggr_list, 1, 0)
                h_corr_aggr_list = np.reshape(h_corr_aggr_list, (-1, args.xdim, args.ydim))
                h_corr_aggr_list = list(h_corr_aggr_list)
                scaled_data['h_yearly_corr'] = h_corr_aggr_list

        drop_features_list = [h for h in list(scaled_data.columns)
                        if h not in feature_list + ['date']]
    else:
        return

    scaled_data.drop(drop_features_list, axis=1, inplace=True)
    print(f'Scaled and Reshaped Features: \n {scaled_data.head()} \n \n')

    # Create the sliding windows
    X_seq, y_seq = create_sliding_win(args, scaled_data, feature_list)

    sequence_data = pd.DataFrame()
    for i, f in enumerate(feature_list):
        sequence_data['x_seq_' + f] = X_seq[i]
    # sequence_data['x_seq'] = X_seq
        sequence_data['y_seq_' + f] = y_seq[i]
    sequence_data['date'] = scaled_data['date']
    print(f'Sequenced Features: \n {sequence_data.head()} \n \n')

    sequence_data.to_pickle('./data/sequence_data/' + args.model + '_seq_data_' + str(args.in_seq_len) + '_' +
                  str(args.out_seq_len) + '.pkl')


# Make sure h_in is the first
def create_sliding_win(args, data, feature_list, stride=1):
    X_list = [[] for _ in range(len(feature_list))]
    y_list = [[] for _ in range(len(feature_list))]
    # Calculate the number of steps across the complete dataset
    steps = list(range(0, len(data), stride))
    # feature_list = ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']

    for i in steps:
        # find the end of this pattern
        end_ix = i + args.in_seq_len
        out_end_ix = end_ix + args.out_seq_len
        # check if we are beyond the dataset
        if out_end_ix > len(data):
            break
        # [rows/steps, #features]
        for j, f in enumerate(feature_list):
            X_list[j].append(data.iloc[i:end_ix][f].values)
            y_list[j].append(data.iloc[end_ix:out_end_ix][f].values)

    return X_list, y_list


# Usage
# x = list(list(x) for x in sliding_window(y))
# def sliding_window(seq, n=4):
#     it = iter(seq)
#     win = deque((next(it, None) for _ in range(n)), maxlen=n)
#     yield win
#     append = win.append
#     for e in it:
#         append(e)
#         yield win
