import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from collections import deque


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def feature1d_builder(data, args):
    if args.use_log_h is True:
        data['log_h_final'] = data['log_h1']
    else:
        data['h_final'] = data['h1']
        data['h_final_yearly_corr'] = data['h1_yearly_corr']
    return


def feature3d_builder(data, args):
    if args.use_log_h is True:
        log_height_list = ["log_h" + str(i + 1) for i in range(args.num_features)]
        # Produces [num_features, data_length]
        h_log_aggr_list = np.array([np.array(data[h]) for h in log_height_list])
        # Change to (data_len, num_features) and then move to 3D
        h_log_aggr_list = np.swapaxes(h_log_aggr_list, 1, 0)
        h_log_aggr_list = np.reshape(h_log_aggr_list, (-1, args.xdim, args.ydim))
        h_log_aggr_list = list(h_log_aggr_list)
        data['log_h_final'] = h_log_aggr_list

    else:
        # Reshape to 3D Space for all features
        # Normalized Height
        height_list = ["h" + str(i + 1) for i in range(args.num_features)]  # This is already scaled
        height_yearly_corr = [h + '_yearly_corr' for h in height_list]
        h_aggr_list = np.array([np.array(data[h]) for h in height_list])
        # Change to (data_len, num_features) and then move to 3D
        h_aggr_list = np.swapaxes(h_aggr_list, 1, 0)
        h_aggr_list = np.reshape(h_aggr_list, (-1, args.xdim, args.ydim))
        h_aggr_list = list(h_aggr_list)
        data['h_final'] = h_aggr_list

        # Yearly Correlation
        h_corr_aggr_list = np.array([np.array(data[h_corr]) for h_corr in height_yearly_corr])
        # Change to (data_len, num_features) and then move to 3D
        h_corr_aggr_list = np.swapaxes(h_corr_aggr_list, 1, 0)
        h_corr_aggr_list = np.reshape(h_corr_aggr_list, (-1, args.xdim, args.ydim))
        h_corr_aggr_list = list(h_corr_aggr_list)
        print(len(h_corr_aggr_list))
        data['h_final_yearly_corr'] = h_corr_aggr_list

    return


# Load and Save Sequence to Pickle
def seq_builder(args, use_log_h):
    if args.model in ['1D', '1d']:
        scaled_data = pd.read_pickle('./data/processed_data/' + args.model + '_train_processed_data' + '.pkl')
    elif args.model in ['3D', '3d']:
        scaled_data = pd.read_pickle('./data/processed_data/' + args.model + '_train_processed_data' + '.pkl')
    else:
        assert "Incorrect model"
        return

    if args.compress_data:
        scaled_data = reduce_mem_usage(scaled_data)

    height_list = ["h" + str(i + 1) for i in range(args.num_features)]
    height_yearly_corr_list = [h + '_yearly_corr' for h in height_list]
    log_height_list = ["log_h" + str(i + 1) for i in range(args.num_features)]

    if args.model in ['1d', '1D']:
        if use_log_h:
            scaled_data['h_in'] = scaled_data['log_h1']
            drop_features_list = [h for h in list(scaled_data.columns)
                                  if h not in ['h_in', 'day_of_year_cos', 'day_of_year_sin', 'year_mod', 'date']]
        else:
            scaled_data['h_in'] = scaled_data['h1']
            scaled_data['h_yearly_corr'] = scaled_data['h1_yearly_corr']
            drop_features_list = [h for h in list(scaled_data.columns)
                            if h not in ['h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod',
                                         'date']]

    elif args.model in ['3d', '3D']:
        if use_log_h:
            log_height_list = ["log_h" + str(i + 1) for i in range(args.num_features)]
            # Produces [num_features, data_length]
            h_log_aggr_list = np.array([np.array(scaled_data[h]) for h in log_height_list])
            # Change to (data_len, num_features) and then move to 3D
            h_log_aggr_list = np.swapaxes(h_log_aggr_list, 1, 0)
            h_log_aggr_list = np.reshape(h_log_aggr_list, (-1, args.xdim, args.ydim))
            h_log_aggr_list = list(h_log_aggr_list)
            scaled_data['h_in'] = h_log_aggr_list

            drop_features_list = [h for h in list(scaled_data.columns)
                                  if h not in ['h_in', 'day_of_year_cos', 'day_of_year_sin', 'year_mod', 'date']]

        else:
            # Reshape to 3D Space for all features
            # Normalized Height
            height_list = ["h" + str(i + 1) for i in range(args.num_features)]  # This is already scaled
            height_yearly_corr = [h + '_yearly_corr' for h in height_list]
            h_aggr_list = np.array([np.array(scaled_data[h]) for h in height_list])
            # Change to (data_len, num_features) and then move to 3D
            h_aggr_list = np.swapaxes(h_aggr_list, 1, 0)
            h_aggr_list = np.reshape(h_aggr_list, (-1, args.xdim, args.ydim))
            h_aggr_list = list(h_aggr_list)
            scaled_data['h_in'] = h_aggr_list

            # Yearly Correlation
            h_corr_aggr_list = np.array([np.array(scaled_data[h_corr]) for h_corr in height_yearly_corr])
            # Change to (data_len, num_features) and then move to 3D
            h_corr_aggr_list = np.swapaxes(h_corr_aggr_list, 1, 0)
            h_corr_aggr_list = np.reshape(h_corr_aggr_list, (-1, args.xdim, args.ydim))
            h_corr_aggr_list = list(h_corr_aggr_list)
            scaled_data['h_yearly_corr'] = h_corr_aggr_list
            drop_features_list = [h for h in list(scaled_data.columns)
                            if h not in ['h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod',
                                        'date']]

    else:
        return

    scaled_data.drop(drop_features_list, axis=1, inplace=True)

    # Now let's break down to Input and Output Sequences
    # Manually calculate the last cut-off based on the last times and use deque to get the input and target values



    norm_sequence_data.to_pickle('./data/sequence_data/' + args.model + '_corr_seq_data' + '.pkl')
    log_sequence_data.to_pickle('./data/sequence_data/' + args.model + '_log_seq_data' + '.pkl')