import pandas as pd
import numpy as np
import pickle
import gc

def load_seq_as_np(argv):
    print("Loading Pandas data frame for sequencing")
    args, feature_list, f_idx, get_date = argv
    y_feature = False
    if f_idx >= len(feature_list):
        f_idx -= len(feature_list)
        y_feature = True

    if args.training_mode in ['train_init', 'train_final']:
        train_df = pd.read_pickle('./data/sequence_data/' + args.model + '_train_seq_data_' + str(args.in_seq_len) + '_' +
                      str(args.out_seq_len) + '.pkl')
    test_df = pd.read_pickle('./data/sequence_data/' + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
                              str(args.out_seq_len) + '.pkl')

    # print(sequence_data.head())
    if args.training_mode == 'train_init':
        train_sequence_data = train_df[train_df['date'] < args.validation_start_date]
        test_sequence_data = train_df[(train_df['date'] >= args.validation_start_date) & (train_df['date'] <
                                                                                            args.testing_start_date)]
    elif args.training_mode == 'train_final':
        train_sequence_data = train_df[train_df['date'] < args.testing_start_date]
        test_sequence_data = test_df[(test_df['date'] >= args.testing_start_date) & (test_df['date'] <
                                                                                            args.testing_end_date)]

    else:
        test_sequence_data = test_df[(test_df['date'] >= args.testing_start_date) & (test_df['date'] <
                                                                                            args.testing_end_date)]
    if get_date:
        train_f_idx = None
        test_f_idx = np.stack(test_sequence_data['date'].dt.strftime('%Y-%m-%d').tolist(), axis=0)
        return (train_f_idx, test_f_idx)

    test_f_idx = pd_to_numpy(test_sequence_data, feature_list, f_idx, y_feature)

    if args.training_mode in ['train_init', 'train_final']:
        train_f_idx = pd_to_numpy(train_sequence_data, feature_list, f_idx, y_feature)
        print(f"Starting train sequencing: {f_idx}")
        train_f_idx = numpy_obj_to_arr(train_f_idx, f_idx)
        print(f"Finished train sequencing: {f_idx}")
        del train_sequence_data
        del train_df

    print(f"Starting test sequencing: {f_idx}")
    test_f_idx = numpy_obj_to_arr(test_f_idx, f_idx)
    print(f"Finished test sequencing: {f_idx}")

    del test_sequence_data
    del test_df
    gc.collect()

    return (train_f_idx, test_f_idx)


def numpy_obj_to_arr(seq_f_idx, f_idx):
    batch_data_shape = list(seq_f_idx.shape)
    row_data_shape = list(seq_f_idx[0][0].shape)

    if len(row_data_shape) > 0:
        new_data = np.stack(seq_f_idx.ravel()).reshape(batch_data_shape[0], batch_data_shape[1], row_data_shape[0],
                                                       row_data_shape[1])
        # print(f"3D: {new_data.shape}")
    else:
        new_data = np.stack(seq_f_idx.ravel()).reshape(batch_data_shape[0], batch_data_shape[1], 1)
        # print(f"Scalar: {new_data.shape}")

    return new_data


def pd_to_numpy(seq_data, feature_list, f_idx, y_feature=False):
    if not y_feature:
        seq_np = np.stack(seq_data['x_seq_' + feature_list[f_idx]].tolist(), axis=0)
    else:
        seq_np = np.stack(seq_data['y_seq_' + feature_list[f_idx]].tolist(), axis=0)

    return seq_np