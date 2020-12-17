import pandas as pd
import numpy as np
import pickle
import gc
import csv

def load_seq_as_np(argv):
    print("Loading Pandas data frame for sequencing")
    args, feature_list, f_idx = argv
    y_feature = False
    if f_idx >= len(feature_list):
        f_idx -= len(feature_list)
        y_feature = True

    if args.training_mode == 'train_init':
        train_sequence_data = pd.read_pickle(
            './data/sequence_data/' + args.model + '_train_seq_data_' + str(args.in_seq_len) + '_' +
            str(args.out_seq_len) + '.pkl')
        # print(sequence_data.head())
        train_sequence_data = train_sequence_data[train_sequence_data['date'] <= args.validation_start_date]
        testing_sequence_data = train_sequence_data[(train_sequence_data['date'] > args.validation_start_date) & (train_sequence_data['date'] <
                                                                                            args.testing_start_date)]
    elif args.training_mode == 'train_final':
        train_sequence_data = pd.read_pickle(
            './data/sequence_data/' + args.model + '_train_seq_data_' + str(args.in_seq_len) + '_' +
            str(args.out_seq_len) + '.pkl')
        # print(sequence_data.head())
        train_sequence_data = train_sequence_data[train_sequence_data['date'] <= args.testing_start_date]
        test_sequence_data = pd.read_pickle(
            './data/sequence_data/' + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
            str(args.out_seq_len) + '.pkl')
        testing_sequence_data = test_sequence_data[(test_sequence_data['date'] > args.testing_start_date) & (test_sequence_data['date'] <
                                                                                            args.testing_end_date)]
    elif args.training_mode == 'test':
        test_sequence_data = pd.read_pickle(
            './data/sequence_data/' + args.model + '_test_seq_data_' + str(args.in_seq_len) + '_' +
            str(args.out_seq_len) + '.pkl')
        testing_sequence_data = test_sequence_data[
            (test_sequence_data['date'] > args.testing_start_date) & (test_sequence_data['date'] <
                                                                      args.testing_end_date)]
    test_f_idx = pd_to_numpy(testing_sequence_data, feature_list, f_idx, y_feature)
    train_sequence_data_f_idx = None
    if not args.training_mode == 'test':
        train_f_idx = pd_to_numpy(train_sequence_data, feature_list, f_idx, y_feature)
        print(f"Starting train convert to numpy: {f_idx}")
        train_sequence_data_f_idx = numpy_obj_to_arr(train_f_idx, f_idx)
        print(f"Finished train convert to numpy: {f_idx}")
        del train_sequence_data

    del testing_sequence_data
    gc.collect()
    print(f"Starting test convert to numpy: {f_idx}")
    testing_sequence_data_f_idx = numpy_obj_to_arr(test_f_idx, f_idx)
    print(f"Finished test convert to numpy: {f_idx}")

    return (train_sequence_data_f_idx, testing_sequence_data_f_idx)
    # return train_sequence_data, testing_sequence_data
    # for f in feature_list:
    #     try:
    #         X.append(train_sequence_data['x_seq_' + f].to_numpy()[0])
    #     except:
    #         # print(np.stack(row['x_seq_' + f].to_numpy()[0], axis=0).shape)
    #         X.append(np.stack(train_sequence_data['x_seq_' + f].to_numpy()[0], axis=0))
    #     try:
    #         y.append(train_sequence_data['y_seq_' + f].to_numpy())
    #     except:
    #         y.append(np.stack(train_sequence_data['y_seq_' + f].to_numpy()[0], axis=0))

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

    #
    # # n_features = feature_list
    # for idx in range(0, len_data - 1):
    #     if idx % 500 == 0: print(f"Feature: {f_idx}, Seq: {seq_f_idx.shape}, Batch Index: {idx}")
    #     if idx == 0:
    #         if f_idx == 0:
    #             print(f'Input: {seq_f_idx.shape}, Ravel: {seq_f_idx.ravel().shape}')
    #             x = np.stack(seq_f_idx.ravel()).reshape(seq_f_idx.shape[0], seq_f_idx.shape[1], 100, 100)
    #             print(x.shape)
    #         # print(f'New data being added: {np.shape(np.stack(data[idx]))}')
    #         # X_values
    #
    #
    #         # x = np.stack(x.ravel().reshape())
    #         break
    #         new_data = np.stack([np.stack(seq_f_idx[idx]), np.stack(seq_f_idx[idx+1])], axis=0)
    #         # print(f"Data Shape: {new_data.shape}, input: {seq_f_idx[idx+1].shape}")
    #         # new_data[1] = np.stack([np.stack(y_f_idx[idx]), np.stack(y_f_idx[idx + 1])], axis=0)
    #         # print(f'New array: {np.shape(new_data[f])}')
    #     else:
    #         # print(f'New data being added: {np.shape(np.expand_dims(np.stack(data[idx]), axis=0))}')
    #         # print(f'Not zero index: {np.shape(new_data[f])}')
    #         new_data = np.concatenate([new_data, np.expand_dims(np.stack(seq_f_idx[idx + 1]), axis=0)], axis=0)
    #         # print(f"Data Shape on concat: {new_data.shape}, input: {seq_f_idx[0][0].shape}")
    #         # new_data[1] = np.concatenate([new_data[1], np.expand_dims(np.stack(y_f_idx[idx + 1]), axis=0)],
    #         #                              axis=0)
    #         # print(f'After adding: {np.shape(new_data[f])}')
    #
    # # print(f"F Index: {f_idx}, Final Data Shape: {new_data.shape}")

    return new_data


def pd_to_numpy(seq_data, feature_list, f_idx, y_feature=False):
    if not y_feature:
        seq_np = np.stack(seq_data['x_seq_' + feature_list[f_idx]].tolist(), axis=0)
    else:
        seq_np = np.stack(seq_data['y_seq_' + feature_list[f_idx]].tolist(), axis=0)

    return seq_np


# Reshape to feed to Matlab
def np_to_csv(y_pred, y_target, y_date, args):
    # Flatten the 3D data, and unroll over the whole sequence
    seq_len = y_pred.shape[1]

    y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    y_target = y_pred.reshape(y_target.shape[0], y_target.shape[1], -1)
    y_target = y_pred.reshape(y_target.shape[0], -1)
    y_date = y_date[:, :, 0]
    y_date = y_date.reshape(y_date.shape[0], -1)

    y_final = np.concatenate((y_date, y_pred, y_target), axis=-1)

    y_pred_f_t = []
    y_target_f_t = []
    date_f_t = []

    for t in range(seq_len):
        y_pred_f_t += ['h_pred_' + str(f) + '_' + str(t) for f in range(args.num_features)]
        y_target_f_t += ['h_target_' + str(f) + '_' + str(t)for f in range(args.num_features)]
        date_f_t += ['date_' + str(t)]

    pred_features = date_f_t + y_pred_f_t + y_target_f_t

    with open('./data/prediction_data/' + args.model + '_predict_data_' + args.predict_run + '.csv', 'w+') as pred_csv:
        csvWriter = csv.writer(pred_csv, delimiter=',', lineterminator='\n')
        csvWriter.writerow(pred_features)
        csvWriter.writerows(y_final)
