import pandas as pd
import numpy as np
import pickle
import gc

def load_seq_as_np(argv):
    print("Loading Pandas data frame for sequencing")
    args, feature_list, f_idx = argv
    y_feature = False
    if f_idx >= len(feature_list):
        f_idx -= len(feature_list)
        y_feature = True

    sequence_data = pd.read_pickle('./data/sequence_data/' + args.model + '_seq_data_' + str(args.in_seq_len) + '_' +
                  str(args.out_seq_len) + '.pkl')
    # print(sequence_data.head())
    if args.training_mode == 'train':
        train_sequence_data = sequence_data[sequence_data['date'] <= args.validation_start_date]
        testing_sequence_data = sequence_data[(sequence_data['date'] > args.validation_start_date) & (sequence_data['date'] <
                                                                                            args.testing_start_date)]
    else:
        train_sequence_data = sequence_data[sequence_data['date'] <= args.testing_start_date]
        testing_sequence_data = sequence_data[(sequence_data['date'] > args.testing_start_date) & (sequence_data['date'] <
                                                                                            args.testing_end_date)]
    train_f_idx = pd_to_numpy(train_sequence_data, feature_list, f_idx, y_feature)
    test_f_idx = pd_to_numpy(testing_sequence_data, feature_list, f_idx, y_feature)

    del train_sequence_data
    del testing_sequence_data
    del sequence_data
    gc.collect()

    print(f"Starting train sequencing: {f_idx}")
    train_sequence_data_f_idx = numpy_obj_to_arr(train_f_idx, f_idx)
    print(f"Finished train sequencing: {f_idx}")
    testing_sequence_data_f_idx = numpy_obj_to_arr(test_f_idx, f_idx)
    print(f"Finished testing sequencing: {f_idx}")
    # with open('./data/sequence_data/numpy/' + args.model + '_train_seq_data' + '.pkl', 'wb') as fp:
    #     pickle.dump(train_sequence_data, fp)
    #
    # with open('./data/sequence_data/numpy/' + args.model + '_test_seq_data' + '.pkl', 'wb') as fp:
    #     pickle.dump(testing_sequence_data, fp)

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