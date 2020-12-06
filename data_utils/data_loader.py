import torch
from torch.utils.data import Dataset, IterableDataset
import gc
import numpy as np
import h5py

class ItrDataset(IterableDataset):
    def __init__(self, args, feature_list):
        super().__init__()
        self.args = args
        self.sequence_data_filename = None
        self.feature_list = feature_list
        self.total_feature = len(self.feature_list)

    def load_sequence_data(self, seq_data_filename):
        self.sequence_data_filename = seq_data_filename

    def create_feature_shapes(self, data):
        X = []
        y = []
        for f in range(self.total_feature):
            # print(f'Idx: {idx}: Feature: {self.sequence_data[f].shape}, Final: {self.sequence_data[f][idx].shape}')
            print(data.shape)
            print(data[f].shape)
            X.append(torch.tensor(data[f], dtype=torch.float32))
            # print(f'Y_Idx: {idx}: Feature: {self.sequence_data[f + total_feature].shape}, Final: {self.sequence_data[f + total_feature][idx].shape}')
            y.append(torch.tensor(data[f + self.total_feature], dtype=torch.float32))
            # try:
            #     y.append(torch.tensor(row['y_seq_' + f].to_numpy(), dtype=torch.float32))
            # except:
            #     y.append(torch.tensor(np.stack(row['y_seq_' + f].to_numpy()[0], axis=0), dtype=torch.float32))
                # print(f"Failed for Y: {f}")
                # print(row['y_seq_' + f].to_numpy().shape)
                # print(row['y_seq_' + f].to_numpy(dtype=float)[0][0].shape)
        # Single batch size
        X_dec = []
        if self.args.use_add_features or self.args.use_yr_corr:
            num_dec_features = len(y)
            X_dec = [y[i] for i in range(1, num_dec_features)]

        return (X, X_dec), y[0]

    def __iter__(self):
        # seq_itr = open(self.sequence_data_filename, 'rb')
        seq_itr = h5py.File(self.sequence_data_filename, 'r')
        map_seq_itr = map(self.create_feature_shapes, seq_itr)

        return map_seq_itr


class ItemDataset(Dataset):
    def __init__(self, args, feature_list, data_type='train'):
        super().__init__()
        self.args = args
        self.sequence_data = None
        self.feature_list = feature_list
        self.total_feature = len(feature_list)
        self.data_type = data_type

    def load_sequence_data(self, seq_data):
        self.sequence_data = h5py.File(seq_data, 'r')
        self.length = self.sequence_data.get(self.data_type + str(0)).shape[0]
        # x_f = np.array(self.sequence_data.get(self.data_type + str(0)))[0]
        # for f in self.sequence_data:
        #     print(f'Total: {f.shape}, Idx: {f[0].shape}')

    def get_feature_shapes(self):
        return

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # row = self.sequence_data.iloc[[idx]]
        # Features in seq_data
        # ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
        X = []
        y = []
        # total_feature = len(self.feature_list)
        for f in range(self.total_feature):
            # print(f'Idx: {idx}: Feature: {self.sequence_data[f].shape}, Final: {self.sequence_data[f][idx].shape}')
            # x_f = np.array(self.sequence_data.get(self.data_type + str(f)))[idx]
            x_f = self.sequence_data[self.data_type + str(f)][idx]
            # print(np.shape(x_f))
            X.append(torch.tensor(x_f, dtype=torch.float32))
            # print(f'Y_Idx: {idx}: Feature: {self.sequence_data[f + total_feature].shape}, Final: {self.sequence_data[f + total_feature][idx].shape}')
            # y_f = np.array(self.sequence_data.get(self.data_type + str(f + self.total_feature)))[idx]
            y_f = self.sequence_data[self.data_type + str(f + self.total_feature)][idx]
            # print(np.shape(y_f))
            y.append(torch.tensor(y_f, dtype=torch.float32))
            # try:
            #     y.append(torch.tensor(row['y_seq_' + f].to_numpy(), dtype=torch.float32))
            # except:
            #     y.append(torch.tensor(np.stack(row['y_seq_' + f].to_numpy()[0], axis=0), dtype=torch.float32))
                # print(f"Failed for Y: {f}")
                # print(row['y_seq_' + f].to_numpy().shape)
                # print(row['y_seq_' + f].to_numpy(dtype=float)[0][0].shape)
        # Single batch size
        X_dec = []
        if self.args.use_add_features or self.args.use_yr_corr:
            num_dec_features = len(y)
            X_dec = [y[i] for i in range(1, num_dec_features)]

        # Extract features from the tensor and put it along
        # if self.args.use_log_h:
        #     # Discard: h_in, h_yearl_corr
        #     X_enc = []
        #     X_enc = [X[:,i] for i in range(6) if i not in [0, 2]]
        #     # Only use the time-dependent features
        #     X_dec = [y[:, i] for i in range(6) if i not in [0, 1, 2]]
        #     y = y[:, 1]
        # else:
        #     # Discard: h_in, h_yearl_corr
        #     # Discard: h_in, h_yearl_corr
        #     X_enc = [X[:, i] for i in range(6) if i is not 1]
        #     # Only use the time-dependent features
        #     X_dec = [y[:, i] for i in range(6) if i not in [0, 1, 2]]
        #     y = y[:, 0]
        # print(self.sequence_data.info('deep'))
        return (X, X_dec), y[0]


# class StoreItemDataset(Dataset):
#     def __init__(self, cat_columns=[], num_columns=[], embed_vector_size=None, decoder_input=True,
#                  ohe_cat_columns=False):
#         super().__init__()
#         self.sequence_data = None
#         self.cat_columns = cat_columns
#         self.num_columns = num_columns
#         self.cat_classes = {}
#         self.cat_embed_shape = []
#         self.cat_embed_vector_size = embed_vector_size if embed_vector_size is not None else {}
#         self.pass_decoder_input = decoder_input
#         self.ohe_cat_columns = ohe_cat_columns
#         self.cat_columns_to_decoder = False
#
#     def get_embedding_shape(self):
#         return self.cat_embed_shape
#
#     def load_sequence_data(self, processed_data):
#         self.sequence_data = processed_data
#
#     def process_cat_columns(self, column_map=None):
#         column_map = column_map if column_map is not None else {}
#         for col in self.cat_columns:
#             self.sequence_data[col] = self.sequence_data[col].astype('category')
#             if col in column_map:
#                 self.sequence_data[col] = self.sequence_data[col].cat.set_categories(column_map[col]).fillna('#NA#')
#             else:
#                 self.sequence_data[col].cat.add_categories('#NA#', inplace=True)
#             self.cat_embed_shape.append(
#                 (len(self.sequence_data[col].cat.categories), self.cat_embed_vector_size.get(col, 50)))
#
#     def __len__(self):
#         return len(self.sequence_data)
#
#     def __getitem__(self, idx):
#         row = self.sequence_data.iloc[[idx]]
#         x_inputs = [torch.tensor(row['x_sequence'].values[0], dtype=torch.float32)]
#         y = torch.tensor(row['y_sequence'].values[0], dtype=torch.float32)
#         if self.pass_decoder_input:
#             decoder_input = torch.tensor(row['y_sequence'].values[0][:, 1:], dtype=torch.float32)
#         if len(self.num_columns) > 0:
#             for col in self.num_columns:
#                 num_tensor = torch.tensor([row[col].values[0]], dtype=torch.float32)
#                 x_inputs[0] = torch.cat((x_inputs[0], num_tensor.repeat(x_inputs[0].size(0)).unsqueeze(1)), axis=1)
#                 decoder_input = torch.cat((decoder_input, num_tensor.repeat(decoder_input.size(0)).unsqueeze(1)),
#                                           axis=1)
#         if len(self.cat_columns) > 0:
#             if self.ohe_cat_columns:
#                 for ci, (num_classes, _) in enumerate(self.cat_embed_shape):
#                     col_tensor = torch.zeros(num_classes, dtype=torch.float32)
#                     col_tensor[row[self.cat_columns[ci]].cat.codes.values[0]] = 1.0
#                     col_tensor_x = col_tensor.repeat(x_inputs[0].size(0), 1)
#                     x_inputs[0] = torch.cat((x_inputs[0], col_tensor_x), axis=1)
#                     if self.pass_decoder_input and self.cat_columns_to_decoder:
#                         col_tensor_y = col_tensor.repeat(decoder_input.size(0), 1)
#                         decoder_input = torch.cat((decoder_input, col_tensor_y), axis=1)
#             else:
#                 cat_tensor = torch.tensor(
#                     [row[col].cat.codes.values[0] for col in self.cat_columns],
#                     dtype=torch.long
#                 )
#                 x_inputs.append(cat_tensor)
#         if self.pass_decoder_input:
#             x_inputs.append(decoder_input)
#             y = torch.tensor(row['y_sequence'].values[0][:, 0], dtype=torch.float32)
#         if len(x_inputs) > 1:
#             return tuple(x_inputs), y
#         return x_inputs[0], y