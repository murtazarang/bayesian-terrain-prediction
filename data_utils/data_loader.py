import torch
from torch.utils.data import Dataset

import numpy as np


class ItemDataset(Dataset):
    def __init__(self, args, feature_list):
        super().__init__()
        self.args = args
        self.sequence_data = None
        self.feature_list = feature_list

    def load_sequence_data(self, seq_data):
        self.sequence_data = seq_data

    def get_feature_shapes(self):
        return

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        row = self.sequence_data.iloc[[idx]]
        # Features in seq_data
        # ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
        X = []
        y = []
        for f in self.feature_list:
            try:
                X.append(torch.tensor(row['x_seq_' + f].to_numpy()[0], dtype=torch.float32))
            except:
                # print(np.stack(row['x_seq_' + f].to_numpy()[0], axis=0).shape)
                X.append(torch.tensor(np.stack(row['x_seq_' + f].to_numpy()[0], axis=0), dtype=torch.float32))
            try:
                y.append(torch.tensor(row['y_seq_' + f].to_numpy(), dtype=torch.float32))
            except:
                y.append(torch.tensor(np.stack(row['y_seq_' + f].to_numpy()[0], axis=0), dtype=torch.float32))
                # print(f"Failed for Y: {f}")
                # print(row['y_seq_' + f].to_numpy().shape)
                # print(row['y_seq_' + f].to_numpy(dtype=float)[0][0].shape)
        # Single batch size
        X_dec = [y[i] for i in range(2, 6)]

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
        return (X, X_dec), y


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