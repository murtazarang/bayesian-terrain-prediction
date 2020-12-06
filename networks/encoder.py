import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from networks.ConvLSTM import ConvLSTMCell


class LSTMEncoder(nn.Module):
    def __init__(self, args, n_features):
        super().__init__()

        self.args = args
        self.hidden_size = args.lstm_dim
        self.n_layers = 1  # number of (stacked) LSTM layers
        # n_features: Do a concat of all the 1D features
        self.lstm = nn.LSTM(n_features,
                             self.hidden_size,
                             num_layers=1,
                             batch_first=True)

    def forward(self, x):
        hidden = self.init_hidden(self.args.batch_size)
        output, hidden = self.lstm(x, hidden)

        return hidden

    def init_hidden(self, batch_size):
        hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))

        return hidden_state, cell_state


# Need to loop this over
class ConvLSTMEncoder(nn.Module):
    def __init__(self, args, n_features):
        super().__init__()
        self.n_layers = 1  # number of (stacked) LSTM layers
        self.args = args
        self.convlstm_c = self.args.preenc_t
        if self.args.use_add_features:
            self.convlstm_c += 1

        # define batch_size, channels, height, width, target_hidden_state
        self.c, self.t, self.h, self.w, self.add_linear_dim = n_features
        self.convlstm1 = ConvLSTMCell(self.args, self.convlstm_c, self.args.preenc_t, kernel_size=(3, 3), bias=True)
        if self.args.twolayer_convlstm:
            self.convlstm2 = ConvLSTMCell(self.args, self.args.out_seq_len, self.args.out_seq_len, kernel_size=(3, 3), bias=True)
        dil_val = [self.args.preenc_dil for i in range(self.args.n_enc_layers)]

        self.preconv = nn.ModuleList([nn.Conv2d(self.c, self.args.preenc_t, kernel_size=self.args.preenc_kernel,
                            dilation=dil_val[i], stride=self.args.preenc_str, padding=self.args.preenc_pad)
                                    for i in range(self.args.n_enc_layers)])

        # ToDo Make sure this linear dim is automated
        if self.args.use_add_features:
            self.add_out_layer = nn.Linear(self.add_linear_dim, 14)

    def forward(self, x, hidden1, hidden2, x_lin=None):
        # print(f'EncIn: {x.shape}')
        preenc_out = []
        for i, enc_conv in enumerate(self.preconv):
            # print(f'{x.shape} for enc cnn {i}')
            x = enc_conv(x)
            x = F.relu(x)
            if self.args.use_bayes_inf:
                x = F.dropout2d(x, p=self.args.preenc_out_droprate, training=True)
                x *= (1.0 - self.args.preenc_out_droprate) ** (-1.0)
            # print(f'Skip adding {x.shape}')
            preenc_out.append(x)

        # print(f'PreEncOut: {x.shape}')

        if self.args.use_add_features:
            add_out = self.add_out_layer(x_lin)
            # print(f'LinOut: {add_out.shape}')
            add_out = torch.reshape(add_out, (-1, 1, x.shape[-2], x.shape[-1]))
            # print(f'LinReshape: {add_out.shape}')
            x = torch.cat([x, add_out], dim=1)
            # print(f'LinConcat: {add_out.shape}')

        h_t1, c_t1 = self.convlstm1(x, hidden1)
        if self.args.twolayer_convlstm:
            hidden2 = self.convlstm2(h_t1, hidden2)
            # print(f'Encoder - Conv1: {h_t1.shape}, Conv2: {h_t2.shape}')

        return (h_t1, c_t1), hidden2, preenc_out

