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