import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from networks.ConvLSTM import ConvLSTMCell

# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2


class LSTMDecoder(nn.Module):
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

        self.out = nn.Linear(args.linear_dim, 1)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        output = self.out(output)

        return output, hidden

    def init_hidden(self, batch_size):
        hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))

        return hidden_state, cell_state


# Need to loop this over
class ConvLSTMDecoder(nn.Module):
    def __init__(self, args, n_features):
        super().__init__()
        self.n_layers = 1  # number of (stacked) LSTM layers
        self.args = args

        # define batch_size, channels, height, width, target_hidden_state
        c, t = n_features  # 128, 2, 100, 100
        self.convlstm = ConvLSTMCell(c, t)
        self.out = nn.Conv2d(c, c, KERNEL_SIZE, padding=PADDING)

    def forward(self, x, hidden):
        hidden = self.convlstm(x, hidden)
        output = self.out(hidden)

        return output, hidden