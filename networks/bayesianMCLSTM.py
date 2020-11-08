import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
Temporal Network with Monte-Carlo Dropout Bayesian Technique
"""


class BayesianMultiLSTM(nn.Module):
    def __init__(self, args, n_features, output_length):
        super(BayesianMultiLSTM, self).__init__()

        self.hidden_size_1 = args.lstm1_dim
        self.hidden_size_2 = args.lstm2_dim
        self.n_layers = 1  # number of (stacked) LSTM layers

        self.lstm1 = nn.LSTM(n_features,
                             self.hidden_size_1,
                             num_layers=1,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=1,
                             batch_first=True)

        self.dense = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=0.5, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=0.5, training=True)
        output = self.dense(output)

        return output

    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
        return hidden_state, cell_state

    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_2))
        cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_2))
        return hidden_state, cell_state

    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)

    def predict(self, X):
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()
