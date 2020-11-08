import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNNEncoder(nn.Module):
    def __init__(self, args, n_features, output_length):
        super().__init__()

        self.hidden_size_1 = args.lstm1_dim
        self.hidden_size_2 = args.lstm2_dim
        self.n_layers = 1  # number of (stacked) LSTM layers

        self.lstm1 = nn.LSTM(n_features,
                             self.hidden_size_1,
                             num_layers=1,
                             batch_first=True)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden = self.init_hidden1(batch_size)
        output, hidden = self.lstm1(x, hidden)

        return output, hidden

    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))

        return hidden_state, cell_state


class DecoderCell(nn.Module):
    def __init__(self, args, n_features, output_length):
        super().__init__()

        self.hidden_size_1 = args.lstm1_dim
        self.n_layers = 1  # number of (stacked) LSTM layers

        self.lstm2 = nn.LSTM(n_features,
                             self.hidden_size_1,
                             num_layers=1,
                             batch_first=True)

        self.out = nn.Linear(args.lstm1_dim, 1)

    def forward(self, x, y, prev_hidden):
        batch_size, seq_len, _ = x.size() #ToDo, fix this
        # hidden = self.init_hidden2(batch_size)
        output, hidden = self.lstm2(x, prev_hidden)
        output = self.out(hidden)

        return output, hidden

    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size_1))

        return hidden_state, cell_state


class EncoderDecoderWrapper(nn.Module):
    def __init__(self, args, encoder, decoder_cell, decoder_input=True, device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.args = args
        # self.output_size = output_size
        # self.teacher_forcing = teacher_forcing
        # self.sequence_length = sequence_len
        self.decoder_input = decoder_input
        self.device = device

    def forward(self, xb):
        # ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
        x_enc, x_dec = xb
        # ToDo write code here to get the features concatenated for network use
        enc_out, enc_hidden = self.encoder(x_enc)
        y_prev = x_enc[:, -1, 0].unsqueeze(1)
        seq_outputs = []
        for i in range(self.output_seq):
            output, enc_hidden = self.decoder_cell(y_prev, x_dec, enc_hidden)
            y_prev = output
            seq_outputs.append(output.squeeze(1))

        return seq_outputs

    def forward(self, xb, yb=None):
        if self.decoder_input:
            decoder_input = xb[-1]
            input_seq = xb[0]
            if len(xb) > 2:
                encoder_output, encoder_hidden = self.encoder(input_seq, *xb[1:-1])
            else:
                encoder_output, encoder_hidden = self.encoder(input_seq)
        else:
            if type(xb) is list and len(xb) > 1:
                input_seq = xb[0]
                encoder_output, encoder_hidden = self.encoder(*xb)
            else:
                input_seq = xb
                encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden
        outputs = torch.zeros(input_seq.size(0), self.output_size, device=self.device)
        y_prev = input_seq[:, -1, 0].unsqueeze(1)
        for i in range(self.output_size):
            step_decoder_input = torch.cat((y_prev, decoder_input[:, i]), axis=1)
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                step_decoder_input = torch.cat((yb[:, i].unsqueeze(1), decoder_input[:, i]), axis=1)
            rnn_output, prev_hidden = self.decoder_cell(prev_hidden, step_decoder_input)
            y_prev = rnn_output
            outputs[:, i] = rnn_output.squeeze(1)

        return outputs