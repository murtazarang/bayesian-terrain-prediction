import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


"""
Spatio-Temporal Network with Monte-Carlo Dropout Bayesian Technique
"""

# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """
    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

    def predict(self, X):
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()


class ConvLSTMEncoder(nn.Module):
    def __init__(self, channel=1, input_feature_len=1, sequence_len=168, hidden_size=100, device='cpu', rnn_dropout=0.2):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.target_hidden_size = 1

        # self.num_layers = rnn_num_layers
        # self.convlstm = ConvLSTMCell(input_size, )
        # self.gru = nn.GRU(
        #     num_layers=rnn_num_layers,
        #     input_size=input_feature_len,
        #     hidden_size=hidden_size,
        #     batch_first=True,
        #     #bidirectional=bidirectional,
        #     dropout=rnn_dropout
        # )
        self.convlstm = ConvLSTMCell(channel, self.target_hidden_size)

        self.device = device

    def forward(self, input_seq):
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device=self.device)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        print(gru_out.shape)
        print(hidden.shape)
        if self.rnn_directions * self.num_layers > 1:
            num_layers = self.rnn_directions * self.num_layers
            if self.rnn_directions > 1:
                gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
                gru_out = torch.sum(gru_out, axis=2)
            hidden = hidden.view(self.num_layers, self.rnn_directions, input_seq.size(0), self.hidden_size)
            if self.num_layers > 0:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
            hidden = hidden.sum(axis=0)
        else:
            hidden.squeeze_(0)
        return gru_out, hidden


class ConvLSTMDecoderCell(nn.Module):
    def __init__(self, input_feature_len, hidden_size, dropout=0.2):
        super().__init__()
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=input_feature_len,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prev_hidden, y):
        rnn_hidden = self.decoder_rnn_cell(y, prev_hidden)
        output = self.out(rnn_hidden)
        return output, self.dropout(rnn_hidden)


class ConvLSTMEncoderDecoderWrapper(nn.Module):
    def __init__(self, encoder, decoder_cell, output_size=3, teacher_forcing=0.3, sequence_len=336, decoder_input=True, device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.output_size = output_size
        self.teacher_forcing = teacher_forcing
        self.sequence_length = sequence_len
        self.decoder_input = decoder_input
        self.device = device

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


def _main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels (time, and height), height, width
    b, c, h, w = 128, 1, 256, 256
    target = 1           # hidden state size
    lr = 1e-1       # learning rate
    T = 30           # sequence length
    max_epoch = 2  # number of epochs

    # set manual seed
    torch.manual_seed(0)

    print('Instantiate model')
    model = ConvLSTMCell(c, target)
    print(repr(model))

    print('Create input and target Variables')
    x = Variable(torch.rand(T, b, c, h, w))
    y = Variable(torch.randn(T, b, target, h, w))

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        state = None
        loss = 0
        for t in range(0, T):
            state = model(x[t], state)
            loss += loss_fn(state[0], y[t])

        print(' > Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss.item()))

        # zero grad parameters
        model.zero_grad()

        # compute new grad parameters through time!
        loss.backward()

        # learning_rate step against the gradient
        for p in model.parameters():
            p.data.sub_(p.grad.data * lr)

    print('Input size:', list(x.data.size()))
    print('Target size:', list(y.data.size()))
    print('Last hidden state size:', list(state[0].size()))


if __name__ == '__main__':
    _main()