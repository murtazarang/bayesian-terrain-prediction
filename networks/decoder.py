import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from networks.ConvLSTM import ConvLSTMCell

# Define some constants
KERNEL_SIZE = 5
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
        output = F.dropout(output, p=self.args.dec_out_droprate, training=True)
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
        self.convlstm_c = self.args.preenc_t
        if self.args.use_add_features:
            self.convlstm_c += 1

        # define batch_size, channels, height, width, target_hidden_state
        # c, t = n_features  # 128, 2, 100, 100
        self.c, self.t, self.h, self.w, self.add_linear_dim = n_features  # 128, 2, 100, 100
        # if self.args.use_skip_conn:
        #     self.c = 2
        # self.convlstm = ConvLSTMCell(self.args, self.convlstm_c, self.convlstm_c)
        self.convlstm1 = ConvLSTMCell(self.args, self.args.preenc_t, self.c, kernel_size=(3, 3), bias=True)
        if self.args.twolayer_convlstm:
            self.convlstm2 = ConvLSTMCell(self.args, self.args.out_seq_len, self.args.out_seq_len, kernel_size=(3, 3), bias=True)
        dil_val = [self.args.preenc_dil for i in range(self.args.n_enc_layers)]
        # self.preconv = nn.ModuleList([nn.Conv2d(self.c, self.args.preenc_t, kernel_size=self.args.preenc_kernel,
        #                     dilation=dil_val[i], stride=self.args.preenc_str, padding=self.args.preenc_pad)
#                             for i in range(self.args.n_enc_layers)])
#         self.preconv = nn.Conv2d(self.c, self.args.preenc_t, kernel_size=self.args.preenc_kernel,
#                                  dilation=self.args.preenc_dil, stride=self.args.preenc_str,
#                                  padding=self.args.preenc_pad)
        # for enc_conv in self.pre_conv:
        self.c_t = self.c
        if self.args.use_skip_conn:
            self.c_t = 2
        self.postconv = nn.ModuleList([nn.ConvTranspose2d(self.c_t, self.args.preenc_t, kernel_size=self.args.preenc_kernel,
                            dilation=dil_val[0], stride=self.args.preenc_str, padding=self.args.preenc_pad),
                           nn.ConvTranspose2d(self.c, self.args.preenc_t, kernel_size=self.args.preenc_kernel,
                                                          dilation=dil_val[1], stride=self.args.preenc_str,
                                                          padding=self.args.preenc_pad, output_padding=1),
                           nn.ConvTranspose2d(self.c_t, self.args.preenc_t, kernel_size=self.args.preenc_kernel,
                                                          dilation=dil_val[2], stride=self.args.preenc_str,
                                                          padding=self.args.preenc_pad, output_padding=1)
                           ])
                            # for i in reversed(range(self.args.n_enc_layers))])
        # self.preconvtran = nn.ModuleList([nn.ConvTranspose2d(self.convlstm_c, 1, kernel_size=self.args.dec_kernel,
        #                          dilation=self.args.dec_dil, stride=self.args.dec_str,
        #                          padding=self.args.dec_pad) for i in range(self.args.n_enc_layers)])

        # self.out = nn.ConvTranspose2d(self.convlstm_c, 1, kernel_size=self.args.dec_kernel,
        #                          dilation=self.args.dec_dil, stride=self.args.dec_str,
        #                          padding=self.args.dec_pad)
        # self.out = nn.Conv2d(self.t, self.t, KERNEL_SIZE, padding=PADDING)
        # ToDo Make sure this linear dim is automated
        if self.args.use_add_features:
            self.add_out_layer = nn.Linear(self.add_linear_dim, 14)

    def forward(self, x, hidden1, hidden2, x_lin=None, skip_x=None):
        # print(f'DecIn: {x.shape}')
        # x = self.preconv(x)
        # x = F.relu(x)
        # if self.args.use_bayes_inf:
        #     x = F.dropout2d(x, p=self.args.preenc_out_droprate, training=True)
        #     x *= (1.0 - self.args.preenc_out_droprate) ** (-1.0)
        # print(f'PreDecOut: {x.shape}')
        if self.args.use_add_features:
            add_out = self.add_out_layer(x_lin)
            # print(f'LinOut: {add_out.shape}')
            add_out = F.relu(add_out)
            add_out = torch.reshape(add_out, (-1, 1, x.shape[-2], x.shape[-1]))
            # print(f'LinReshape: {add_out.shape}')
            x = torch.cat([x, add_out], dim=1)
            # print(f'LinConcat: {add_out.shape}')
        h_t1, c_t1 = self.convlstm1(x, hidden1)
        if self.args.twolayer_convlstm:
            hidden2 = self.convlstm2(h_t1, hidden2)
            output = None
        else:
            # print(f'Decoder - Conv1: {h_t1.shape}')
            output = h_t1
            # Regenerate the output after sequencing
            for i, dec_conv in enumerate(self.postconv):
                if self.args.use_skip_conn:
                    # print(f'Size before skip: {output.shape} and skip: {skip_x[i].shape}')
                    if i in self.args.skip_layers:
                        output = torch.cat([output, skip_x[i]], dim=1)
                        # print(f'Size after skip: {output.shape}')
                output = dec_conv(output)
                # print(f'Size after deconv: {output.shape}')
                if i != self.args.n_enc_layers - 1:
                    output = F.relu(output)
                    if self.args.use_bayes_inf:
                        output = F.dropout2d(output, p=self.args.preenc_out_droprate, training=True)
                        output *= (1.0 - self.args.preenc_out_droprate) ** (-1.0)
                else:
                    output = F.sigmoid(output)

        return (h_t1, c_t1), hidden2, output