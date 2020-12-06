import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from networks.ConvLSTM import ConvLSTMCell


class EncoderDecoderWrapper3d(nn.Module):
    def __init__(self, args, encoder, decoder_cell, feature_list, n_features=None, decoder_input=True):
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.args = args
        self.decoder_input = decoder_input
        self.device = self.args.device
        self.feature_list = feature_list

        if self.args.model in ['3d', '3D']:
            self.c, self.t, self.h, self.w, self.add_linear_dim = n_features  # 128, 2, 100, 100
            dil_val = [self.args.preenc_dil for i in range(self.args.n_enc_layers)]

            self.c1 = 16
            self.c2 = 16
            self.c3d = 16
            if self.args.use_skip_conn:
                self.c3d += self.c

            self.convlstm2d_enc = nn.ModuleList([
                ConvLSTMCell(self.args, self.c, self.c1, kernel_size=(5, 5), bias=True),
                ConvLSTMCell(self.args, self.c1, self.c2, kernel_size=(5, 5), bias=True)
                ])

            self.convlstm2d_dec = nn.ModuleList([
                ConvLSTMCell(self.args, self.c2, self.c2, kernel_size=(5, 5), bias=True),
                ConvLSTMCell(self.args, self.c2, self.c2, kernel_size=(1, 1), bias=True)
            ])

            self.batch_norm2d_enc = nn.ModuleList([
                nn.BatchNorm2d(self.c1),
                nn.BatchNorm2d(self.c2)
                ])

            self.batch_norm2d_dec = nn.ModuleList([
                nn.BatchNorm2d(self.c2),
                nn.BatchNorm2d(self.c2)
            ])

            self.conv3d_dec = nn.Conv3d(self.c3d, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def autoencoder(self, x, x_dec, hidden_enc, hidden_dec):
        outputs = []

        for t in range(self.args.in_seq_len):
            x_enc_in = x[t]
            for i, convlstm_enc in enumerate(self.convlstm2d_enc):
                hidden_enc[i] = convlstm_enc(x_enc_in, hidden_enc[i])
                x_enc_in, _ = hidden_enc[i]
                x_enc_in = self.batch_norm2d_enc[i](x_enc_in)
                # x_enc_in = F.relu(x_enc_in)
                x_enc_in = F.dropout2d(x_enc_in, p=self.args.enc_droprate, training=True)

        x_dec_in = x_enc_in
        for t in range(self.args.out_seq_len):
            for i, convlstm_dec in enumerate(self.convlstm2d_dec):
                if self.args.use_skip_conn and i == len(self.convlstm2d_dec) - 1:
                    x_enc_skip_1, _ = hidden_enc[0]
                    x_dec_in = torch.add(x_dec_in, x_enc_skip_1)
                hidden_dec[i] = convlstm_dec(x_dec_in, hidden_dec[i])
                x_dec_in, _ = hidden_dec[i]
                x_dec_in = self.batch_norm2d_dec[i](x_dec_in)
                # x_dec_in = F.relu(x_dec_in)
                x_dec_in = F.dropout2d(x_dec_in, p=self.args.dec_droprate, training=True)

            x_dec_out = x_dec_in
            if self.args.use_skip_conn:
                x_dec_out = torch.cat((x_dec_out, x[-1]), 1)

            outputs.append(x_dec_out)

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.conv3d_dec(outputs)
        outputs = nn.Sigmoid()(outputs)
        outputs = torch.squeeze(outputs, 1)

        return outputs

    def forward(self, xb):
        # ['h_in' OR 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
        seq_outputs = []
        x_enc, x_dec = xb
        if self.args.model in ['1d', '1D']:
            x_enc_in = torch.cat((x_enc[0], x_enc[-3], x_enc[-2], x_enc[-1]), -1)
            x_dec_in = torch.cat((x_dec[-3], x_dec[-2], x_dec[-1]), -1)
            if self.args.use_yr_corr:
                x_enc_in = torch.cat((x_enc_in, x_enc[1]), -1)
                x_dec_in = torch.cat((x_dec_in, x_dec[1]), -1)

            x_enc = x_enc_in
            x_dec = x_dec_in

            # Already Time Major
            enc_out, enc_hidden = self.encoder(x_enc)
            y_prev = enc_out
            for i in range(self.args.out_seq_len):
                y_prev = torch.cat((y_prev, x_dec[:, i, :]), -1)
                output, enc_hidden = self.decoder_cell(y_prev, enc_hidden)
                y_prev = output
                seq_outputs.append(output.squeeze(1))

        # Convert to Time Major for 3D
        elif self.args.model in ['3d', '3D']:
            x_lin_dec_in = None
            x_lin_enc_in = None

            # Initialize hidden states
            hidden_enc = [None for i in range(len(self.convlstm2d_enc))]
            hidden_dec = [None for i in range(len(self.convlstm2d_dec))]

            if self.args.use_add_features:
                x_lin_enc = torch.cat([x_enc[-3], x_enc[-2], x_enc[-1]], dim=-1)
                x_lin_dec = torch.cat([x_dec[-3], x_dec[-2], x_dec[-1]], dim=-1)
                x_lin_enc = torch.transpose(x_lin_enc, 1, 0)
                x_lin_dec = torch.transpose(x_lin_dec, 1, 0)

            if self.args.use_yr_corr:
                x_enc = torch.stack((x_enc[0], x_enc[1]), -3)
                x_enc = torch.transpose(x_enc, 1, 0)
                x_dec = torch.transpose(x_dec[0], 1, 0)
                x_dec = x_dec.unsqueeze(-3)
            else:
                # print(f'Before reshape: {x_enc[1].shape}')
                x_enc = torch.transpose(x_enc[0], 1, 0)
                # print(f'After time major: {x_enc.shape}')
                x_enc = x_enc.unsqueeze(-3)
                # print(f'After new dim: {x_enc.shape}')

            seq_outputs = self.autoencoder(x_enc, x_dec, hidden_enc, hidden_dec)

        return seq_outputs