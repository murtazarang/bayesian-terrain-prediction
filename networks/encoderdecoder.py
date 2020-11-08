import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderDecoderWrapper(nn.Module):
    def __init__(self, args, encoder, decoder_cell, decoder_input=True):
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.args = args
        self.decoder_input = decoder_input
        self.device = self.args.device

    def forward(self, xb):
        # ['h_in', 'log_h_in', 'h_yearly_corr', 'day_of_year_cos', 'day_of_year_sin', 'year_mod']
        seq_outputs = []
        if self.args.model in ['1d', '1D']:
            x_enc, x_dec = xb
            enc_out, enc_hidden = self.encoder(x_enc)
            y_prev = x_enc[:, -1, :].unsqueeze(1)
            for i in range(self.args.out_seq_len):
                y_prev = torch.cat((y_prev, x_dec[:, i, :]), -1)
                output, enc_hidden = self.decoder_cell(y_prev, enc_hidden)
                y_prev = output
                seq_outputs.append(output.squeeze(1))
        elif self.args.model in ['3d', '3D']:
            x_enc = xb
            prev_hidden = None
            for i in range(self.args.in_seq_len):
                prev_hidden = self.encoder(x_enc[:, i, :], prev_hidden)
            output = x_enc[:, -1, :]
            for i in range(self.args.out_seq_len):
                output, prev_hidden = self.decoder_cell(output, prev_hidden)
                seq_outputs.append(output.squeeze(1))
        return seq_outputs