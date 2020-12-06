import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderDecoderWrapper(nn.Module):
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
            if self.args.use_skip_conn:
                self.c = 2
            self.dec_3d_conv = nn.Conv3d(self.args.out_seq_len, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            dil_val = [self.args.preenc_dil for i in range(self.args.n_enc_layers)]
            self.postconv = nn.ModuleList(
                [nn.ConvTranspose2d(self.c, self.args.preenc_t, kernel_size=self.args.preenc_kernel,
                                    dilation=dil_val[0], stride=self.args.preenc_str, padding=self.args.preenc_pad),
                 nn.ConvTranspose2d(self.c, self.args.preenc_t, kernel_size=self.args.preenc_kernel,
                                    dilation=dil_val[1], stride=self.args.preenc_str,
                                    padding=self.args.preenc_pad, output_padding=1),
                 nn.ConvTranspose2d(self.c, self.args.preenc_t, kernel_size=self.args.preenc_kernel,
                                    dilation=dil_val[2], stride=self.args.preenc_str,
                                    padding=self.args.preenc_pad, output_padding=1)
                 ])

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

            # x_enc = torch.cat((x_enc[1], x_enc[3], x_enc[4], x_enc[5]), -1)
            # x_dec = torch.cat((x_dec[3], x_dec[4], x_dec[5]), -1)
            # else:
            #     x_enc = torch.cat((x_enc[0], x_enc[2], x_enc[3], x_enc[4], x_enc[5]), -1)
            #     x_dec = torch.cat((x_enc[2], x_dec[3], x_dec[4], x_dec[5]), -1)
            # Already Time Major
            enc_out, enc_hidden = self.encoder(x_enc)
            y_prev = enc_out
            for i in range(self.args.out_seq_len):
                y_prev = torch.cat((y_prev, x_dec[:, i, :]), -1)
                output, enc_hidden = self.decoder_cell(y_prev, enc_hidden)
                y_prev = output
                seq_outputs.append(output.squeeze(1))

        # Convert to Time Major
        elif self.args.model in ['3d', '3D']:
            x_lin_dec_in = None
            x_lin_enc_in = None
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

            enc_hidden1 = None
            enc_hidden2 = None

            # Encoder operations
            for i in range(self.args.in_seq_len):
                if self.args.use_add_features:
                    x_lin_enc_in = x_lin_enc[i]
                enc_hidden1, enc_hidden2, skip_enc_out = self.encoder(x_enc[i], enc_hidden1, enc_hidden2, x_lin_enc_in)

            # output = x_enc[-1]
            # Get the last out of the encoder operation
            if self.args.twolayer_convlstm:
                dec_in, _ = enc_hidden2
            else:
                dec_in, _ = enc_hidden1
            # Decoder operations
            # print(f'Encoder Output {output.shape}')
            dec_hidden1 = None
            dec_hidden2 = None
            skip_enc_out = list(reversed(skip_enc_out))
            if not self.args.use_skip_conn:
                skip_enc_out = None

            # print(f'Enc In to Dec: {dec_in.shape}')
            for i in range(self.args.out_seq_len):
                if self.args.use_add_features:
                    x_lin_dec_in = x_lin_dec[i]
                # print(f'Dec In for {i} : {dec_in.shape}')
                dec_hidden1, dec_hidden2, dec_output = self.decoder_cell(dec_in, dec_hidden1, dec_hidden2, x_lin_dec_in, skip_enc_out)
                if self.args.twolayer_convlstm:
                    dec_in, _ = dec_hidden2
                    # print(f'Output Shape: {output.shape}')
                    # seq_outputs.append(output.squeeze(1))
                    seq_outputs.append(dec_in)
                else:
                    dec_in, _ = dec_hidden1
                    seq_outputs.append(dec_output.squeeze(1))

                if self.args.use_yr_corr:
                    dec_in = torch.cat((dec_in, x_dec[i]), -3)

            seq_outputs = torch.stack(seq_outputs, 1)

            if self.args.twolayer_convlstm:
                # print(f'Shape of hidden stack: {seq_outputs.shape}')
                seq_outputs = seq_outputs.permute(0, 2, 1, 3, 4)
                # print(f'Shape of after permute: {seq_outputs.shape}')
                seq_outputs = self.dec_3d_conv(seq_outputs)
                # print(f'Shape after 3d conv: {seq_outputs.shape}')
                seq_outputs = seq_outputs.permute(2, 0, 1, 3, 4)

                # Regenerate the output after sequencing
                output = []
                for t in range(self.args.out_seq_len):
                    deconv_in = seq_outputs[t]
                    for i, dec_conv in enumerate(self.postconv):
                        if self.args.use_skip_conn:
                            # print(f'{deconv_in.shape} for dec cnn {i} and skip: {skip_enc_out[i].shape}')
                            deconv_in = torch.cat([deconv_in, skip_enc_out[i]], dim=1)
                        deconv_in = dec_conv(deconv_in)
                        if i != self.args.n_enc_layers - 1:
                            deconv_in = F.relu(deconv_in)
                            if self.args.use_bayes_inf:
                                deconv_in = F.dropout2d(deconv_in, p=self.args.preenc_out_droprate, training=True)
                                deconv_in *= (1.0 - self.args.preenc_out_droprate) ** (-1.0)
                    output.append(deconv_in)
                output = torch.cat(output, 1)
            else:
                output = seq_outputs
            # print(f'final output shape: {output.shape}')

        return output
