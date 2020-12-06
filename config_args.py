import argparse


def parse_args():
    parser = argparse.ArgumentParser("TimeSeries")

    parser.add_argument("--exp_name", type=str, default='skip_in_convlst_10_10')
    parser.add_argument("--lr_search", default=False, action='store_true')
    parser.add_argument("--train_network", default=False, action='store_true')

    parser.add_argument("--device", type=str, default="cuda", choices=['cuda', 'cpu'])

    parser.add_argument("--model", type=str, default="3D", choices=["1D", "1d", "3D", "3d"])

    parser.add_argument("--num_features", type=int, default=10000)
    parser.add_argument("--xdim", type=int, default=100)
    parser.add_argument("--ydim", type=int, default=100)

    parser.add_argument("--in_seq_len", type=int, default=10)
    parser.add_argument("--out_seq_len", type=int, default=10)

    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.32E-05)
    parser.add_argument("--lr_decay", type=float, default=1.0E-06)

    parser.add_argument("--train_split", type=int, default=8258)

    # 1D Bayesian LSTM Architecture
    parser.add_argument("--lstm1_dim", type=int, default=128)
    parser.add_argument("--lstm2_dim", type=int, default=32)

    # 2D ConvLSTM Architecture
    parser.add_argument("--preenc_t", type=int, default=1)
    parser.add_argument("--preenc_kernel", type=int, default=3)
    parser.add_argument("--preenc_dil", type=int, default=2)
    parser.add_argument("--preenc_pad", type=int, default=2)
    parser.add_argument("--preenc_str", type=int, default=2)
    parser.add_argument("--n_enc_layers", type=int, default=3)

    # 2D ConvLSTM Architecture
    parser.add_argument("--dec_kernel", type=int, default=5)
    parser.add_argument("--dec_dil", type=int, default=13)
    parser.add_argument("--dec_pad", type=int, default=1)
    parser.add_argument("--dec_str", type=int, default=1)

    # Bayesian Uncertainity Parameters
    parser.add_argument("--use_bayes_inf", default=True, action='store_true')
    parser.add_argument("--enc_droprate", type=float, default=0.1)
    parser.add_argument("--dec_droprate", type=float, default=0.1)
    parser.add_argument("--n_samples", type=int, default=100)

    # Data pre-processing methods
    parser.add_argument("--no_cylical_dates", default=False, action='store_true')
    parser.add_argument("--height_correction", type=str, default='log', choices=['log', 'autocorrelation'])
    parser.add_argument("--training_mode", type=str, default="train", choices=["test", "train"])
    parser.add_argument("--testing_start_date", type=str, default='2008-01-01')
    parser.add_argument("--testing_end_date", type=str, default='2009-12-31')
    parser.add_argument("--validation_start_date", type=str, default='2007-01-01')

    # Data Processing Requirements
    parser.add_argument("--load_data", default=False, action='store_true')
    parser.add_argument("--compress_data", default=False, action='store_true')
    parser.add_argument("--sequence_data", default=False, action='store_true')
    parser.add_argument("--sequence_to_np", default=False, action='store_true')
    parser.add_argument("--dataset", type=str, default='_Fertilizer3dAnnual')

    # Features
    parser.add_argument("--use_log_h", default=False, action='store_true')
    parser.add_argument("--use_add_features", default=False, action='store_true')
    parser.add_argument("--use_yr_corr", default=False, action='store_true')
    parser.add_argument("--use_skip_conn", default=True, action='store_true')
    parser.add_argument("--twolayer_convlstm", default=False, action='store_true')
    parser.add_argument("--skip_layers", type=list, default=[0, 2])

    # Model type
    parser.add_argument("--use3d_autoencoder", default=True, action='store_true')

    # Test Time

    return parser.parse_args()
